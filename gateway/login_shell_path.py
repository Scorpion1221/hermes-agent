"""Merge the user's login-shell PATH into a service-launched gateway.

systemd and launchd start the gateway with the PATH written into the unit / plist: Hermes and
system directories, plus (launchd) a snapshot of whichever shell last ran ``hermes gateway
install``. Directories only the user's rc files add (nvm, pyenv, asdf, cargo, Linuxbrew, ...) are
missing, so a CLI installed there is ``command not found`` for the bot until the unit is
regenerated from the right shell — and an update run from a plain SSH session drops them again.

At startup a gateway that a Hermes service started runs the user's shell once as an interactive
login shell and reads ``$PATH`` between sentinel markers so profile banners can't corrupt it (the
desktop app's ``apps/desktop/electron/shell-path.ts`` does the same). Unlike the desktop app,
login-only entries are APPENDED: whatever the unit resolves today keeps priority, so the PM-managed
runtimes Hermes provisions for itself still win over the user's copies of node/npm/uv. The probe
gets a login-like environment without the gateway's secrets, runs in its own session and is bounded
by one timeout; any failure leaves PATH as the unit set it. ``gateway.login_shell_path: false`` in
config.yaml turns it off.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
from collections.abc import Mapping, MutableMapping
from typing import Any

from gateway.restart import EXTERNAL_GATEWAY_SUPERVISOR_ENV, launchd_job_label
from utils import is_truthy_value

logger = logging.getLogger(__name__)

_PATH_START = "__HERMES_LOGIN_PATH_START__"
_PATH_END = "__HERMES_LOGIN_PATH_END__"
# Valid in sh-family shells and fish alike: fish rejects ``${PATH}`` and joins a quoted "$PATH" with ':'.
_PROBE_COMMAND = f"printf '%s%s%s' {_PATH_START} \"$PATH\" {_PATH_END}"
ATTEMPT_TIMEOUT_S = 5.0
# What a fresh login carries. The gateway's own environment holds the secrets loaded from .env and
# its supervisor markers; neither belongs in the user's rc files.
_PROBE_ENV_KEYS = ("HOME", "USER", "LOGNAME", "SHELL", "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR")
# Set only by Hermes' own launchers (systemd unit, launchd plist, s6 run script) and the
# --external-supervisor opt-in. Raw XPC_SERVICE_NAME and INVOCATION_ID are not markers here: IDE and
# iTerm terminals export an ``application.*`` XPC label, systemd-spawned terminals and CI runners
# inherit INVOCATION_ID, and a gateway run from those already has the user's shell PATH.
_LAUNCH_MARKERS = ("HERMES_SUPERVISED_CHILD", "HERMES_S6_SUPERVISED_CHILD", EXTERNAL_GATEWAY_SUPERVISOR_ENV)


class _ProbeTimeout(Exception):
    """The probe ran out of time before printing the sentinel."""


def started_by_hermes_service(env: Mapping[str, str]) -> bool:
    """A Hermes launch marker, or the ``ai.hermes.*`` launchd label (plists that predate the markers)."""
    return bool(launchd_job_label(env)) or any(is_truthy_value(env.get(name)) for name in _LAUNCH_MARKERS)


def login_shell_executable(env: Mapping[str, str]) -> str:
    """The user's login shell: ``$SHELL``, then the passwd entry (launchd sets no ``$SHELL``)."""
    shell = (env.get("SHELL") or "").strip()
    if not shell:
        try:
            import pwd

            shell = pwd.getpwuid(os.getuid()).pw_shell  # windows-footgun: ok — only reached on POSIX
        except (ImportError, KeyError, OSError):
            shell = ""
    return shell or ("/bin/zsh" if sys.platform == "darwin" else "/bin/bash")


def extract_sentinel_path(stdout: str) -> str | None:
    """``$PATH`` from between the LAST start marker and the next end marker, so a profile that
    echoes the environment (or the command line itself) can't poison the capture."""
    start = stdout.rfind(_PATH_START)
    if start == -1:
        return None
    value_start = start + len(_PATH_START)
    end = stdout.find(_PATH_END, value_start)
    if end == -1:
        return None
    return stdout[value_start:end].strip() or None


def login_only_entries(current: str, login: str) -> list[str]:
    """Absolute *login* entries missing from *current*, in login order. Relative entries (``.``,
    ``bin``) are dropped: they would resolve against the gateway's cwd, not the user's shell."""
    present = set(current.split(os.pathsep))
    return list(dict.fromkeys(
        entry for entry in login.split(os.pathsep) if entry and os.path.isabs(entry) and entry not in present))


def _run_probe(shell: str, flag: str, env: Mapping[str, str], timeout: float) -> str | None:
    try:
        proc = subprocess.Popen(
            [shell, flag, _PROBE_COMMAND], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env={key: env[key] for key in _PROBE_ENV_KEYS if env.get(key)},
            # Own session, so a hung rc and everything it started are killed together.
            start_new_session=True)
    except OSError:
        return None
    try:
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        try:
            stdout, _ = proc.communicate(timeout=1)
        except subprocess.TimeoutExpired as late:
            # A daemon the rc started outside the session still holds the pipe: keep what arrived.
            stdout = late.stdout or exc.stdout or b""
            proc.kill()
            proc.wait()
        # An rc can hang after the sentinel printed (an exit trap, a logout script).
        captured = extract_sentinel_path(stdout.decode("utf-8", "replace"))
        if captured:
            return captured
        raise _ProbeTimeout from None
    # A profile may exit nonzero after the sentinel printed: trust the sentinel, not the exit code.
    return extract_sentinel_path(stdout.decode("utf-8", "replace"))


def capture_login_shell_path(env: Mapping[str, str], timeout: float = ATTEMPT_TIMEOUT_S) -> str | None:
    """The PATH the user's login shell builds, or None."""
    if os.name != "posix":
        return None
    shell = login_shell_executable(env)
    # -i also sources ~/.zshrc / ~/.bashrc past their interactivity guards, where nvm-style managers
    # live. Some shells swallow a combined -ilc with a non-tty stdin, so fall back to a plain login
    # shell — but not after a timeout: the same rc would hang it too, and startup waits once at most.
    for flag in ("-ilc", "-lc"):
        try:
            captured = _run_probe(shell, flag, env, timeout)
        except _ProbeTimeout:
            return None
        if captured:
            return captured
    return None


def apply_login_shell_path(
    env: MutableMapping[str, str] | None = None, timeout: float = ATTEMPT_TIMEOUT_S,
) -> list[str] | None:
    """Append the login shell's PATH-only entries to *env* (default ``os.environ``).

    Returns the entries added, or None when the login shell's PATH could not be resolved (PATH is
    then left exactly as it was).
    """
    env = os.environ if env is None else env
    login_path = capture_login_shell_path(env, timeout)
    if not login_path:
        return None
    current = env.get("PATH", "")
    added = login_only_entries(current, login_path)
    if added:
        env["PATH"] = os.pathsep.join(filter(None, [current, *added]))
    return added


def apply_for_service(
    config: Mapping[str, Any] | None, env: MutableMapping[str, str] | None = None,
    timeout: float = ATTEMPT_TIMEOUT_S,
) -> list[str] | None:
    """Merge the login-shell PATH when a generated service/launcher started the gateway.

    A gateway run from a terminal already has that terminal's PATH. *config* is the launch
    profile's config.yaml (PATH is process-wide); ``gateway.login_shell_path: false`` disables it.
    """
    env = os.environ if env is None else env
    if not started_by_hermes_service(env):
        return None
    gateway_cfg = (config or {}).get("gateway")
    if isinstance(gateway_cfg, Mapping) and not is_truthy_value(
            gateway_cfg.get("login_shell_path", True), default=True):
        return None
    added = apply_login_shell_path(env, timeout)
    if added:
        logger.info("Added login-shell PATH entries: %s", os.pathsep.join(added))
    return added
