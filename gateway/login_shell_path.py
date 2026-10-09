"""Merge the user's login-shell PATH into a service-launched gateway.

systemd and launchd start the gateway with the PATH baked into the unit /
plist, which is a snapshot of whichever shell last ran ``hermes gateway
install`` or ``hermes update``. That shell often lacks the directories only
the user's rc files add (nvm, pyenv, asdf, cargo, ...): an update run from a
plain SSH session drops them, and every CLI the user installs there later is
missing for the bot until the unit is regenerated from the right shell.

Resolving the login shell's PATH once at gateway start makes the bot see what
the user's terminal sees, whichever shell wrote the unit. Same approach as
the desktop app (``apps/desktop/electron/shell-path.ts``): run the user's
shell as an interactive login shell, read ``$PATH`` between sentinel markers
so profile banners can't corrupt it, and bound every attempt by a timeout.

Entries already on PATH keep their order, so whatever the unit resolves
today (its venv, managed Node) still wins; login-shell-only entries are
appended. Any failure leaves PATH untouched.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from typing import Mapping, MutableMapping, Optional

_PATH_START = "__HERMES_LOGIN_PATH_START__"
_PATH_END = "__HERMES_LOGIN_PATH_END__"
_PROBE_COMMAND = f"printf '%s' \"{_PATH_START}${{PATH}}{_PATH_END}\""
ATTEMPT_TIMEOUT_S = 5.0


def login_shell_executable(env: Mapping[str, str]) -> str:
    """Return the user's login shell: ``$SHELL``, then the passwd entry."""
    shell = (env.get("SHELL") or "").strip()
    if shell:
        return shell
    # launchd does not set $SHELL for agents.
    try:
        import pwd

        shell = pwd.getpwuid(os.getuid()).pw_shell
    except (ImportError, KeyError, OSError):
        shell = ""
    if shell:
        return shell
    return "/bin/zsh" if sys.platform == "darwin" else "/bin/bash"


def extract_sentinel_path(stdout: str) -> Optional[str]:
    """Read ``$PATH`` from between the sentinel markers.

    Uses the LAST start marker so a profile that echoes the environment (or
    the command line itself) can't poison the capture with an earlier match.
    """
    text = stdout or ""
    start = text.rfind(_PATH_START)
    if start == -1:
        return None
    value_start = start + len(_PATH_START)
    end = text.find(_PATH_END, value_start)
    if end == -1:
        return None
    return text[value_start:end].strip() or None


def merge_path_entries(current: str, login: str) -> str:
    """Keep *current* entries in order, then append login-only entries.

    Relative login-shell entries (``.``, ``bin``) are dropped: they would
    resolve against the gateway's working directory, not the user's shell.
    """
    ordered = [entry for entry in current.split(os.pathsep) if entry] + [
        entry for entry in login.split(os.pathsep) if entry and os.path.isabs(entry)
    ]
    return os.pathsep.join(dict.fromkeys(ordered))


# What a fresh login carries. The gateway's own environment holds the
# secrets loaded from .env and its supervisor markers; neither belongs in the
# user's rc files.
_PROBE_ENV_KEYS = ("HOME", "USER", "LOGNAME", "SHELL", "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR")


def _probe_env(env: Mapping[str, str]) -> dict[str, str]:
    return {key: env[key] for key in _PROBE_ENV_KEYS if env.get(key)}


class _ProbeTimeout(Exception):
    pass


def _run_probe(shell: str, flag: str, env: Mapping[str, str], timeout: float) -> Optional[str]:
    try:
        proc = subprocess.Popen(
            [shell, flag, _PROBE_COMMAND],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=_probe_env(env),
            encoding="utf-8",
            errors="replace",
            # Own process group, so a hung rc and anything it spawned can be
            # killed together.
            start_new_session=True,
        )
    except OSError:
        return None
    try:
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            stdout, _ = proc.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout = ""
        # A profile can hang after the sentinel printed (an exit trap, a
        # logout script) — keep what it already wrote.
        captured = extract_sentinel_path(stdout)
        if captured:
            return captured
        raise _ProbeTimeout()
    # A profile may exit nonzero after the sentinel printed — trust the
    # sentinel, not the exit code.
    return extract_sentinel_path(stdout)


def capture_login_shell_path(
    env: Optional[Mapping[str, str]] = None,
    timeout: float = ATTEMPT_TIMEOUT_S,
) -> Optional[str]:
    """Return the PATH the user's login shell builds, or None."""
    if os.name != "posix":
        return None
    env = os.environ if env is None else env
    shell = login_shell_executable(env)
    # -i also sources ~/.zshrc / ~/.bashrc past their interactivity guards,
    # where nvm-style managers live. Some shells swallow combined -ilc with a
    # non-tty stdin (macOS system bash 3.2), so fall back to a plain login
    # shell — but not after a timeout: the same rc would hang it too, and
    # startup should wait for one timeout at most.
    for flag in ("-ilc", "-lc"):
        try:
            captured = _run_probe(shell, flag, env, timeout)
        except _ProbeTimeout:
            return None
        if captured:
            return captured
    return None


def apply_login_shell_path(
    env: Optional[MutableMapping[str, str]] = None,
    timeout: float = ATTEMPT_TIMEOUT_S,
) -> Optional[list[str]]:
    """Merge the login shell's PATH into *env* (default ``os.environ``).

    Returns the entries that were not on PATH before, or None when the login
    shell's PATH could not be resolved (PATH is then left as it was).
    """
    env = os.environ if env is None else env
    login_path = capture_login_shell_path(env, timeout)
    if not login_path:
        return None
    current = env.get("PATH", "")
    merged = merge_path_entries(current, login_path)
    before = set(current.split(os.pathsep))
    added = [entry for entry in merged.split(os.pathsep) if entry not in before]
    if merged != current:
        env["PATH"] = merged
    return added


def apply_for_service(
    config: Optional[Mapping[str, object]],
    env: Optional[MutableMapping[str, str]] = None,
    timeout: float = ATTEMPT_TIMEOUT_S,
) -> Optional[list[str]]:
    """Merge the login-shell PATH when a supervisor launched the gateway.

    A gateway run from a terminal already has that terminal's PATH. Only a
    systemd/launchd/external-supervisor launch runs on a baked snapshot.
    Disabled with ``gateway.login_shell_path: false`` in config.yaml.
    """
    from gateway.restart import is_gateway_supervisor_process
    from utils import is_truthy_value

    env = os.environ if env is None else env
    gateway_cfg = (config or {}).get("gateway") or {}
    if isinstance(gateway_cfg, Mapping) and not is_truthy_value(
        gateway_cfg.get("login_shell_path", True), default=True
    ):
        return None
    if not is_gateway_supervisor_process(env):
        return None
    return apply_login_shell_path(env, timeout)
