"""The gateway appends the user's login-shell PATH when a service/launcher starts it.

A systemd unit / launchd plist carries a fixed PATH (on macOS, the PATH of whichever shell last
wrote it), so a tool that only the user's rc files put on PATH (nvm, pyenv, cargo) was
``command not found`` for the bot. These tests run a real login shell against a temp HOME.
"""

from __future__ import annotations

import os
import shutil
import stat
import time
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig
from gateway.login_shell_path import apply_for_service, extract_sentinel_path, login_only_entries

pytestmark = pytest.mark.platforms("posix")  # POSIX login shells

BASH = shutil.which("bash") or "/bin/bash"
UNIT_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"


def _install_tool(directory, name="nvm-only-tool"):
    directory.mkdir(parents=True, exist_ok=True)
    tool = directory / name
    tool.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    return tool


def _service_env(home, **extra):
    """The environment a generated unit / plist hands the gateway."""
    env = {"HOME": str(home), "SHELL": BASH, "PATH": UNIT_PATH, "HERMES_SUPERVISED_CHILD": "1"}
    env.update(extra)
    return env


@pytest.fixture
def home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    return home


def test_tool_added_by_profile_becomes_resolvable_after_the_unit_entries(home):
    tool = _install_tool(home / ".nvm" / "versions" / "node" / "v22" / "bin")
    (home / ".profile").write_text(f'export PATH="{tool.parent}:$PATH"\n', encoding="utf-8")
    env = _service_env(home)

    added = apply_for_service({}, env, timeout=10)

    # /etc/profile may add system dirs too (macOS path_helper).
    assert str(tool.parent) in added
    assert shutil.which(tool.name, path=env["PATH"]) == str(tool)
    # The unit's own entries (its venv, PM-managed runtimes) keep priority.
    assert env["PATH"].startswith(UNIT_PATH + os.pathsep)


def test_profile_noise_does_not_corrupt_path(home):
    tool = _install_tool(home / "bin")
    (home / ".profile").write_text(
        f'echo "Welcome back!"\nexport PATH="{tool.parent}:$PATH"\necho "PATH is $PATH"\n', encoding="utf-8")
    env = _service_env(home)

    apply_for_service({}, env, timeout=10)

    assert shutil.which(tool.name, path=env["PATH"]) == str(tool)
    assert all(os.path.isabs(entry) for entry in env["PATH"].split(os.pathsep))


def test_failing_profile_leaves_unit_path(home):
    (home / ".profile").write_text("exit 3\n", encoding="utf-8")
    env = _service_env(home)

    assert apply_for_service({}, env, timeout=10) is None
    assert env["PATH"] == UNIT_PATH


def test_hanging_profile_is_probed_once_and_leaves_unit_path(home, tmp_path):
    runs = tmp_path / "runs.txt"
    (home / ".profile").write_text(f'echo run >> "{runs}"\nsleep 30\n', encoding="utf-8")
    env = _service_env(home)

    started = time.monotonic()
    assert apply_for_service({}, env, timeout=1) is None
    assert time.monotonic() - started < 6
    # Startup waits for one timeout at most: the plain-login fallback would hang on the same rc.
    assert runs.read_text(encoding="utf-8-sig").splitlines() == ["run"]
    assert env["PATH"] == UNIT_PATH


def test_profile_hanging_after_the_sentinel_still_yields_its_path(home):
    tool = _install_tool(home / "bin")
    (home / ".profile").write_text(f"trap 'sleep 30' EXIT\nexport PATH=\"{tool.parent}:$PATH\"\n",
                                   encoding="utf-8")
    env = _service_env(home)

    started = time.monotonic()
    added = apply_for_service({}, env, timeout=1)

    assert time.monotonic() - started < 6
    assert str(tool.parent) in added
    assert shutil.which(tool.name, path=env["PATH"]) == str(tool)


def test_gateway_secrets_and_markers_are_not_handed_to_rc_files(home, tmp_path):
    dump = tmp_path / "rc-env.txt"
    (home / ".profile").write_text(f'env > "{dump}"\n', encoding="utf-8")
    env = _service_env(home, OPENAI_API_KEY="sk-test-not-for-rc", FEISHU_APP_SECRET="s3cret")

    apply_for_service({}, env, timeout=10)

    seen = dump.read_text(encoding="utf-8-sig")
    assert "sk-test-not-for-rc" not in seen
    assert "s3cret" not in seen
    assert "HERMES_SUPERVISED_CHILD" not in seen


def test_terminal_launch_is_left_alone(home):
    tool = _install_tool(home / "bin")
    (home / ".profile").write_text(f'export PATH="{tool.parent}:$PATH"\n', encoding="utf-8")
    env = _service_env(home)
    del env["HERMES_SUPERVISED_CHILD"]

    assert apply_for_service({}, env, timeout=10) is None
    assert env["PATH"] == UNIT_PATH


@pytest.mark.parametrize("value", [False, "false", "off"])
def test_config_switch_disables_the_merge(home, value):
    tool = _install_tool(home / "bin")
    (home / ".profile").write_text(f'export PATH="{tool.parent}:$PATH"\n', encoding="utf-8")
    env = _service_env(home)

    assert apply_for_service({"gateway": {"login_shell_path": value}}, env, timeout=10) is None
    assert env["PATH"] == UNIT_PATH


def test_login_only_entries_are_absolute_new_and_in_login_order():
    added = login_only_entries("/venv/bin:/usr/bin:/bin", "/home/u/.nvm/bin:.:bin:/usr/bin:/home/u/.cargo/bin")
    assert added == ["/home/u/.nvm/bin", "/home/u/.cargo/bin"]


def test_sentinel_uses_the_last_marker():
    start, end = "__HERMES_LOGIN_PATH_START__", "__HERMES_LOGIN_PATH_END__"
    noisy = f"+ printf {start}$PATH{end}\nbanner\n{start}/real/bin:/usr/bin{end}"
    assert extract_sentinel_path(noisy) == "/real/bin:/usr/bin"
    assert extract_sentinel_path("no markers here") is None


@pytest.fixture
def _startup_harness(monkeypatch, tmp_path):
    """``start_gateway`` up to a runner that exits cleanly, as in test_runner_startup_failures.py."""
    seen = {}

    class _CleanExitRunner:
        def __init__(self, config):
            self.config = config
            self.should_exit_cleanly = True
            self.exit_reason = None
            self.exit_code = None
            self.adapters = {}

        async def start(self):
            seen["PATH"] = os.environ["PATH"]
            return True

        async def stop(self):
            return None

    monkeypatch.setattr("gateway.run._host_attach_or_none", AsyncMock(return_value=None))
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr("gateway.run.GatewayRunner", _CleanExitRunner)
    yield seen
    from gateway import host_rendezvous as hr

    hr.release_host_lock(hr.ROLE_GATEWAY)
    hr.clear_record(hr.ROLE_GATEWAY)


@pytest.mark.asyncio
@pytest.mark.parametrize("gateway_yaml, merged", [("gateway: {}\n", True),
                                                   ("gateway:\n  login_shell_path: false\n", False)])
async def test_supervised_gateway_start_reads_config_and_merges_login_path(
        monkeypatch, tmp_path, home, _startup_harness, gateway_yaml, merged):
    """E2E through start_gateway with a real config.yaml in HERMES_HOME."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(gateway_yaml, encoding="utf-8")
    tool = _install_tool(home / ".local" / "share" / "pnpm")
    (home / ".profile").write_text(f'export PATH="{tool.parent}:$PATH"\n', encoding="utf-8")
    unit_path = os.environ["PATH"]
    for key, value in _service_env(home, PATH=unit_path).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr("gateway.run._hermes_home", hermes_home)

    from gateway.run import start_gateway

    assert await start_gateway(config=GatewayConfig(), replace=False, verbosity=None) is True
    resolved = shutil.which(tool.name, path=_startup_harness["PATH"])
    assert resolved == (str(tool) if merged else None)
    assert _startup_harness["PATH"].startswith(unit_path)
