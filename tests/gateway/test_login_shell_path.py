"""The gateway merges the user's login-shell PATH when a supervisor starts it.

A systemd unit / launchd plist carries the PATH of whichever shell last wrote
it, so a tool that only the user's rc files put on PATH (nvm, pyenv, cargo)
vanished for the bot after ``hermes update`` ran from a shell without them.
These tests run a real login shell against a temp HOME.
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
import time

import pytest

from gateway.config import GatewayConfig
from gateway.login_shell_path import (
    apply_for_service,
    extract_sentinel_path,
    merge_path_entries,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX login shells"
)

BASH = shutil.which("bash") or "/bin/bash"
UNIT_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"


def _install_tool(directory, name="nvm-only-tool"):
    directory.mkdir(parents=True, exist_ok=True)
    tool = directory / name
    tool.write_text("#!/bin/sh\necho ok\n")
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    return tool


def _service_env(home, **extra):
    """The environment a systemd unit hands the gateway."""
    env = {
        "HOME": str(home),
        "SHELL": BASH,
        "PATH": UNIT_PATH,
        "INVOCATION_ID": "unit-test",
    }
    env.update(extra)
    return env


@pytest.fixture
def home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    return home


def test_tool_added_by_profile_is_resolvable(home):
    tool = _install_tool(home / ".nvm" / "versions" / "node" / "v22" / "bin")
    (home / ".profile").write_text(f'export PATH="{tool.parent}:$PATH"\n')
    env = _service_env(home)

    added = apply_for_service({}, env, timeout=10)

    # /etc/profile may add system dirs too (macOS path_helper).
    assert str(tool.parent) in added
    assert shutil.which(tool.name, path=env["PATH"]) == str(tool)
    # The unit's own entries keep priority.
    assert env["PATH"].startswith(UNIT_PATH + os.pathsep)


def test_profile_noise_does_not_corrupt_path(home):
    tool = _install_tool(home / "bin")
    (home / ".profile").write_text(
        f'echo "Welcome back!"\nexport PATH="{tool.parent}:$PATH"\necho "PATH is $PATH"\n'
    )
    env = _service_env(home)

    apply_for_service({}, env, timeout=10)

    assert shutil.which(tool.name, path=env["PATH"]) == str(tool)
    assert all(os.path.isabs(entry) for entry in env["PATH"].split(os.pathsep))


def test_failing_profile_leaves_unit_path(home):
    (home / ".profile").write_text("exit 3\n")
    env = _service_env(home)

    assert apply_for_service({}, env, timeout=10) is None
    assert env["PATH"] == UNIT_PATH


def test_hanging_profile_times_out_once_and_leaves_unit_path(home):
    (home / ".profile").write_text("sleep 30\n")
    env = _service_env(home)

    started = time.monotonic()
    assert apply_for_service({}, env, timeout=1) is None
    # One timeout plus the kill grace — the -lc fallback must not run again.
    assert time.monotonic() - started < 6
    assert env["PATH"] == UNIT_PATH


def test_gateway_secrets_are_not_handed_to_rc_files(home, tmp_path):
    dump = tmp_path / "rc-env.txt"
    (home / ".profile").write_text(f'env > "{dump}"\n')
    env = _service_env(home, OPENAI_API_KEY="sk-test-not-for-rc", FEISHU_APP_SECRET="s3cret")

    apply_for_service({}, env, timeout=10)

    seen = dump.read_text()
    assert "sk-test-not-for-rc" not in seen
    assert "s3cret" not in seen
    assert "INVOCATION_ID" not in seen


def test_terminal_launch_is_left_alone(home):
    tool = _install_tool(home / "bin")
    (home / ".profile").write_text(f'export PATH="{tool.parent}:$PATH"\n')
    env = _service_env(home)
    del env["INVOCATION_ID"]

    assert apply_for_service({}, env, timeout=10) is None
    assert env["PATH"] == UNIT_PATH


def test_config_switch_disables_the_merge(home):
    tool = _install_tool(home / "bin")
    (home / ".profile").write_text(f'export PATH="{tool.parent}:$PATH"\n')
    env = _service_env(home)

    assert apply_for_service({"gateway": {"login_shell_path": False}}, env, timeout=10) is None
    assert env["PATH"] == UNIT_PATH


def test_merge_keeps_current_order_and_drops_relative_entries():
    merged = merge_path_entries(
        "/venv/bin:/usr/bin:/bin",
        "/home/u/.nvm/bin:.:bin:/usr/bin:/home/u/.cargo/bin",
    )
    assert merged.split(os.pathsep) == [
        "/venv/bin",
        "/usr/bin",
        "/bin",
        "/home/u/.nvm/bin",
        "/home/u/.cargo/bin",
    ]


def test_sentinel_uses_the_last_marker():
    start, end = "__HERMES_LOGIN_PATH_START__", "__HERMES_LOGIN_PATH_END__"
    noisy = f"+ printf {start}$PATH{end}\nbanner\n{start}/real/bin:/usr/bin{end}"
    assert extract_sentinel_path(noisy) == "/real/bin:/usr/bin"
    assert extract_sentinel_path("no markers here") is None


@pytest.mark.asyncio
async def test_supervised_gateway_start_merges_login_path(monkeypatch, tmp_path, home):
    """E2E through start_gateway with a real config.yaml in HERMES_HOME."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("gateway: {}\n")
    tool = _install_tool(home / ".local" / "share" / "pnpm")
    (home / ".profile").write_text(f'export PATH="{tool.parent}:$PATH"\n')

    for key, value in _service_env(home).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    seen_path = {}

    class _CleanExitRunner:
        def __init__(self, config):
            self.config = config
            self.should_exit_cleanly = True
            self.exit_reason = None
            self.exit_code = None
            self.adapters = {}

        async def start(self):
            seen_path["PATH"] = os.environ["PATH"]
            return True

        async def stop(self):
            return None

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr("gateway.run.GatewayRunner", _CleanExitRunner)

    from gateway.run import start_gateway

    assert await start_gateway(config=GatewayConfig(), replace=False, verbosity=None) is True
    assert shutil.which(tool.name, path=seen_path["PATH"]) == str(tool)
