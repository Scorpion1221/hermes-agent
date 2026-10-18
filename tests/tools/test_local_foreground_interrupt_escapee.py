"""A foreground interrupt gives processes that left the command's process group the same SIGTERM
grace as the group itself.

LocalEnvironment._kill_process TERMs the command's process group, waits, then KILLs it. A
descendant that escaped the group via ``setsid`` (GNU ``timeout`` does, so ``timeout 400 bash
deploy.sh`` runs the whole job outside the group) used to get SIGKILL with no SIGTERM first: it
could not record that it was interrupted, and a write could be cut mid-way. It now gets the
group's SIGTERM relayed once and shares its grace; only a survivor is SIGKILLed.
"""

import contextlib
import os
import shlex
import shutil
import signal
import sys
import textwrap
import threading
import time

import pytest

from tools.environments.local import LocalEnvironment
from tools.interrupt import set_interrupt

pytestmark = [pytest.mark.platforms("posix"), pytest.mark.live_system_guard_bypass]

# The job: logs every SIGTERM to the marker and keeps "recording" for 0.3 s before it exits (a
# second TERM in that window would be logged too); ``ignore`` ignores SIGTERM instead.
_JOB = textwrap.dedent(
    """
    import os, signal, sys, time
    ready, marker, mode = sys.argv[1:4]
    termed = []
    def on_term(*_):
        with open(marker, "a") as fh:
            fh.write("TERM\\n")
        termed.append(time.monotonic())
    signal.signal(signal.SIGTERM, signal.SIG_IGN if mode == "ignore" else on_term)
    with open(ready + ".tmp", "w") as fh:
        fh.write(str(os.getpid()))
    os.replace(ready + ".tmp", ready)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not (termed and time.monotonic() - termed[0] > 0.3):
        time.sleep(0.02)
    """
).strip()

# Starts the job in its own session (out of the command's group) and waits for it. ``orphan``:
# dies on SIGTERM, leaving the job behind. ``relay``: forwards SIGTERM to the job and keeps
# waiting, the way a supervisor stops its ``start_new_session`` children; it forwards 0.1 s
# late so a TERM sent to the job directly would not merge with its own and stays visible.
_LAUNCHER = textwrap.dedent(
    """
    import signal, subprocess, sys, time
    jobs = []
    if sys.argv[1] == "relay":
        signal.signal(signal.SIGTERM, lambda *_: time.sleep(0.1) or [job.send_signal(signal.SIGTERM) for job in jobs])
    jobs.append(subprocess.Popen(sys.argv[2:], start_new_session=True))
    jobs[0].wait()
    """
).strip()


def _gone(psutil, pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _interrupt_running_job(tmp_path, prefix: list, mode: str):
    """Run ``prefix + job`` in the foreground and interrupt it once the job is up; returns
    (result, job pid, TERMs the job logged, seconds from the interrupt to the return)."""
    ready, marker = tmp_path / "ready", tmp_path / "marker"
    (tmp_path / "job.py").write_text(_JOB)
    (tmp_path / "launcher.py").write_text(_LAUNCHER)
    job = [str(tmp_path / "job.py"), str(ready), str(marker), mode]
    cmd = shlex.join([*prefix, sys.executable, *job])
    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    result: dict = {}
    worker = threading.Thread(target=lambda: result.update(env.execute(cmd, timeout=30)))
    worker.start()
    job_pid = None
    try:
        deadline = time.monotonic() + 20
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists(), "the job never came up"
        job_pid = int(ready.read_text())
        interrupted_at = time.monotonic()
        set_interrupt(True, thread_id=worker.ident)
        worker.join(timeout=20)
        assert not worker.is_alive(), "execute() did not return after the interrupt"
        elapsed = time.monotonic() - interrupted_at
        terms = marker.read_text().splitlines() if marker.exists() else []
        return result, job_pid, terms, elapsed
    finally:
        if worker.is_alive():
            set_interrupt(True, thread_id=worker.ident)
            worker.join(timeout=20)
        set_interrupt(False, thread_id=worker.ident)
        if job_pid is not None:
            with contextlib.suppress(OSError):
                os.kill(job_pid, signal.SIGKILL)
        env.cleanup()


def _launcher(tmp_path, mode: str) -> list:
    return [sys.executable, str(tmp_path / "launcher.py"), mode]


def test_job_left_behind_outside_the_group_gets_sigterm_once(tmp_path):
    pytest.importorskip("psutil")

    result, _, terms, _ = _interrupt_running_job(tmp_path, _launcher(tmp_path, "orphan"), "record")

    assert result["returncode"] == 130, result
    assert terms == ["TERM"], "the job outside the group was not given exactly one SIGTERM"


def test_job_under_gnu_timeout_gets_sigterm(tmp_path):
    """``timeout`` leaves the group; the TERM relayed to it reaches the job it runs."""
    pytest.importorskip("psutil")
    timeout_bin = shutil.which("timeout")
    if timeout_bin is None:
        pytest.skip("no timeout(1) on PATH")

    result, _, terms, _ = _interrupt_running_job(tmp_path, [timeout_bin, "30"], "record")

    assert result["returncode"] == 130, result
    assert "TERM" in terms, "the job under timeout was killed without a SIGTERM first"


def test_job_under_a_shell_under_gnu_timeout_gets_sigterm_once(tmp_path):
    """``timeout N bash script.sh`` (the deploy recipe's shape): timeout's TERM to its group reaches
    the shell and the job together. The shell dying of it must not get the job a second TERM."""
    pytest.importorskip("psutil")
    timeout_bin = shutil.which("timeout")
    if timeout_bin is None:
        pytest.skip("no timeout(1) on PATH")
    shell = ["bash", "-c", '"$@"; exit $?', "bash"]  # the trailing exit keeps bash from exec'ing the job

    result, _, terms, _ = _interrupt_running_job(tmp_path, [timeout_bin, "30", *shell], "record")

    assert result["returncode"] == 130, result
    assert terms == ["TERM"]


_RUNS = [sys.executable, "-c", "import subprocess, sys; subprocess.Popen(sys.argv[1:]).wait()"]  # dies on TERM


def test_job_beside_a_parent_we_terminate_alone_gets_sigterm(tmp_path):
    """Leader -> parent -> job, all in one group outside the command's. The teardown TERMs the
    orphaned leader, then the parent it leaves behind, each alone, so the job needs its own TERM."""
    pytest.importorskip("psutil")

    result, _, terms, _ = _interrupt_running_job(tmp_path, [*_launcher(tmp_path, "orphan"), *_RUNS, *_RUNS], "record")

    assert result["returncode"] == 130, result
    assert terms == ["TERM"]


def test_job_under_a_leader_its_supervisor_stopped_alone_gets_sigterm(tmp_path):
    """A supervisor TERMs its ``start_new_session`` child alone, not the child's group: when that
    child dies of it, the job it ran in its group needs a TERM from the teardown."""
    pytest.importorskip("psutil")

    result, _, terms, _ = _interrupt_running_job(tmp_path, [*_launcher(tmp_path, "relay"), *_RUNS], "record")

    assert result["returncode"] == 130, result
    assert terms == ["TERM"]


def test_parent_that_forwards_sigterm_is_not_doubled(tmp_path):
    """A live parent got the group's TERM and stops its own setsid child: a second TERM from the
    teardown would re-enter the child's handler while it is still recording."""
    pytest.importorskip("psutil")

    result, _, terms, _ = _interrupt_running_job(tmp_path, _launcher(tmp_path, "relay"), "record")

    assert result["returncode"] == 130, result
    assert terms == ["TERM"]


def test_job_outside_the_group_that_ignores_sigterm_is_killed_after_the_grace(tmp_path):
    psutil = pytest.importorskip("psutil")

    result, job_pid, _, elapsed = _interrupt_running_job(tmp_path, _launcher(tmp_path, "orphan"), "ignore")

    assert result["returncode"] == 130, result
    assert _gone(psutil, job_pid), f"job {job_pid} ignoring SIGTERM outlived the interrupt"
    # The grace is bounded: an interrupt must not wait on a process that refuses to stop.
    assert elapsed < 8.0, f"interrupt took {elapsed:.1f}s to return"
