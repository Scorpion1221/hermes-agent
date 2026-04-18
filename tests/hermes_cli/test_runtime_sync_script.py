from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[2]
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync_runtime_hermes.sh"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _summary_lines(stdout: str) -> dict[str, str]:
    summary: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        summary[key.strip()] = value.strip()
    return summary


def test_sync_runtime_hermes_script_is_valid_shell():
    result = subprocess.run(["bash", "-n", str(SYNC_SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    content = SYNC_SCRIPT.read_text(encoding="utf-8")
    assert "--file-list PATH" in content
    assert "importlib.import_module" in content
    assert "py_compile" in content
    assert "--gateway-cmd CMD" in content


def test_sync_runtime_hermes_script_syncs_backs_up_and_smoke_tests(tmp_path):
    source_root = tmp_path / "source"
    runtime_root = tmp_path / "runtime"
    backup_root = tmp_path / "backups"

    _write(source_root / "demo_pkg/__init__.py", 'VALUE = "source-init"\n')
    _write(source_root / "demo_pkg/tool.py", 'VALUE = "source-tool"\n')
    _write(source_root / "notes.txt", "fresh runtime note\n")

    _write(runtime_root / "demo_pkg/__init__.py", 'VALUE = "runtime-init"\n')
    _write(runtime_root / "demo_pkg/tool.py", 'VALUE = "runtime-tool"\n')
    _write(runtime_root / "notes.txt", "old runtime note\n")

    file_list = tmp_path / "sync-files.txt"
    file_list.write_text(
        textwrap.dedent(
            """
            # Python files
            demo_pkg/__init__.py
            demo_pkg/tool.py

            # Non-Python file should still sync
            notes.txt
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(SYNC_SCRIPT),
            "--source-root",
            str(source_root),
            "--runtime-root",
            str(runtime_root),
            "--backup-root",
            str(backup_root),
            "--python",
            sys.executable,
            "--file-list",
            str(file_list),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = _summary_lines(result.stdout)

    assert summary["synced_files"] == "3"
    assert summary["py_compile"] == "ok"
    assert summary["import_smoke"] == "ok"
    assert summary["gateway_restart"] == "skipped"
    assert "import_smoke_modules=demo_pkg,demo_pkg.tool" in result.stdout

    assert (runtime_root / "demo_pkg/__init__.py").read_text(encoding="utf-8") == 'VALUE = "source-init"\n'
    assert (runtime_root / "demo_pkg/tool.py").read_text(encoding="utf-8") == 'VALUE = "source-tool"\n'
    assert (runtime_root / "notes.txt").read_text(encoding="utf-8") == "fresh runtime note\n"

    backup_dir = Path(summary["backup_dir"])
    assert backup_dir.exists()
    assert (backup_dir / "demo_pkg/__init__.py").read_text(encoding="utf-8") == 'VALUE = "runtime-init"\n'
    assert (backup_dir / "demo_pkg/tool.py").read_text(encoding="utf-8") == 'VALUE = "runtime-tool"\n'
    assert (backup_dir / "notes.txt").read_text(encoding="utf-8") == "old runtime note\n"


def test_sync_runtime_hermes_script_optionally_restarts_gateway(tmp_path):
    source_root = tmp_path / "source"
    runtime_root = tmp_path / "runtime"
    backup_root = tmp_path / "backups"

    _write(source_root / "demo_pkg/__init__.py", 'VALUE = "source"\n')
    _write(runtime_root / "demo_pkg/__init__.py", 'VALUE = "runtime"\n')

    restart_marker = tmp_path / "gateway-restarted.txt"
    restart_stub = tmp_path / "restart-gateway.sh"
    restart_stub.write_text(
        "#!/usr/bin/env bash\n"
        f"printf restarted > {shlex.quote(str(restart_marker))}\n",
        encoding="utf-8",
    )
    restart_stub.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            str(SYNC_SCRIPT),
            "--source-root",
            str(source_root),
            "--runtime-root",
            str(runtime_root),
            "--backup-root",
            str(backup_root),
            "--python",
            sys.executable,
            "--file",
            "demo_pkg/__init__.py",
            "--restart",
            "--gateway-cmd",
            str(restart_stub),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = _summary_lines(result.stdout)
    assert summary["gateway_restart"] == "ok"
    assert restart_marker.read_text(encoding="utf-8") == "restarted"


def test_sync_runtime_hermes_script_honors_explicit_import_list(tmp_path):
    source_root = tmp_path / "source"
    runtime_root = tmp_path / "runtime"
    backup_root = tmp_path / "backups"

    _write(source_root / "demo_pkg/__init__.py", 'VALUE = "source-init"\n')
    _write(source_root / "demo_pkg/tool.py", 'VALUE = "source-tool"\n')
    _write(runtime_root / "demo_pkg/__init__.py", 'VALUE = "runtime-init"\n')
    _write(runtime_root / "demo_pkg/tool.py", 'VALUE = "runtime-tool"\n')

    result = subprocess.run(
        [
            "bash",
            str(SYNC_SCRIPT),
            "--source-root",
            str(source_root),
            "--runtime-root",
            str(runtime_root),
            "--backup-root",
            str(backup_root),
            "--python",
            sys.executable,
            "--file",
            "demo_pkg/__init__.py",
            "--file",
            "demo_pkg/tool.py",
            "--import",
            "demo_pkg.tool",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = _summary_lines(result.stdout)

    assert summary["py_compile"] == "ok"
    assert summary["import_smoke"] == "ok"
    assert "import_smoke_modules=demo_pkg.tool" in result.stdout
    assert "import_smoke_modules=demo_pkg,demo_pkg.tool" not in result.stdout
