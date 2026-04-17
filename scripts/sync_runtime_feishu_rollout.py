#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_ROOT = Path.home() / '.hermes' / 'hermes-agent'
SYNC_FILES = [
    'gateway/platforms/base.py',
    'gateway/platforms/feishu.py',
    'gateway/platforms/feishu_inbound/__init__.py',
    'gateway/platforms/feishu_inbound/parse.py',
    'gateway/platforms/feishu_inbound/lookup.py',
    'gateway/platforms/feishu_inbound/render.py',
    'gateway/platforms/feishu_inbound/bridge.py',
    'gateway/platforms/feishu_inbound/media_index.py',
    'gateway/run.py',
    'run_agent.py',
    'agent/auxiliary_client.py',
    'agent/display.py',
    'plugins/memory/honcho/__init__.py',
    'plugins/memory/honcho/client.py',
]


def backup_and_copy(source_root: Path, runtime_root: Path, files: list[str]) -> Path:
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    backup_dir = Path.home() / '.hermes' / 'backups' / f'feishu-runtime-sync-{ts}'
    for rel in files:
        src = source_root / rel
        dst = runtime_root / rel
        if not src.exists():
            raise FileNotFoundError(f'Missing source file: {src}')
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            bak = backup_dir / rel
            bak.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, bak)
        shutil.copy2(src, dst)
    return backup_dir


def run_py_compile(runtime_root: Path, files: list[str]) -> None:
    cmd = [str(runtime_root / 'venv' / 'bin' / 'python'), '-m', 'py_compile', *[str(runtime_root / rel) for rel in files]]
    subprocess.run(cmd, check=True)


def restart_gateway() -> None:
    subprocess.run([str(Path.home() / '.local' / 'bin' / 'hermes'), 'gateway', 'restart'], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Sync Feishu migration files into ~/.hermes runtime and optionally restart the gateway.')
    parser.add_argument('--runtime-root', default=str(DEFAULT_RUNTIME_ROOT))
    parser.add_argument('--restart', action='store_true')
    args = parser.parse_args()

    runtime_root = Path(args.runtime_root).expanduser().resolve()
    backup_dir = backup_and_copy(SOURCE_ROOT, runtime_root, SYNC_FILES)
    run_py_compile(runtime_root, SYNC_FILES)
    print(f'backup_dir={backup_dir}')
    print('py_compile=ok')
    if args.restart:
        restart_gateway()
        print('gateway_restart=ok')


if __name__ == '__main__':
    main()
