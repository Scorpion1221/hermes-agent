#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOURCE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_RUNTIME_ROOT="$HOME/.hermes/hermes-agent"
DEFAULT_BACKUP_ROOT="$HOME/.hermes/backups"
DEFAULT_GATEWAY_CMD="$HOME/.local/bin/hermes gateway restart"

usage() {
  cat <<'USAGE'
Usage: scripts/sync_runtime_hermes.sh [options]

Sync a selected file set from the source repo into a Hermes runtime checkout,
back up overwritten runtime files, run py_compile/import smoke, and optionally
restart the gateway.

Options:
  --source-root PATH   Source repo root to copy from (default: repo root)
  --runtime-root PATH  Runtime checkout to sync into (default: ~/.hermes/hermes-agent)
  --backup-root PATH   Backup base directory (default: ~/.hermes/backups)
  --python PATH        Python executable for smoke checks
  --file RELPATH       Relative path to sync; repeatable
  --file-list PATH     Newline-delimited file list; blank lines and # comments ignored
  --import MODULE      Module to import during smoke check; repeatable
                       If omitted, importable .py files are auto-derived
  --restart            Restart the gateway after a successful sync/smoke run
  --gateway-cmd CMD    Shell command used for restart
                       (default: ~/.local/bin/hermes gateway restart)
  -h, --help           Show this help text
USAGE
}

trim_whitespace() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

path_to_module() {
  local rel="$1"
  local module_path
  local part
  local parts

  case "$rel" in
    *.py) ;;
    *) return 1 ;;
  esac

  if [[ "$rel" == */__init__.py ]]; then
    module_path="${rel%/__init__.py}"
  else
    module_path="${rel%.py}"
  fi

  if [[ -z "$module_path" ]]; then
    return 1
  fi

  IFS='/' read -r -a parts <<< "$module_path"
  for part in "${parts[@]}"; do
    if [[ ! "$part" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      return 1
    fi
  done

  printf '%s' "${module_path//\//.}"
}

source_root="$DEFAULT_SOURCE_ROOT"
runtime_root="$DEFAULT_RUNTIME_ROOT"
backup_root="$DEFAULT_BACKUP_ROOT"
python_cmd=""
gateway_cmd="$DEFAULT_GATEWAY_CMD"
restart_gateway=0
explicit_imports=0
sync_files=()
import_modules=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-root)
      [[ $# -ge 2 ]] || { echo "Missing value for --source-root" >&2; exit 1; }
      source_root="$2"
      shift 2
      ;;
    --runtime-root)
      [[ $# -ge 2 ]] || { echo "Missing value for --runtime-root" >&2; exit 1; }
      runtime_root="$2"
      shift 2
      ;;
    --backup-root)
      [[ $# -ge 2 ]] || { echo "Missing value for --backup-root" >&2; exit 1; }
      backup_root="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || { echo "Missing value for --python" >&2; exit 1; }
      python_cmd="$2"
      shift 2
      ;;
    --file)
      [[ $# -ge 2 ]] || { echo "Missing value for --file" >&2; exit 1; }
      sync_files+=("$2")
      shift 2
      ;;
    --file-list)
      [[ $# -ge 2 ]] || { echo "Missing value for --file-list" >&2; exit 1; }
      [[ -r "$2" ]] || { echo "Missing or unreadable file list: $2" >&2; exit 1; }
      while IFS= read -r line || [[ -n "$line" ]]; do
        line="$(trim_whitespace "$line")"
        [[ -z "$line" ]] && continue
        [[ "$line" == \#* ]] && continue
        sync_files+=("$line")
      done < "$2"
      shift 2
      ;;
    --import)
      [[ $# -ge 2 ]] || { echo "Missing value for --import" >&2; exit 1; }
      import_modules+=("$2")
      explicit_imports=1
      shift 2
      ;;
    --restart)
      restart_gateway=1
      shift
      ;;
    --gateway-cmd)
      [[ $# -ge 2 ]] || { echo "Missing value for --gateway-cmd" >&2; exit 1; }
      gateway_cmd="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ${#sync_files[@]} -eq 0 ]]; then
  echo "No files selected. Use --file and/or --file-list." >&2
  exit 1
fi

source_root="$(cd "$source_root" && pwd)"
runtime_root="$(mkdir -p "$runtime_root" && cd "$runtime_root" && pwd)"
backup_root="$(mkdir -p "$backup_root" && cd "$backup_root" && pwd)"

if [[ -z "$python_cmd" ]]; then
  if [[ -x "$runtime_root/venv/bin/python" ]]; then
    python_cmd="$runtime_root/venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    python_cmd="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    python_cmd="$(command -v python)"
  else
    echo "No python executable found for smoke checks." >&2
    exit 1
  fi
fi

if [[ ! -x "$python_cmd" ]]; then
  echo "Python executable is not runnable: $python_cmd" >&2
  exit 1
fi

python_files=()
validated_files=()
for rel in "${sync_files[@]}"; do
  src="$source_root/$rel"
  if [[ ! -f "$src" ]]; then
    echo "Missing source file: $src" >&2
    exit 1
  fi
  validated_files+=("$rel")
  if [[ "$rel" == *.py ]]; then
    python_files+=("$rel")
  fi
done

if [[ $explicit_imports -eq 0 ]]; then
  for rel in "${python_files[@]}"; do
    if module_name="$(path_to_module "$rel")"; then
      import_modules+=("$module_name")
    fi
  done
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
backup_dir="$backup_root/hermes-runtime-sync-$timestamp"
mkdir -p "$backup_dir"

for rel in "${validated_files[@]}"; do
  dst="$runtime_root/$rel"
  if [[ -e "$dst" ]]; then
    mkdir -p "$(dirname "$backup_dir/$rel")"
    cp -p "$dst" "$backup_dir/$rel"
  fi
done

for rel in "${validated_files[@]}"; do
  src="$source_root/$rel"
  dst="$runtime_root/$rel"
  mkdir -p "$(dirname "$dst")"
  cp -p "$src" "$dst"
done

if [[ ${#python_files[@]} -gt 0 ]]; then
  runtime_python_files=()
  for rel in "${python_files[@]}"; do
    runtime_python_files+=("$runtime_root/$rel")
  done
  "$python_cmd" -m py_compile "${runtime_python_files[@]}"
  py_compile_status="ok"
else
  py_compile_status="skipped"
fi

if [[ ${#import_modules[@]} -gt 0 ]]; then
  (
    cd "$runtime_root"
    PYTHONPATH="$runtime_root${PYTHONPATH:+:$PYTHONPATH}" \
      "$python_cmd" -c 'import importlib, sys
for module_name in sys.argv[1:]:
    importlib.import_module(module_name)
print("import_smoke_modules=" + ",".join(sys.argv[1:]))' "${import_modules[@]}"
  )
  import_smoke_status="ok"
else
  import_smoke_status="skipped"
fi

if [[ $restart_gateway -eq 1 ]]; then
  bash -lc "$gateway_cmd"
  gateway_restart_status="ok"
else
  gateway_restart_status="skipped"
fi

echo "source_root=$source_root"
echo "runtime_root=$runtime_root"
echo "backup_dir=$backup_dir"
echo "synced_files=${#validated_files[@]}"
echo "py_compile=$py_compile_status"
echo "import_smoke=$import_smoke_status"
echo "gateway_restart=$gateway_restart_status"
