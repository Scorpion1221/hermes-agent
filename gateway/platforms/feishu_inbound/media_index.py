from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

from hermes_constants import get_hermes_home


_INDEX_LOCK = threading.RLock()
_INDEX_PATH = get_hermes_home() / 'cache' / 'feishu_media_index.json'


@dataclass(frozen=True)
class FeishuMediaIndexEntry:
    platform: str
    message_id: str
    file_key: str
    cached_path: str
    content_type: str = ''
    resource_type: str = ''
    updated_at: float = 0.0

    @property
    def key(self) -> str:
        return make_media_index_key(self.platform, self.message_id, self.file_key)


def make_media_index_key(platform: str, message_id: str, file_key: str) -> str:
    return f"{platform}:{str(message_id or '').strip()}:{str(file_key or '').strip()}"


def _ensure_parent() -> None:
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_raw() -> Dict[str, dict]:
    try:
        return json.loads(_INDEX_PATH.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _save_raw(data: Dict[str, dict]) -> None:
    _ensure_parent()
    tmp = _INDEX_PATH.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    tmp.replace(_INDEX_PATH)


def get_feishu_media_index_entry(message_id: str, file_key: str, *, platform: str = 'feishu') -> Optional[FeishuMediaIndexEntry]:
    key = make_media_index_key(platform, message_id, file_key)
    with _INDEX_LOCK:
        data = _load_raw()
        item = data.get(key)
        if not isinstance(item, dict):
            return None
        entry = FeishuMediaIndexEntry(**item)
        if not entry.cached_path or not Path(entry.cached_path).exists():
            data.pop(key, None)
            _save_raw(data)
            return None
        return entry


def put_feishu_media_index_entry(
    *,
    message_id: str,
    file_key: str,
    cached_path: str,
    content_type: str = '',
    resource_type: str = '',
    platform: str = 'feishu',
) -> FeishuMediaIndexEntry:
    entry = FeishuMediaIndexEntry(
        platform=platform,
        message_id=str(message_id or '').strip(),
        file_key=str(file_key or '').strip(),
        cached_path=str(cached_path or '').strip(),
        content_type=str(content_type or '').strip(),
        resource_type=str(resource_type or '').strip(),
        updated_at=time.time(),
    )
    with _INDEX_LOCK:
        data = _load_raw()
        data[entry.key] = asdict(entry)
        _save_raw(data)
    return entry


def remove_feishu_media_index_entry(message_id: str, file_key: str, *, platform: str = 'feishu') -> bool:
    key = make_media_index_key(platform, message_id, file_key)
    with _INDEX_LOCK:
        data = _load_raw()
        removed = data.pop(key, None) is not None
        if removed:
            _save_raw(data)
        return removed
