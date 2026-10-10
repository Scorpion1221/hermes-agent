"""Batch display-name lookup for Feishu users, warming the adapter's sender-name cache.

A merge-forward transcript names every distinct sender, so one ``contact/v3/users/batch`` call
replaces a ``contact/v3/users/{id}`` GET per sender. Feishu rejects the whole batch when any id is
unknown to the tenant (an external contact, a departed user), so a failed batch falls back to the
adapter's single lookup for that chunk.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import time
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

_BATCH_SIZE = 50  # contact/v3/users/batch accepts at most 50 user_ids per call
NAME_CACHE_MAX = 2048
_NAME_FIELDS = ("name", "display_name", "nickname", "en_name")


async def prefetch_user_names(adapter: Any, open_ids: Iterable[str]) -> None:
    """Cache display names for ``open_ids`` not already cached; failures leave them unnamed."""
    from .adapter import _FEISHU_SENDER_NAME_TTL_SECONDS

    pending = [oid for oid in dict.fromkeys(open_ids) if oid and adapter._get_cached_sender_name(oid) is None]
    for start in range(0, len(pending), _BATCH_SIZE):
        chunk = pending[start:start + _BATCH_SIZE]
        names = await _fetch_user_names(adapter, chunk)
        if names is None:
            await asyncio.gather(*(adapter._resolve_sender_name_from_api(oid) for oid in chunk))
            continue
        expire_at = time.time() + _FEISHU_SENDER_NAME_TTL_SECONDS
        adapter._sender_name_cache.update((oid, (name, expire_at)) for oid, name in names.items())
    _trim(adapter._sender_name_cache)


async def _fetch_user_names(adapter: Any, open_ids: list[str]) -> Optional[dict[str, str]]:
    try:
        content = await adapter._tenant_get_raw(
            "/open-apis/contact/v3/users/batch",
            queries=[("user_id_type", "open_id"), *(("user_ids", oid) for oid in open_ids)],
        )
        payload = json.loads(content) if content else {}
    except Exception:
        logger.debug("[Feishu] Batch user-name lookup failed", exc_info=True)
        return None
    code = payload.get("code") if isinstance(payload, dict) else None
    if code != 0:
        logger.debug("[Feishu] Batch user-name lookup refused: code=%s", code)
        return None
    names: dict[str, str] = {}
    for user in (payload.get("data") or {}).get("items") or []:
        if not isinstance(user, dict):
            continue
        name = next((str(user[key]).strip() for key in _NAME_FIELDS if str(user.get(key) or "").strip()), "")
        if user.get("open_id") and name:
            names[str(user["open_id"])] = name
    return names


def _trim(cache: dict[str, tuple[str, float]]) -> None:
    """Drop expired names, then the oldest entries beyond ``NAME_CACHE_MAX``."""
    now = time.time()
    for key in [key for key, (_name, expire_at) in cache.items() if expire_at <= now]:
        del cache[key]
    for key in list(itertools.islice(cache, max(0, len(cache) - NAME_CACHE_MAX))):
        del cache[key]
