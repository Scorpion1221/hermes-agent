"""Batch display-name lookup for Feishu users, warming the adapter's sender-name cache.

A merge-forward transcript names every distinct sender, so one ``contact/v3/users/batch`` call
replaces a ``contact/v3/users/{id}`` GET per sender. Feishu rejects the whole batch when any id is
unknown to the tenant (an external contact, a departed user) and names those ids in the error's
``field_violations``; they are cached as nameless ("") for the usual TTL, so the next forward does
not repeat the refused call, and the rest of the chunk is batched again. Only an id Feishu itself
reported unknown, or one a successful batch answered without a name, is cached as nameless: the
adapter treats "" as known and never looks the id up again, so a guess there would hide a real
sender's name from their own messages. A refusal that names no ids, or a failed call, falls back to
the adapter's single lookup per sender (a few at a time), which caches only the names it finds.

Bot names come from ``bot/v3/bots/basic_batch`` instead: the contact API does not know bots.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

_BATCH_SIZE = 50  # contact/v3/users/batch accepts at most 50 user_ids per call
_SINGLE_LOOKUP_CONCURRENCY = 4  # fallback users/{id} GETs in flight at once
NAME_CACHE_MAX = 2048
_NAME_FIELDS = ("name", "display_name", "nickname", "en_name")


@dataclass(frozen=True)
class _Refused:
    """Feishu answered but rejected the batch; ``unknown`` = the ids it named as invalid (may be empty)."""

    unknown: frozenset[str]


async def prefetch_user_names(adapter: Any, open_ids: Iterable[str]) -> None:
    """Cache display names for ``open_ids`` not already cached; failures leave them unnamed."""
    from .adapter import _FEISHU_SENDER_NAME_TTL_SECONDS

    pending = [oid for oid in dict.fromkeys(open_ids) if oid and adapter._get_cached_sender_name(oid) is None]
    for start in range(0, len(pending), _BATCH_SIZE):
        chunk = pending[start:start + _BATCH_SIZE]
        names = await _fetch_user_names(adapter, chunk)
        unknown: frozenset[str] = frozenset()
        if isinstance(names, _Refused) and names.unknown:
            unknown = names.unknown & set(chunk)
            rest = [oid for oid in chunk if oid not in unknown]
            names = await _fetch_user_names(adapter, rest) if rest else {}
        answered: Iterable[str] = chunk
        if not isinstance(names, dict):  # no usable answer: per-sender lookups, which remember no misses
            await _resolve_each(adapter, [oid for oid in chunk if oid not in unknown])
            names, answered = {}, unknown
        expire_at = time.time() + _FEISHU_SENDER_NAME_TTL_SECONDS
        adapter._sender_name_cache.update(
            (oid, (names.get(oid, ""), expire_at)) for oid in answered if adapter._get_cached_sender_name(oid) is None
        )
    _trim(adapter._sender_name_cache)


async def _resolve_each(adapter: Any, open_ids: list[str]) -> None:
    semaphore = asyncio.Semaphore(_SINGLE_LOOKUP_CONCURRENCY)

    async def resolve(oid: str) -> None:
        async with semaphore:
            await adapter._resolve_sender_name_from_api(oid)

    await asyncio.gather(*(resolve(oid) for oid in open_ids))


async def _fetch_user_names(adapter: Any, open_ids: list[str]) -> Any:
    """``{open_id: name}`` for the ids Feishu named, a ``_Refused``, or None on a transport failure."""
    try:
        content = await adapter._tenant_get_raw(
            "/open-apis/contact/v3/users/batch",
            queries=[("user_id_type", "open_id"), *(("user_ids", oid) for oid in open_ids)],
        )
        payload = json.loads(content) if content else {}
    except Exception:
        logger.debug("[Feishu] Batch user-name lookup failed", exc_info=True)
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("code") != 0:
        logger.debug("[Feishu] Batch user-name lookup refused: code=%s", payload.get("code"))
        violations = (payload.get("error") or {}).get("field_violations") or []
        return _Refused(frozenset(
            str(v["value"]) for v in violations if isinstance(v, dict) and v.get("field") == "user_ids" and v.get("value")
        ))
    names: dict[str, str] = {}
    for user in (payload.get("data") or {}).get("items") or []:
        if not isinstance(user, dict):
            continue
        name = next((str(user[key]).strip() for key in _NAME_FIELDS if str(user.get(key) or "").strip()), "")
        if user.get("open_id") and name:
            names[str(user["open_id"])] = name
    return names


async def fetch_bot_names(adapter: Any, bot_ids: list[str]) -> Optional[dict[str, str]]:
    """``{open_id: name}`` from ``bot/v3/bots/basic_batch`` (the contact API has no bot names); None on failure."""
    if not adapter._client or not bot_ids:
        return None
    try:
        content = await adapter._tenant_get_raw(
            "/open-apis/bot/v3/bots/basic_batch", queries=[("bot_ids", oid) for oid in bot_ids],
        )
        if not content:
            return None
        payload = json.loads(content)
        if payload.get("code") != 0:
            return None
        bots = (payload.get("data") or {}).get("bots") or {}
        return {oid: str(info.get("name") or "").strip() for oid, info in bots.items() if oid}
    except Exception:
        logger.debug("[Feishu] Failed to fetch bot names for %s", bot_ids, exc_info=True)
        return None


def _trim(cache: dict[str, tuple[str, float]]) -> None:
    """Drop expired names, then the oldest entries beyond ``NAME_CACHE_MAX``."""
    now = time.time()
    for key in [key for key, (_name, expire_at) in cache.items() if expire_at <= now]:
        del cache[key]
    for key in list(itertools.islice(cache, max(0, len(cache) - NAME_CACHE_MAX))):
        del cache[key]
