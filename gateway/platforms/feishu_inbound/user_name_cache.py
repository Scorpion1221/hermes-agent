from __future__ import annotations

"""Feishu sender-name cache helpers.

Inspired by openclaw-lark's sender lookup layer, this module keeps sender-name
cache entries scoped to a concrete Feishu account/app and exposes async helpers
for single-user and batch-oriented resolution.

The cache itself is intentionally conservative:

* cache entries are scoped by account/accountId so multiple Feishu accounts
  never share sender-name lookups;
* opaque Feishu IDs such as ``ou_...`` / ``on_...`` are never surfaced as
  display names;
* batch lookups gracefully fall back to single-user resolution when an account
  cannot provide a batch profile API.
"""

import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

DEFAULT_FEISHU_SENDER_NAME_TTL_SECONDS = 10 * 60
_DEFAULT_ACCOUNT_SCOPE = "__default__"
_OPAQUE_FEISHU_USER_ID_PREFIXES = ("ou_", "on_")
_VISIBLE_NAME_FIELDS = (
    "user_name",
    "sender_name",
    "name",
    "display_name",
    "nickname",
    "en_name",
    "full_name",
    "real_name",
    "alias",
)

SingleUserResolver = Callable[[str], Awaitable[Any]]
BatchUserResolver = Callable[[Sequence[str]], Awaitable[Mapping[str, Any]]]


@dataclass(frozen=True)
class FeishuSenderNameCacheEntry:
    account_scope: str
    sender_id: str
    user_name: str
    cached_at: float
    expires_at: float


def _coerce_visible_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _extract_visible_name(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        for key in _VISIBLE_NAME_FIELDS:
            text = _coerce_visible_text(value.get(key))
            if text:
                return text
        return ""
    for key in _VISIBLE_NAME_FIELDS:
        text = _coerce_visible_text(getattr(value, key, None))
        if text:
            return text
    return _coerce_visible_text(value)


def normalize_feishu_account_scope(account_scope: Optional[str]) -> str:
    normalized = str(account_scope or "").strip()
    return normalized or _DEFAULT_ACCOUNT_SCOPE


def normalize_feishu_sender_id(sender_id: Optional[str]) -> str:
    return str(sender_id or "").strip()


def is_probably_feishu_opaque_user_id(value: Optional[str]) -> bool:
    normalized = normalize_feishu_sender_id(value)
    return normalized.startswith(_OPAQUE_FEISHU_USER_ID_PREFIXES)


def suppress_opaque_feishu_user_id(value: Optional[str]) -> str:
    normalized = normalize_feishu_sender_id(value)
    if not normalized or is_probably_feishu_opaque_user_id(normalized):
        return ""
    return normalized


def coerce_feishu_sender_display_name(*, sender_name: Optional[Any] = None, sender_id: Optional[str] = None) -> str:
    """Return the best visible sender label without leaking opaque Feishu IDs."""
    visible_name = suppress_opaque_feishu_user_id(_extract_visible_name(sender_name))
    if visible_name:
        return visible_name
    return suppress_opaque_feishu_user_id(sender_id)


class FeishuSenderNameCache:
    """TTL cache for sender display names keyed by ``(account_scope, sender_id)``."""

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_FEISHU_SENDER_NAME_TTL_SECONDS,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self._ttl_seconds = max(float(ttl_seconds), 0.0)
        self._time_fn = time_fn
        self._entries: dict[tuple[str, str], FeishuSenderNameCacheEntry] = {}

    @staticmethod
    def make_cache_key(account_scope: Optional[str], sender_id: Optional[str]) -> tuple[str, str]:
        return (
            normalize_feishu_account_scope(account_scope),
            normalize_feishu_sender_id(sender_id),
        )

    def __len__(self) -> int:
        self.prune_expired()
        return len(self._entries)

    # Compatibility helpers for existing adapter/tests that still treat the
    # sender-name cache like ``dict[sender_id] = (name, expires_at)``.
    def __contains__(self, sender_id: object) -> bool:
        return self._get_entry(account_scope=None, sender_id=str(sender_id or "")) is not None

    def __getitem__(self, sender_id: str) -> tuple[str, float]:
        entry = self._get_entry(account_scope=None, sender_id=sender_id)
        if entry is None:
            raise KeyError(sender_id)
        return (entry.user_name, entry.expires_at)

    def __setitem__(self, sender_id: str, value: tuple[str, float]) -> None:
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("Expected (user_name, expires_at) tuple")
        normalized_sender_id = normalize_feishu_sender_id(sender_id)
        visible_name = coerce_feishu_sender_display_name(sender_name=value[0])
        expires_at = float(value[1])
        if not normalized_sender_id or not visible_name:
            return
        cached_at = min(self._time_fn(), expires_at)
        entry = FeishuSenderNameCacheEntry(
            account_scope=normalize_feishu_account_scope(None),
            sender_id=normalized_sender_id,
            user_name=visible_name,
            cached_at=cached_at,
            expires_at=expires_at,
        )
        self._entries[(entry.account_scope, entry.sender_id)] = entry

    def get(self, sender_id: Optional[str], default: Optional[tuple[str, float]] = None) -> Optional[tuple[str, float]]:
        entry = self._get_entry(account_scope=None, sender_id=sender_id)
        if entry is None:
            return default
        return (entry.user_name, entry.expires_at)

    def pop(self, sender_id: Optional[str], default: Any = None) -> Any:
        normalized_sender_id = normalize_feishu_sender_id(sender_id)
        if not normalized_sender_id:
            return default
        key = self.make_cache_key(None, normalized_sender_id)
        entry = self._entries.pop(key, None)
        if entry is None:
            return default
        return (entry.user_name, entry.expires_at)

    def _get_entry(self, *, account_scope: Optional[str], sender_id: Optional[str]) -> Optional[FeishuSenderNameCacheEntry]:
        normalized_sender_id = normalize_feishu_sender_id(sender_id)
        if not normalized_sender_id:
            return None
        normalized_scope = normalize_feishu_account_scope(account_scope)
        key = (normalized_scope, normalized_sender_id)
        entry = self._entries.get(key)
        if entry is None and normalized_scope != _DEFAULT_ACCOUNT_SCOPE:
            key = (_DEFAULT_ACCOUNT_SCOPE, normalized_sender_id)
            entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= self._time_fn():
            self._entries.pop(key, None)
            return None
        return entry

    def get_cached_entry(
        self,
        sender_id: Optional[str],
        *,
        account_scope: Optional[str] = None,
    ) -> Optional[FeishuSenderNameCacheEntry]:
        return self._get_entry(account_scope=account_scope, sender_id=sender_id)

    def get_cached_name(self, sender_id: Optional[str], *, account_scope: Optional[str] = None) -> Optional[str]:
        entry = self._get_entry(account_scope=account_scope, sender_id=sender_id)
        return entry.user_name if entry is not None else None

    def put_cached_name(
        self,
        sender_id: Optional[str],
        user_name: Optional[str],
        *,
        account_scope: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
    ) -> Optional[FeishuSenderNameCacheEntry]:
        normalized_sender_id = normalize_feishu_sender_id(sender_id)
        visible_name = coerce_feishu_sender_display_name(sender_name=_extract_visible_name(user_name))
        if not normalized_sender_id or not visible_name:
            return None

        now = self._time_fn()
        ttl = self._ttl_seconds if ttl_seconds is None else max(float(ttl_seconds), 0.0)
        entry = FeishuSenderNameCacheEntry(
            account_scope=normalize_feishu_account_scope(account_scope),
            sender_id=normalized_sender_id,
            user_name=visible_name,
            cached_at=now,
            expires_at=now + ttl,
        )
        self._entries[(entry.account_scope, entry.sender_id)] = entry
        return entry

    def clear_account(self, account_scope: Optional[str]) -> int:
        normalized_scope = normalize_feishu_account_scope(account_scope)
        keys = [key for key in self._entries if key[0] == normalized_scope]
        for key in keys:
            self._entries.pop(key, None)
        return len(keys)

    def prune_expired(self) -> int:
        now = self._time_fn()
        expired_keys = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired_keys:
            self._entries.pop(key, None)
        return len(expired_keys)

    async def resolve_user_name(
        self,
        sender_id: Optional[str],
        *,
        account_scope: Optional[str] = None,
        resolver: Optional[SingleUserResolver] = None,
    ) -> Optional[str]:
        normalized_sender_id = normalize_feishu_sender_id(sender_id)
        if not normalized_sender_id:
            return None

        cached_name = self.get_cached_name(normalized_sender_id, account_scope=account_scope)
        if cached_name is not None:
            return cached_name
        if resolver is None:
            return None

        try:
            resolved = await resolver(normalized_sender_id)
        except Exception:
            return None

        visible_name = coerce_feishu_sender_display_name(sender_name=resolved)
        if not visible_name:
            return None
        self.put_cached_name(normalized_sender_id, visible_name, account_scope=account_scope)
        return visible_name

    async def resolve_user_names(
        self,
        sender_ids: Iterable[str],
        *,
        account_scope: Optional[str] = None,
        batch_resolver: Optional[BatchUserResolver] = None,
        single_resolver: Optional[SingleUserResolver] = None,
    ) -> dict[str, Optional[str]]:
        """Resolve multiple sender IDs with cache-first semantics.

        The returned mapping is ordered by the first occurrence of each normalized
        sender ID in ``sender_ids``. If ``batch_resolver`` is missing or reports
        ``NotImplementedError``, unresolved IDs fall back to ``single_resolver``.
        """

        ordered_ids: list[str] = []
        results: dict[str, Optional[str]] = {}
        for raw_sender_id in sender_ids:
            normalized_sender_id = normalize_feishu_sender_id(raw_sender_id)
            if not normalized_sender_id or normalized_sender_id in results:
                continue
            ordered_ids.append(normalized_sender_id)
            results[normalized_sender_id] = self.get_cached_name(
                normalized_sender_id,
                account_scope=account_scope,
            )

        unresolved = [sender_id for sender_id in ordered_ids if results[sender_id] is None]

        if unresolved and batch_resolver is not None:
            try:
                batch_result = await batch_resolver(tuple(unresolved))
            except NotImplementedError:
                batch_result = {}
            except Exception:
                batch_result = {}

            if not isinstance(batch_result, Mapping):
                batch_result = {}

            for sender_id in unresolved:
                visible_name = coerce_feishu_sender_display_name(sender_name=batch_result.get(sender_id))
                if not visible_name:
                    continue
                self.put_cached_name(sender_id, visible_name, account_scope=account_scope)
                results[sender_id] = visible_name

            unresolved = [sender_id for sender_id in unresolved if results[sender_id] is None]

        if unresolved and single_resolver is not None:
            for sender_id in unresolved:
                results[sender_id] = await self.resolve_user_name(
                    sender_id,
                    account_scope=account_scope,
                    resolver=single_resolver,
                )

        return results


async def resolve_feishu_sender_display_name(
    cache: FeishuSenderNameCache,
    sender_id: Optional[str],
    *,
    sender_name: Optional[Any] = None,
    account_scope: Optional[str] = None,
    resolver: Optional[SingleUserResolver] = None,
) -> str:
    """Resolve a single sender name and always return a safe display label.

    This helper is the display-oriented companion to :func:`resolve_feishu_sender_name`.
    It prefers a visible sender name already present on the event payload, then the
    cache/API lookup, and finally a sanitized fallback that never leaks opaque
    Feishu IDs.
    """

    visible_name = coerce_feishu_sender_display_name(
        sender_name=sender_name,
        sender_id=sender_id,
    )
    normalized_sender_id = normalize_feishu_sender_id(sender_id)
    if visible_name and normalized_sender_id:
        cache.put_cached_name(normalized_sender_id, visible_name, account_scope=account_scope)
        return visible_name

    resolved_name = await resolve_feishu_sender_name(
        cache,
        sender_id,
        account_scope=account_scope,
        resolver=resolver,
    )
    if resolved_name:
        return resolved_name

    return suppress_opaque_feishu_user_id(sender_id)


async def resolve_feishu_sender_display_names(
    cache: FeishuSenderNameCache,
    sender_ids: Iterable[str],
    *,
    account_scope: Optional[str] = None,
    batch_resolver: Optional[BatchUserResolver] = None,
    single_resolver: Optional[SingleUserResolver] = None,
) -> dict[str, str]:
    """Resolve multiple sender names and always return safe display labels."""

    raw_results = await resolve_feishu_sender_names(
        cache,
        sender_ids,
        account_scope=account_scope,
        batch_resolver=batch_resolver,
        single_resolver=single_resolver,
    )
    safe_results: dict[str, str] = {}
    for sender_id, sender_name in raw_results.items():
        safe_results[sender_id] = coerce_feishu_sender_display_name(
            sender_name=sender_name,
            sender_id=sender_id,
        )
    return safe_results


async def resolve_feishu_sender_name(
    cache: FeishuSenderNameCache,
    sender_id: Optional[str],
    *,
    account_scope: Optional[str] = None,
    resolver: Optional[SingleUserResolver] = None,
) -> Optional[str]:
    return await cache.resolve_user_name(
        sender_id,
        account_scope=account_scope,
        resolver=resolver,
    )


async def resolve_feishu_sender_names(
    cache: FeishuSenderNameCache,
    sender_ids: Iterable[str],
    *,
    account_scope: Optional[str] = None,
    batch_resolver: Optional[BatchUserResolver] = None,
    single_resolver: Optional[SingleUserResolver] = None,
) -> dict[str, Optional[str]]:
    return await cache.resolve_user_names(
        sender_ids,
        account_scope=account_scope,
        batch_resolver=batch_resolver,
        single_resolver=single_resolver,
    )


__all__ = [
    "DEFAULT_FEISHU_SENDER_NAME_TTL_SECONDS",
    "FeishuSenderNameCache",
    "FeishuSenderNameCacheEntry",
    "coerce_feishu_sender_display_name",
    "is_probably_feishu_opaque_user_id",
    "normalize_feishu_account_scope",
    "normalize_feishu_sender_id",
    "resolve_feishu_sender_display_name",
    "resolve_feishu_sender_display_names",
    "resolve_feishu_sender_name",
    "resolve_feishu_sender_names",
    "suppress_opaque_feishu_user_id",
]
