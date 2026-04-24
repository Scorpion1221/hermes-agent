from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gateway.platforms.feishu_inbound.user_name_cache import (
    FeishuSenderNameCache,
    coerce_feishu_sender_display_name,
    is_probably_feishu_opaque_user_id,
    resolve_feishu_sender_display_name,
    resolve_feishu_sender_display_names,
    resolve_feishu_sender_name,
    resolve_feishu_sender_names,
    suppress_opaque_feishu_user_id,
)


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_sender_name_cache_is_scoped_per_account() -> None:
    clock = FakeClock()
    cache = FeishuSenderNameCache(time_fn=clock)

    cache.put_cached_name("ou_same", "Alice", account_scope="app-alpha")
    cache.put_cached_name("ou_same", "Bob", account_scope="app-beta")

    assert cache.get_cached_name("ou_same", account_scope="app-alpha") == "Alice"
    assert cache.get_cached_name("ou_same", account_scope="app-beta") == "Bob"
    assert cache.clear_account("app-alpha") == 1
    assert cache.get_cached_name("ou_same", account_scope="app-alpha") is None
    assert cache.get_cached_name("ou_same", account_scope="app-beta") == "Bob"


def test_get_cached_name_evicts_expired_entries() -> None:
    clock = FakeClock()
    cache = FeishuSenderNameCache(ttl_seconds=10, time_fn=clock)

    cache.put_cached_name("ou_alice", "Alice", account_scope="app-alpha")
    assert cache.get_cached_name("ou_alice", account_scope="app-alpha") == "Alice"

    clock.advance(10)

    assert len(cache) == 0
    assert cache.get_cached_name("ou_alice", account_scope="app-alpha") is None
    assert len(cache) == 0
    assert cache.prune_expired() == 0


@pytest.mark.asyncio
async def test_resolve_user_name_prefers_cache_before_calling_resolver() -> None:
    cache = FeishuSenderNameCache()
    cache.put_cached_name("ou_alice", "Alice", account_scope="app-alpha")
    resolver = AsyncMock(return_value="Alice from API")

    result = await resolve_feishu_sender_name(
        cache,
        "ou_alice",
        account_scope="app-alpha",
        resolver=resolver,
    )

    assert result == "Alice"
    resolver.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_user_name_caches_visible_name_and_swallows_failures() -> None:
    cache = FeishuSenderNameCache()
    resolver = AsyncMock(side_effect=[{"name": "Alice"}, RuntimeError("boom")])

    first = await cache.resolve_user_name("ou_alice", account_scope="app-alpha", resolver=resolver)
    second = await cache.resolve_user_name("ou_bob", account_scope="app-alpha", resolver=resolver)

    assert first == "Alice"
    assert second is None
    assert cache.get_cached_name("ou_alice", account_scope="app-alpha") == "Alice"
    assert cache.get_cached_name("ou_bob", account_scope="app-alpha") is None


@pytest.mark.asyncio
async def test_display_name_helper_prefers_payload_name_and_uses_safe_fallback() -> None:
    cache = FeishuSenderNameCache()
    visible_resolver = AsyncMock(return_value="Should not be called")
    opaque_resolver = AsyncMock(return_value="ou_hidden")

    visible = await resolve_feishu_sender_display_name(
        cache,
        "ou_alice",
        sender_name={"display_name": "Alice"},
        account_scope="app-alpha",
        resolver=visible_resolver,
    )
    opaque = await resolve_feishu_sender_display_name(
        cache,
        "ou_hidden",
        sender_name="ou_hidden",
        account_scope="app-alpha",
        resolver=opaque_resolver,
    )
    fallback = await resolve_feishu_sender_display_name(
        cache,
        "Bob",
        account_scope="app-alpha",
        resolver=None,
    )

    assert visible == "Alice"
    assert opaque == ""
    assert fallback == "Bob"
    visible_resolver.assert_not_awaited()
    opaque_resolver.assert_awaited_once_with("ou_hidden")
    assert cache.get_cached_name("ou_alice", account_scope="app-alpha") == "Alice"
    assert cache.get_cached_name("Bob", account_scope="app-alpha") == "Bob"


@pytest.mark.asyncio
async def test_batch_resolution_uses_batch_then_falls_back_to_single_for_remaining_ids() -> None:
    cache = FeishuSenderNameCache()
    cache.put_cached_name("ou_cached", "Cached Alice", account_scope="app-alpha")

    batch_resolver = AsyncMock(return_value={"ou_batch": "Batch Bob"})
    single_resolver = AsyncMock(side_effect=lambda sender_id: {"ou_single": "Single Carol"}.get(sender_id))

    result = await resolve_feishu_sender_names(
        cache,
        ["ou_cached", "ou_batch", "ou_single", "ou_cached"],
        account_scope="app-alpha",
        batch_resolver=batch_resolver,
        single_resolver=single_resolver,
    )

    assert result == {
        "ou_cached": "Cached Alice",
        "ou_batch": "Batch Bob",
        "ou_single": "Single Carol",
    }
    batch_resolver.assert_awaited_once_with(("ou_batch", "ou_single"))
    single_resolver.assert_awaited_once_with("ou_single")
    assert cache.get_cached_name("ou_batch", account_scope="app-alpha") == "Batch Bob"
    assert cache.get_cached_name("ou_single", account_scope="app-alpha") == "Single Carol"


@pytest.mark.asyncio
async def test_batch_resolution_preserves_batch_api_when_direct_batch_lookup_is_unavailable() -> None:
    cache = FeishuSenderNameCache()
    batch_resolver = AsyncMock(side_effect=NotImplementedError)
    single_resolver = AsyncMock(side_effect=lambda sender_id: {"ou_alice": "Alice", "ou_bob": "Bob"}[sender_id])

    result = await cache.resolve_user_names(
        ["ou_alice", "ou_bob"],
        account_scope="app-alpha",
        batch_resolver=batch_resolver,
        single_resolver=single_resolver,
    )

    assert result == {"ou_alice": "Alice", "ou_bob": "Bob"}
    batch_resolver.assert_awaited_once_with(("ou_alice", "ou_bob"))
    assert single_resolver.await_count == 2


@pytest.mark.asyncio
async def test_display_batch_resolution_returns_safe_labels_and_preserves_order() -> None:
    cache = FeishuSenderNameCache()
    batch_resolver = AsyncMock(return_value={"ou_batch": "Batch Bob", "ou_hidden": "ou_hidden"})
    single_resolver = AsyncMock(
        side_effect=lambda sender_id: {
            "ou_hidden": None,
            "ou_single": "Single Carol",
            "Human": "Human",
        }.get(sender_id)
    )

    result = await resolve_feishu_sender_display_names(
        cache,
        ["ou_batch", "ou_hidden", "ou_single", "Human", "ou_batch"],
        account_scope="app-alpha",
        batch_resolver=batch_resolver,
        single_resolver=single_resolver,
    )

    assert list(result) == ["ou_batch", "ou_hidden", "ou_single", "Human"]
    assert result == {
        "ou_batch": "Batch Bob",
        "ou_hidden": "",
        "ou_single": "Single Carol",
        "Human": "Human",
    }
    batch_resolver.assert_awaited_once_with(("ou_batch", "ou_hidden", "ou_single", "Human"))
    assert single_resolver.await_count == 3


@pytest.mark.parametrize(
    ("sender_name", "sender_id", "expected"),
    [
        ("Alice", "ou_alice", "Alice"),
        ("", "Human Fallback", "Human Fallback"),
        ("ou_hidden_sender", "on_hidden_union", ""),
        (None, "ou_hidden_sender", ""),
    ],
)
def test_opaque_id_helpers_hide_open_and_union_ids(
    sender_name: str | None,
    sender_id: str | None,
    expected: str,
) -> None:
    assert is_probably_feishu_opaque_user_id("ou_hidden_sender") is True
    assert is_probably_feishu_opaque_user_id("on_hidden_union") is True
    assert is_probably_feishu_opaque_user_id("Alice") is False
    assert suppress_opaque_feishu_user_id(sender_name) == (
        expected if sender_name and not sender_name.startswith(("ou_", "on_")) else ""
    )
    assert coerce_feishu_sender_display_name(sender_name=sender_name, sender_id=sender_id) == expected
