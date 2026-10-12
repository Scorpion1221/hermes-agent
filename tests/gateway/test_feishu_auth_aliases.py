"""A Feishu sender authorizes under any of their ids (open_id / user_id / union_id).

Feishu reports up to three ids per sender, and the adapter's primary flips to the tenant-scoped user_id
once the app gains that scope; allowlists and pairing approvals written earlier hold the open_id. The
same person must keep their access. Driven through the real GatewayRunner authorization and the real
adapter sender-profile resolution.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.session import Platform, SessionSource
from plugins.platforms.feishu.adapter import FeishuAdapter


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("FEISHU_ALLOWED_USERS", "FEISHU_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS"):
        monkeypatch.delenv(var, raising=False)


def _runner(approved: set[str] = frozenset()):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda platform, uid: uid in approved)
    return runner


def _source_for(open_id: str, user_id: str, union_id: str) -> SessionSource:
    adapter = FeishuAdapter(PlatformConfig())

    async def _no_name(*_a, **_k):
        return None

    adapter._resolve_sender_name_from_api = _no_name
    sender = SimpleNamespace(open_id=open_id, user_id=user_id, union_id=union_id)
    profile = asyncio.run(adapter._resolve_sender_profile(sender))
    source = adapter.build_source(chat_id="oc_dm", chat_type="dm", user_id=profile["user_id"],
                                  user_name=None, user_id_alt=profile["user_id_alt"])
    source.auth_aliases = profile["auth_aliases"]
    return source


def test_an_allowlist_written_with_the_open_id_still_admits_the_sender(monkeypatch):
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_alice")
    source = _source_for("ou_alice", "u_tenant_alice", "on_alice")

    assert source.user_id == "u_tenant_alice"  # the primary id flipped to the tenant-scoped one
    assert _runner()._is_user_authorized(source) is True


def test_a_pairing_approved_under_the_open_id_still_counts():
    source = _source_for("ou_alice", "u_tenant_alice", "on_alice")

    assert _runner(approved={"ou_alice"})._is_user_authorized(source) is True


def test_someone_else_is_still_refused(monkeypatch):
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_alice")
    source = _source_for("ou_mallory", "u_tenant_mallory", "on_mallory")

    assert _runner()._is_user_authorized(source) is False


def test_a_source_without_aliases_is_judged_on_its_user_id_only(monkeypatch):
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_alice")
    source = SessionSource(platform=Platform.FEISHU, chat_id="oc_dm", chat_type="dm", user_id="u_tenant_alice")

    assert _runner()._is_user_authorized(source) is False
