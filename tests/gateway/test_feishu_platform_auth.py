from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def test_feishu_allowlist_matches_open_id_alias_when_primary_id_is_tenant_id(
    monkeypatch,
):
    """Legacy open_id allowlists must keep working when Feishu provides user_id."""
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_legacy")
    monkeypatch.delenv("FEISHU_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)

    runner = GatewayRunner(GatewayConfig())
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_args: False)

    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_dm",
        chat_type="dm",
        user_id="433ff9f5",
        user_name="Alice",
    )
    source.auth_user_ids = ["433ff9f5", "ou_legacy"]

    assert runner._is_user_authorized(source) is True


def test_feishu_pairing_matches_open_id_alias_when_primary_id_is_tenant_id(
    monkeypatch,
):
    """Existing pairing approvals keyed by open_id should survive user_id hydration."""
    monkeypatch.delenv("FEISHU_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("FEISHU_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)

    runner = GatewayRunner(GatewayConfig())
    runner.pairing_store = SimpleNamespace(
        is_approved=lambda platform, user_id: platform == "feishu" and user_id == "ou_legacy"
    )

    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_dm",
        chat_type="dm",
        user_id="433ff9f5",
        user_name="Alice",
    )
    source.auth_user_ids = ["433ff9f5", "ou_legacy"]

    assert runner._is_user_authorized(source) is True


@pytest.mark.asyncio
async def test_feishu_sender_profile_keeps_all_sender_ids_for_auth(monkeypatch):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig())
    monkeypatch.setattr(
        adapter,
        "_resolve_sender_name_from_api",
        AsyncMock(return_value="Alice"),
    )
    sender_id = SimpleNamespace(
        open_id="ou_legacy",
        user_id="433ff9f5",
        union_id="on_union",
    )

    profile = await adapter._resolve_sender_profile(sender_id)

    assert profile["user_id"] == "433ff9f5"
    assert profile["user_id_alt"] == "on_union"
    assert profile["auth_user_ids"] == ("433ff9f5", "ou_legacy", "on_union")


@pytest.mark.asyncio
async def test_feishu_group_event_admitted_by_platform_acl_skips_gateway_user_allowlist(
    monkeypatch,
    tmp_path,
):
    """Feishu per-group ACL is authoritative for admitted group messages.

    Regression: a group message could pass ``platforms.feishu.extra.group_rules``
    in the adapter, then get silently dropped by the generic gateway
    ``FEISHU_ALLOWED_USERS`` check.
    """
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_owner")
    monkeypatch.delenv("FEISHU_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)

    runner = GatewayRunner(GatewayConfig())

    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_group",
        chat_type="group",
        user_id="ou_stranger",
        user_name="stranger",
    )
    event = MessageEvent(text="hello", message_type=MessageType.TEXT, source=source)
    event.platform_auth_passed = True

    auth_called = False

    def _deny_auth(_self, _source):
        nonlocal auth_called
        auth_called = True
        return False

    async def _sentinel(_self, _event, *_args):
        raise RuntimeError("sentinel: reached agent dispatch")

    monkeypatch.setattr(GatewayRunner, "_is_user_authorized", _deny_auth)
    monkeypatch.setattr(GatewayRunner, "_handle_message_with_agent", _sentinel)

    with pytest.raises(RuntimeError, match="sentinel"):
        await runner._handle_message(event)

    assert auth_called is False


@pytest.mark.asyncio
async def test_feishu_process_inbound_group_message_marks_platform_auth_passed(
    monkeypatch,
):
    """A Feishu group message that made it past _admit carries the auth marker."""
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter(
        PlatformConfig(
            extra={
                "group_rules": {
                    "oc_group": {
                        "policy": "open",
                    },
                },
            }
        )
    )
    monkeypatch.setattr(
        adapter,
        "_extract_message_content",
        AsyncMock(return_value=("hello", MessageType.TEXT, [], [], [])),
    )
    monkeypatch.setattr(
        adapter,
        "get_chat_info",
        AsyncMock(return_value={"name": "Group", "type": "group"}),
    )
    monkeypatch.setattr(
        adapter,
        "_resolve_sender_profile",
        AsyncMock(
            return_value={
                "user_id": "u_stranger",
                "user_name": "stranger",
                "user_id_alt": None,
                "auth_user_ids": ("u_stranger", "ou_stranger"),
            }
        ),
    )
    captured: list[MessageEvent] = []

    async def _capture(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "_dispatch_inbound_event", _capture)

    message = SimpleNamespace(
        chat_id="oc_group",
        chat_type="group",
        message_type="text",
        thread_id=None,
        parent_id=None,
        root_id=None,
        upper_message_id=None,
    )
    sender_id = SimpleNamespace(open_id="ou_stranger", user_id="u_stranger", union_id=None)

    await adapter._process_inbound_message(
        data=SimpleNamespace(),
        message=message,
        sender_id=sender_id,
        chat_type="group",
        message_id="om_1",
        is_bot=False,
    )

    assert len(captured) == 1
    assert getattr(captured[0], "platform_auth_passed", False) is True
    assert captured[0].source.auth_user_ids == ["u_stranger", "ou_stranger"]
