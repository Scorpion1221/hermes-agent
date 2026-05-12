from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


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
    from gateway.platforms.feishu import FeishuAdapter

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
