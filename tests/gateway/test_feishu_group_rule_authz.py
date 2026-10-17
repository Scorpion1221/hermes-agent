"""A Feishu group member admitted by that chat's own ``group_rules`` entry is answered, idle or busy.

The adapter gates group traffic with ``group_rules`` at intake, and the gateway then runs its user
allowlist again. Before the fix, with FEISHU_ALLOWED_USERS set to anyone else, a member admitted by an
``open`` rule or named in the chat's ``allowlist`` was silently ignored, both on a first message and on
a follow-up sent while the bot was answering someone. The grant stays narrow (SECURITY.md §2.6): only
an explicit rule for that chat counts, never the default policy or a blacklist, and DMs are unchanged.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType

OWNER, MEMBER, OUTSIDER = "ou_owner", "ou_member", "ou_outsider"
CHAT, OTHER_CHAT = "oc_ruled", "oc_unruled"
_AUTH_ENV = (
    "FEISHU_ALLOWED_USERS", "FEISHU_ALLOW_ALL_USERS", "FEISHU_GROUP_POLICY",
    "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS",
)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in _AUTH_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", OWNER)  # the bot's owner; group members are not listed
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")


def _gateway(group_rules: dict, **extra):
    """A real FeishuAdapter built from ``extra.group_rules``, wired to a real GatewayRunner auth gate."""
    from gateway.run import GatewayRunner
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig(enabled=True, extra={"group_rules": group_rules, **extra}))
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.FEISHU: adapter.config})
    runner.adapters = {Platform.FEISHU: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    adapter.gateway_runner = runner
    return runner, adapter


def _event(adapter, user_id: str, *, chat_id: str = CHAT, chat_type: str = "group") -> MessageEvent:
    source = adapter.build_source(chat_id=chat_id, chat_type=chat_type, user_id=user_id, user_name=user_id)
    return MessageEvent(text="1+1", message_type=MessageType.TEXT, source=source, message_id=f"om_{user_id}")


def _admitted_by_intake(adapter, user_id: str, chat_id: str = CHAT) -> bool:
    """What the adapter itself decides for this sender in this chat (``_admit``'s group step)."""
    return adapter._allow_group_message(SimpleNamespace(open_id=user_id, user_id=None), chat_id)


def _idle_answered(runner, event) -> bool:
    """The idle ingress gate (``_hm_admit_event``): None means the message was dropped."""
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._hm_report_ignored_dm = AsyncMock()

    async def _passthrough(ev, _source):
        return ev

    runner._hm_pre_gateway_dispatch_hook = _passthrough
    return asyncio.run(runner._hm_admit_event(event)) is not None


def _busy_queued(runner, adapter, event) -> bool:
    """The busy gate (``_handle_active_session_busy_message``) while another member's turn runs:
    an admitted follow-up waits in the session's pending slot; a refused one never gets there."""
    key = adapter._event_session_key(event)
    runner._session_state(key).turn.agent = object()  # someone's turn is running
    runner._busy_input_mode = "queue"
    handled = asyncio.run(runner._handle_active_session_busy_message(event, key))
    assert handled is True  # either way the gateway owns it (queued, or dropped as unauthorized)
    return key in adapter._pending_messages


@pytest.mark.parametrize("rule, sender", [
    ({"policy": "open"}, OUTSIDER),
    ({"policy": "allowlist", "allowlist": [MEMBER]}, MEMBER),
], ids=["open-rule", "allowlist-rule"])
def test_member_admitted_by_the_chats_rule_is_answered_on_a_first_message(rule, sender):
    runner, adapter = _gateway({CHAT: rule})
    assert _admitted_by_intake(adapter, sender)

    assert _idle_answered(runner, _event(adapter, sender)) is True


@pytest.mark.parametrize("rule, sender", [
    ({"policy": "open"}, OUTSIDER),
    ({"policy": "allowlist", "allowlist": [MEMBER]}, MEMBER),
], ids=["open-rule", "allowlist-rule"])
def test_member_admitted_by_the_chats_rule_is_queued_while_the_bot_answers_someone(rule, sender):
    runner, adapter = _gateway({CHAT: rule})

    assert _busy_queued(runner, adapter, _event(adapter, sender)) is True


@pytest.mark.parametrize("rule, sender", [
    ({"policy": "allowlist", "allowlist": [MEMBER]}, OUTSIDER),  # not on this chat's list
    ({"policy": "blacklist", "blacklist": ["ou_blocked"]}, OUTSIDER),  # blacklist admits strangers: no grant
    ({"policy": "blacklist", "blacklist": [OUTSIDER]}, OUTSIDER),
    ({"policy": "admin_only"}, OUTSIDER),
    ({"policy": "disabled"}, OUTSIDER),
    ({"allowlist": [MEMBER]}, OUTSIDER),  # no explicit policy: intake reads it as open, the gateway must not
], ids=["allowlist-miss", "blacklist-other", "blacklisted", "admin-only", "disabled", "policy-omitted"])
def test_rules_that_do_not_name_the_sender_grant_nothing(rule, sender):
    runner, adapter = _gateway({CHAT: rule})

    assert _idle_answered(runner, _event(adapter, sender)) is False
    assert _busy_queued(runner, adapter, _event(adapter, sender)) is False


@pytest.mark.parametrize("rule", [{"policy": "admin_only"}, {"policy": "allowlist", "allowlist": [MEMBER]}],
                         ids=["admin-only", "allowlist-without-the-admin"])
def test_an_admin_is_answered_in_a_chat_whose_rule_admits_admins(rule):
    """Intake admits ``admins`` before a chat's rule: the gateway must not then drop them."""
    runner, adapter = _gateway({CHAT: rule}, admins=["ou_admin"])
    assert _admitted_by_intake(adapter, "ou_admin")

    assert _idle_answered(runner, _event(adapter, "ou_admin")) is True
    assert _busy_queued(runner, adapter, _event(adapter, "ou_admin")) is True


def test_a_rule_for_one_chat_grants_nothing_in_another():
    runner, adapter = _gateway({CHAT: {"policy": "open"}}, default_group_policy="open")
    assert _admitted_by_intake(adapter, OUTSIDER, OTHER_CHAT)  # the default policy admits at intake...

    # ...but it is not an operator-named grant, so the gateway's allowlist still decides.
    assert _idle_answered(runner, _event(adapter, OUTSIDER, chat_id=OTHER_CHAT)) is False
    assert _busy_queued(runner, adapter, _event(adapter, OUTSIDER, chat_id=OTHER_CHAT)) is False


def test_the_allowlist_still_answers_its_own_members_everywhere():
    runner, adapter = _gateway({CHAT: {"policy": "allowlist", "allowlist": [MEMBER]}})

    assert _idle_answered(runner, _event(adapter, OWNER, chat_id=OTHER_CHAT)) is True
    assert _idle_answered(runner, _event(adapter, OWNER, chat_type="dm", chat_id="oc_dm")) is True


def test_a_group_rule_never_reaches_dms():
    """A member named on a group's allowlist is still a stranger in a DM."""
    runner, adapter = _gateway({CHAT: {"policy": "open"}, "oc_dm": {"policy": "open"}})

    assert _idle_answered(runner, _event(adapter, OUTSIDER, chat_id="oc_dm", chat_type="dm")) is False
    assert _busy_queued(runner, adapter, _event(adapter, OUTSIDER, chat_id="oc_dm", chat_type="dm")) is False


def test_removing_the_rule_revokes_a_queued_members_access():
    """The grant is read on every check, never captured on the event."""
    runner, adapter = _gateway({CHAT: {"policy": "allowlist", "allowlist": [MEMBER]}})
    event = _event(adapter, MEMBER)
    assert _idle_answered(runner, event) is True

    adapter._group_rules.pop(CHAT)

    assert runner._is_user_authorized_for_source(event.source) is False


def test_stored_routes_do_not_inherit_the_adapter_grant():
    """Plugin injection re-checks a stored route without adapter-time grants (like Discord roles)."""
    runner, adapter = _gateway({CHAT: {"policy": "open"}})
    source = _event(adapter, OUTSIDER).source

    assert runner._is_user_authorized_for_source(source) is True
    assert runner._is_user_authorized_for_source(source, allow_adapter_delegation=False) is False
