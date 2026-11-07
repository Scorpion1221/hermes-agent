"""A Feishu group member let in by the chat's own ``group_rules`` entry is answered, not refused at
admission. Mirrors the cloud bot's setup (2026-11-07): the default group policy is ``allowlist``,
FEISHU_ALLOWED_USERS holds open_ids, and the team chats carry an explicit ``policy: open`` rule.
Real runner, real Feishu adapter, real authority store, real ``prepare_native``.
"""
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_envelope import prepare_native
from gateway.session_ingress_context import native_callback, register_transport_home

OPEN_CHAT, OTHER_CHAT = "oc_team_open", "oc_other"
ADMIN = ("ou_admin", "433ff9f5", "on_admin")      # (open_id, tenant user_id, union_id)
MEMBER = ("ou_member", "77a2d58a", "on_member")    # not on any allowlist; the chat rule admits them


async def _authority(tmp_path, monkeypatch):
    from plugins.platforms.feishu.adapter import FeishuAdapter
    for var in ("FEISHU_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", ADMIN[0])
    monkeypatch.setenv("FEISHU_GROUP_POLICY", "allowlist")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # the authority's home is the transport home
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)  # admit only; never run a turn
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / "state.db")
    runner.session_store._db = runner._session_db = db
    adapter = FeishuAdapter(PlatformConfig(enabled=True, extra={
        "admins": [ADMIN[0]], "default_group_policy": "allowlist",
        "group_rules": {OPEN_CHAT: {"policy": "open", "require_mention": True}}}))
    adapter.gateway_runner, runner.adapters = runner, {Platform.FEISHU: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="named", db=db)
    runner._wire_adapter_handlers(adapter)
    register_transport_home(runner, None, tmp_path)  # what the default handler does at wiring
    return authority, adapter, tmp_path


async def _admit_as_intake(authority, home, event):
    """``prepare_native`` inside the adapter callback the gateway wraps every inbound message in."""
    with native_callback(authority.runner, event, home):
        return await prepare_native(authority.runner, event)


def _event(adapter, who, chat):
    open_id, user_id, union_id = who
    source = adapter.build_source(chat_id=chat, chat_type="group", user_id=user_id, user_name="x",
                                  user_id_alt=union_id)
    source.auth_aliases = (open_id, union_id)
    return MessageEvent(text="hi", source=source, message_id=f"om_{user_id}_{chat}")


@pytest.mark.asyncio
@pytest.mark.parametrize("who", [ADMIN, MEMBER], ids=["admin", "rule_member"])
async def test_intake_and_admission_agree_in_an_open_rule_chat(tmp_path, monkeypatch, who):
    authority, adapter, home = await _authority(tmp_path, monkeypatch)
    try:
        event = _event(adapter, who, OPEN_CHAT)
        assert authority.runner._is_user_authorized_for_source(event.source) is True  # intake admits
        await _admit_as_intake(authority, home, event)  # admission must not refuse what intake admitted
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_a_member_outside_any_rule_is_still_refused(tmp_path, monkeypatch):
    from hermes_state_runtime import RuntimeStoreError
    authority, adapter, home = await _authority(tmp_path, monkeypatch)
    try:
        event = _event(adapter, MEMBER, OTHER_CHAT)  # no rule for this chat; the default policy never grants
        assert authority.runner._is_user_authorized_for_source(event.source) is False
        with pytest.raises(RuntimeStoreError) as raised:
            await _admit_as_intake(authority, home, event)
        assert raised.value.reason == "permission_denied"
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_a_stored_route_without_provenance_gets_no_adapter_grant(tmp_path, monkeypatch):
    """Plugin injection / stored routes carry no callback provenance: the chat rule must not widen them."""
    from hermes_state_runtime import RuntimeStoreError
    authority, adapter, _home = await _authority(tmp_path, monkeypatch)
    try:
        with pytest.raises(RuntimeStoreError) as raised:
            await prepare_native(authority.runner, _event(adapter, MEMBER, OPEN_CHAT))
        assert raised.value.reason == "permission_denied"
    finally:
        authority.db.close()
