"""A Feishu sender admitted under an alias id is still authorized when its stored admission drains.

The durable FIFO snapshots the inbound source as JSON (``prepare_native``) and the drain re-checks
the stored sender before running it (``restore_native`` + ``_validate_native``). Real runner,
real Feishu adapter, real authority store; only the Feishu network is absent.
"""
import json

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_envelope import _validate_native, prepare_native, restore_native


async def _feishu_authority(tmp_path, monkeypatch, allowed):
    from plugins.platforms.feishu.adapter import FeishuAdapter
    for var in ("FEISHU_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", allowed)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter = FeishuAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner, runner.adapters = runner, {Platform.FEISHU: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="named")
    runner._wire_adapter_handlers(adapter)
    return authority, adapter


def _feishu_source(adapter, *, chat_type="group"):
    # The adapter's primary id is the tenant user_id; the allowlist was written with the open_id.
    source = adapter.build_source(chat_id="oc_team", chat_type=chat_type, user_id="u_tenant_alice",
                                  user_name="Alice", user_id_alt="on_alice")
    source.auth_aliases = ("ou_alice", "on_alice")
    return source


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", ["group", "dm"])
async def test_an_alias_admitted_sender_passes_the_drain_recheck(tmp_path, monkeypatch, chat_type):
    authority, adapter = await _feishu_authority(tmp_path, monkeypatch, "ou_alice")
    try:
        event = MessageEvent(text="hi", source=_feishu_source(adapter, chat_type=chat_type), message_id="om_1")
        payload = json.loads(json.dumps(await prepare_native(authority.runner, event)))  # as the row stores it
        stored = restore_native(payload, authority.runner)

        assert stored.source.auth_aliases == ("ou_alice", "on_alice")
        _validate_native(authority.runner, stored)  # raised permission_denied before the fix
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_a_sender_no_id_of_which_is_allowed_is_still_refused(tmp_path, monkeypatch):
    from hermes_state_runtime import RuntimeStoreError
    authority, adapter = await _feishu_authority(tmp_path, monkeypatch, "ou_bob")
    try:
        with pytest.raises(RuntimeStoreError) as raised:
            await prepare_native(authority.runner, MessageEvent(text="hi", source=_feishu_source(adapter),
                                                                message_id="om_2"))
        assert raised.value.reason == "permission_denied"
    finally:
        authority.db.close()
