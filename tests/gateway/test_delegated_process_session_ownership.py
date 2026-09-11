"""A delegated process must never /resume its child into the gateway chat.

Exercise real terminal spawning, SQLite lineage, watcher delivery and session
routing. Only the platform network and model turn are replaced.
"""

import asyncio
import json
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import pytest_asyncio

from agent.delegation_context import delegated_child_context
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import GatewayRunner
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from gateway.session_context import clear_session_vars, set_session_vars
from hermes_state import AsyncSessionDB
from tools import async_delegation as ad


class _Adapter(BasePlatformAdapter):
    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, *args, **kwargs):
        return SendResult(success=True)

    async def get_chat_info(self, chat_id):
        return {"name": "test", "type": "group"}


@pytest_asyncio.fixture
async def gateway(tmp_path, monkeypatch, request):
    import tools.process_registry as pr

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(pr, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = pr.ProcessRegistry()
    monkeypatch.setattr(pr, "process_registry", registry)
    ad._reset_for_tests()
    platform = getattr(request, "param", Platform.FEISHU)
    config = GatewayConfig()
    store = SessionStore(tmp_path / "sessions", config)
    source = SessionSource(
        platform=platform, chat_id="test-group", chat_type="group", user_id="test-user",
    )
    parent = store.get_or_create_session(source).session_id
    key = store.get_or_create_session(source).session_key
    child = "delegated-backend"
    store._db.create_session(
        child, source=platform.value, parent_session_id=parent,
        model_config={"_delegate_from": parent},
    )
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._session_db = AsyncSessionDB(store._db)
    runner._session_source_cache = {}
    runner._completion_delivery_lock = threading.Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    runner._background_tasks = set()
    adapter = _Adapter(PlatformConfig(typing_indicator=False), platform)
    runner.adapters = {platform: adapter}
    consumed = asyncio.Queue()

    async def handle(event):
        current = await runner.async_session_store.get_or_create_session(
            event.source, touch_activity=False,
        )
        resolved = await runner._resolve_async_delegation_session(
            current, event.metadata["gateway_session_id"],
        )
        await consumed.put(resolved.session_id if resolved else None)

    adapter.set_message_handler(handle)
    try:
        yield SimpleNamespace(
            runner=runner, store=store, db=store._db, source=source,
            parent=parent, child=child, key=key, registry=registry,
            consumed=consumed, home=tmp_path,
        )
    finally:
        await runner._cancel_process_completion_batch_tasks()
        await asyncio.gather(*list(adapter._session_tasks.values()))
        registry.kill_all()
        store.close_all_db_handles()
        ad._reset_for_tests()


@pytest.mark.asyncio
@pytest.mark.parametrize("gateway", [Platform.FEISHU, Platform.TELEGRAM], indirect=True)
async def test_child_process_completion_preserves_parent_and_delegate_result(gateway):
    from tools.terminal_tool import terminal_tool

    g = gateway
    dispatched_at = time.time()
    ad._persist_dispatch({
        "delegation_id": "backend", "session_key": g.key,
        "parent_session_id": g.parent, "dispatched_at": dispatched_at,
    })
    tokens = set_session_vars(
        platform=g.source.platform.value, source=g.source.platform.value,
        chat_id=g.source.chat_id, chat_type=g.source.chat_type,
        user_id=g.source.user_id, session_key=g.key, session_id=g.parent,
        cwd=str(g.home),
    )
    try:
        def run_child():
            with delegated_child_context(g.child):
                return terminal_tool(
                    command="sleep 0.1; printf 'DEPLOY_DONE\\n'", background=True,
                    notify_on_complete=True, task_id="sa-0-test",
                    session_id=g.child, workdir=str(g.home),
                )

        result = json.loads(await asyncio.to_thread(run_child))
    finally:
        clear_session_vars(tokens)
    assert result["notify_on_complete"] is True
    proc = g.registry.get(result["session_id"])
    # The durable stamp is the execution owner, not the gateway route owner.
    assert proc.parent_session_id == g.child

    async def exited():
        while not proc.exited:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(exited(), 5)
    # A serialized/replayed watcher must work without any child ContextVar.
    watcher = json.loads(json.dumps(g.registry.pending_watchers.pop()))
    watcher["check_interval"] = 0
    await asyncio.wait_for(g.runner._run_process_watcher(watcher), 5)
    assert await asyncio.wait_for(g.consumed.get(), 5) == g.parent
    assert g.store.get_or_create_session(g.source).session_id == g.parent
    assert g.db.get_session(g.parent)["ended_at"] is None
    assert g.db.get_session(g.parent)["end_reason"] is None

    event = {
        "type": "async_delegation", "delegation_id": "backend",
        "session_key": g.key, "parent_session_id": g.parent,
        "status": "completed", "summary": "Backend verified",
        "dispatched_at": dispatched_at, "completed_at": time.time(),
    }
    ad._persist_completion(event, {"status": "completed", "summary": event["summary"]})
    assert await g.runner._deliver_completion_notification(event["summary"], event) is True
    assert await asyncio.wait_for(g.consumed.get(), 5) == g.parent
    record = ad.get_durable_delegation("backend")
    assert record["delivery_state"] == "delivered"
    assert record["delivery_attempts"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("end_reason", ["session_reset", "user_exit", "session_switch", "new_session"])
async def test_child_cannot_bypass_parent_user_boundary(gateway, end_reason):
    g = gateway
    g.db.end_session(g.parent, end_reason=end_reason)
    assert await g.runner._classify_completion_target(g.child) == "terminal"
    current = g.store._entries[g.key]
    assert await g.runner._resolve_async_delegation_session(current, g.child) is None
    assert current.session_id == g.parent
    assert g.db.get_session(g.parent)["end_reason"] == end_reason


@pytest.mark.asyncio
async def test_nested_and_compressed_child_follows_parent_compression(gateway):
    g = gateway
    g.db.create_session(
        "nested-child", source="feishu", parent_session_id=g.child,
        model_config={"_delegate_from": g.child},
    )
    for parent, tip, config in [
        ("nested-child", "nested-tip", {"_delegate_from": g.child}),
        (g.parent, "parent-tip", {}),
    ]:
        g.db.publish_compression_child(
            parent_session_id=parent, child_session_id=tip, source="feishu",
            messages=[{"role": "user", "content": "Preserved context"}],
            model_config=config, require_compression_lease=False,
        )
    g.db.end_session(g.child, end_reason="completed")
    assert await g.runner._classify_completion_target("nested-tip") == "deliver"
    resolved = await g.runner._resolve_async_delegation_session(
        g.store._entries[g.key], "nested-tip",
    )
    assert resolved.session_id == "parent-tip"
    assert g.store._entries[g.key].session_id == "parent-tip"
    assert g.db.get_session(g.parent)["end_reason"] == "compression"


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["missing-parent", "delegated-backend"])
async def test_missing_or_cyclic_delegate_ownership_fails_closed(gateway, owner):
    g = gateway
    g.db.patch_session_model_config(g.child, {"_delegate_from": owner})
    assert await g.runner._classify_completion_target(g.child) == "terminal"
    assert await g.runner._resolve_async_delegation_session(
        g.store._entries[g.key], g.child,
    ) is None
    assert g.db.get_session(g.parent)["ended_at"] is None


@pytest.mark.asyncio
async def test_child_cannot_override_an_unrelated_live_route(gateway):
    g = gateway
    unrelated = SessionSource(platform=Platform.FEISHU, chat_id="other", chat_type="group")
    current = g.store.get_or_create_session(unrelated)
    assert await g.runner._resolve_async_delegation_session(current, g.child) is None
    assert g.store.get_or_create_session(unrelated).session_id == current.session_id
    assert g.db.get_session(current.session_id)["ended_at"] is None


@pytest.mark.asyncio
async def test_explicitly_resumed_child_remains_a_gateway_conversation(gateway):
    g = gateway
    # Unlike an inherited chat route, explicit /resume publishes durable
    # gateway identity on the child. Notifications must respect that choice.
    current = g.store.switch_session(g.key, g.child)
    assert g.db.get_session(g.child)["session_key"] == g.key
    assert await g.runner._classify_completion_target(g.child) == "deliver"
    assert await g.runner._resolve_async_delegation_session(current, g.child) is current
    # Switching back to the root must not revive late work from the child.
    g.store.switch_session(g.key, g.parent)
    assert await g.runner._classify_completion_target(g.child) == "terminal"


@pytest.mark.asyncio
async def test_child_created_after_parent_compression_is_not_a_gateway_owner(gateway):
    g = gateway
    g.db.publish_compression_child(
        parent_session_id=g.parent, child_session_id="parent-tip", source="feishu",
        messages=[{"role": "user", "content": "Compressed parent"}],
        require_compression_lease=False,
    )
    # An async worker can create its row after the parent rotates. SessionDB's
    # compression-origin backfill then supplies a key even to this delegate.
    g.db.create_session(
        "late-child", source="feishu", parent_session_id=g.parent,
        model_config={"_delegate_from": g.parent},
    )
    assert g.db.get_session("late-child")["session_key"] == g.key
    resolved = await g.runner._resolve_async_delegation_session(
        g.store._entries[g.key], "late-child",
    )
    assert resolved.session_id == "parent-tip"
    assert g.store._entries[g.key].session_id == "parent-tip"
    assert g.db.get_session(g.parent)["end_reason"] == "compression"


@pytest.mark.asyncio
async def test_owner_lookup_failure_is_retryable(gateway, monkeypatch):
    g = gateway
    original = g.runner._session_db.get_session

    async def lookup(session_id):
        if session_id == g.parent:
            raise OSError("Session database temporarily unavailable")
        return await original(session_id)

    monkeypatch.setattr(g.runner._session_db, "get_session", lookup)
    assert await g.runner._classify_completion_target(g.child) == "retry"
    assert await g.runner._resolve_async_delegation_session(
        g.store._entries[g.key], g.child,
    ) is None
    assert g.db.get_session(g.parent)["ended_at"] is None


def test_process_batches_keep_different_spawning_sessions_separate():
    route = {"session_key": "agent:main:feishu:group:test", "platform": "feishu", "chat_id": "test"}
    assert GatewayRunner._completion_notification_batch_key(
        {**route, "parent_session_id": "old-child"},
    ) != GatewayRunner._completion_notification_batch_key(
        {**route, "parent_session_id": "new-parent"},
    )
