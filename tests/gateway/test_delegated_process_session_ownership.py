"""Process-completion batching keeps each spawning session's completions apart.

Delegate-completion ownership itself (a child's completion reaching its human-facing parent,
boundaries, cycles, compression) is upstream's ``run_notifications_ownership`` and is covered by
``test_async_delegation_session_binding.py`` and ``test_completion_owner_retry.py``.
"""

from gateway.run import GatewayRunner


def test_process_batches_keep_different_spawning_sessions_separate():
    """Two completions for one chat but different spawning sessions (before and after /new) must not
    coalesce: the batch is delivered under its first event's owner, which would drop or misroute the other."""
    route = {"session_key": "agent:main:feishu:group:test", "platform": "feishu", "chat_id": "test"}

    def key(evt):
        return GatewayRunner._event_route_key(evt, GatewayRunner._COMPLETION_BATCH_KEY_FIELDS)

    assert key({**route, "parent_session_id": "old-session"}) != key({**route, "parent_session_id": "new-session"})
    assert key({**route, "parent_session_id": "same"}) == key({**route, "parent_session_id": "same"})
