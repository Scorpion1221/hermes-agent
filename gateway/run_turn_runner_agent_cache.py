"""Agent-cache lookups for a gateway turn, split out of ``run_turn_runner``.

The cache is keyed by session_key; each entry is ``(agent, signature, message_count, session_id)``.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any


def cached_sid_is_dead(runner: Any, ctx: Any, cache_lock: Any, cache: Any) -> tuple:
    """(peeked cached session_id, is_dead) — checked OUTSIDE the cache lock. "cached sid != current
    sid" normally means an intentional switch (reuse), but the routing-key self-heal yields the same
    shape with an agent bound to a DEAD session; reusing it re-binds the dead sid and loops."""
    peek_sid = None
    if cache_lock and cache is not None:
        with cache_lock:
            entry = cache.get(ctx.session_key)
        if entry and len(entry) > 3:
            peek_sid = entry[3]
    dead = False
    if peek_sid is not None and ctx.session_id is not None and peek_sid != ctx.session_id:
        # The cache is keyed by session_key, so the snapshot's row lives in that key's profile
        # store; its id is no longer in the routing index once the self-heal moved the key on.
        with suppress(Exception):
            dead = runner.session_store._is_session_ended_in_db(peek_sid, session_key=ctx.session_key)
    return peek_sid, dead
