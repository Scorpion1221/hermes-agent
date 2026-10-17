"""Split a live streamed reply where a mid-turn follow-up (steer / redirect) enters the model's context.

The agent fires ``steer_consumed_callback`` on its worker thread right after the follow-up row is in
``messages``. Everything streamed before that point answered the earlier request; what streams after
it answers the follow-up, which the user sent below the live reply. An adapter whose stream IS the
message (Feishu CardKit) opts in with ``seal_live_reply(chat_id)``: the gateway flushes what is
streamed so far, the adapter seals it, and the next frame opens a fresh reply below the follow-up.
Adapters without the hook keep their single stream untouched.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any, Callable

logger = logging.getLogger("gateway.run")

_FLUSH_TIMEOUT_SECONDS = 3.0
_SEAL_TIMEOUT_SECONDS = 10.0


def _clarify_expired_notice() -> str:
    """Rendered on a native clarify card whose wait ended without a click (mirrors the notice the
    Slack click handler shows on a dead entry)."""
    from agent.i18n import t

    return t("gateway.clarify.expired")


def make_steer_consumed_callback(turn_runner: Any) -> Callable[[], None]:
    """The turn's ``agent.steer_consumed_callback``: blocking, runs on the agent worker thread."""

    def _on_steer_consumed() -> None:
        consumer = turn_runner._stream_consumer()
        adapter = getattr(consumer, "adapter", None)
        # Class lookup: a MagicMock adapter must not fabricate the hook.
        if adapter is None or getattr(type(adapter), "seal_live_reply", None) is None:
            return
        flush = getattr(consumer, "flush_pending_sync", None)
        if callable(flush):
            with suppress(Exception):
                flush(timeout=_FLUSH_TIMEOUT_SECONDS)  # what answered the earlier request goes on screen first
        fut = turn_runner._schedule(adapter.seal_live_reply(str(consumer.chat_id)), "seal_live_reply scheduling error")
        if fut is None:
            return
        try:
            fut.result(timeout=_SEAL_TIMEOUT_SECONDS)
        except Exception:
            logger.warning("Sealing the live reply after a mid-turn follow-up failed", exc_info=True)

    return _on_steer_consumed
