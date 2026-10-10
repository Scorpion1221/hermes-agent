"""Display name of whoever clicked a Feishu approval / update-prompt card button.

The SDK runs card callbacks synchronously, and Feishu wants the response — the resolved card, "Approved
by <name>" — within about 3 s. Over the websocket the callback runs on the SDK's own thread, so on a
name-cache miss it may wait a bounded time for the contact lookup on the adapter loop; a lookup that
overruns keeps going and warms the cache for the next click. The webhook transport calls the callback on
the adapter loop itself, where waiting would starve the very lookup it waits for, so only the cache
answers there. Without a name the card shows the open_id.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_LOOKUP_BUDGET_SECONDS = 1.5  # leaves headroom inside Feishu's ~3 s card-callback deadline


def card_operator_name(adapter: Any, operator: Any) -> str:
    """The operator's display name, else their open_id."""
    open_id = str(getattr(operator, "open_id", "") or "")
    user_id = str(getattr(operator, "user_id", "") or "")
    # Inbound messages cache names under user_id when the event carries one (``_resolve_sender_profile``).
    cached = adapter._get_cached_sender_name(user_id) or adapter._get_cached_sender_name(open_id)
    return cached or _looked_up_name(adapter, open_id) or open_id


def _looked_up_name(adapter: Any, open_id: str) -> Optional[str]:
    loop = adapter._loop
    if not open_id or _running_on(loop):
        return None
    from agent.async_utils import safe_schedule_threadsafe

    future = safe_schedule_threadsafe(
        adapter._resolve_sender_name_from_api(open_id), loop, logger=logger,
        log_message="[Feishu] Could not schedule the card operator name lookup", log_level=logging.WARNING,
    )
    if future is None:
        return None
    try:
        return future.result(timeout=_LOOKUP_BUDGET_SECONDS)
    except TimeoutError:
        logger.info("[Feishu] Card operator name lookup for %s took over %.1fs; showing the open_id",
                    open_id, _LOOKUP_BUDGET_SECONDS)
    except concurrent.futures.CancelledError:
        logger.info("[Feishu] Card operator name lookup for %s was cancelled (adapter stopping)", open_id)
    return None


def _running_on(loop: Any) -> bool:
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False
