"""When a Feishu message was sent: its own ``create_time``, not when Hermes finished reading it."""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
from typing import Any


def message_sent_at(message: Any) -> datetime:
    """When the user sent ``message`` (Feishu's ``create_time``, epoch ms). Attachment downloads can take
    seconds, so the time the event is built is not when it was sent; fall back to now when absent."""
    try:
        create_time_ms = int(str(getattr(message, "create_time", "") or ""))
    except ValueError:
        create_time_ms = 0
    if create_time_ms > 0:
        with contextlib.suppress(OverflowError, OSError, ValueError):
            return datetime.fromtimestamp(create_time_ms / 1000, timezone.utc)
    return datetime.now(timezone.utc)
