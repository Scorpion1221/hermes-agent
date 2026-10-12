"""Busy-text debounce state for ``BasePlatformAdapter`` (one buffered burst per session)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.platforms.event import MessageEvent


@dataclass
class TextDebounceState:
    event: "MessageEvent"
    task: asyncio.Task | None
    first_ts: float
    last_ts: float

    def cancel_timer(self, *, unless: asyncio.Task | None = None) -> None:
        """Cancel the pending flush timer (if live and not ``unless``)."""
        if self.task is not None and self.task is not unless and not self.task.done():
            self.task.cancel()
