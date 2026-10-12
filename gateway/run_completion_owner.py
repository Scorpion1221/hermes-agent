"""Which gateway conversation owns an async completion, for GatewayRunner.

Split out of ``gateway/run_notifications.py``; bound onto ``GatewayRunner`` via the MRO.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from gateway.session import SessionEntry

logger = logging.getLogger("gateway.run")


class GatewayCompletionOwnerMixin:
    """Resolve the session an async completion belongs to before it is delivered."""

    async def _get_completion_owner(self, session_id: str) -> tuple[str, Optional[dict]]:
        """Resolve a delegated execution to its durable conversation owner.

        Terminal watchers stamp the executing session, which may be a delegate child that inherited
        its parent's chat route. Such a child must never become the gateway conversation through
        switch_session(). Follow the explicit ``_delegate_from`` marker, not parent_session_id (also
        used by compression and /branch); compression continuations keep the marker. Resolved at
        consumption time so checkpoint-recovered watchers get the same answer.

        Missing or cyclic ownership fails closed (``row`` is None). DB/decoding errors propagate so
        the completion pre-flight can retry rather than acknowledge a loss.
        """
        seen = set()
        while session_id not in seen:
            seen.add(session_id)
            row = await self._session_db.get_session(session_id)
            if row is None:
                return session_id, None
            config = row.get("model_config") or {}
            if isinstance(config, str):
                config = json.loads(config)
            if not isinstance(config, dict):
                raise ValueError("Invalid completion-owner model_config")
            owner = config.get("_delegate_from")
            if owner is None or owner == "":
                return session_id, row
            if not isinstance(owner, str) or not owner.strip():
                raise ValueError("Invalid completion-owner delegation marker")
            if row.get("session_key") and await self._delegate_owns_its_route(session_id, row):
                return session_id, row
            session_id = owner.strip()
        logger.warning("Cyclic delegation ownership for completion session %s", session_id)
        return session_id, None

    async def _delegate_owns_its_route(self, session_id: str, row: dict) -> bool:
        """/resume can promote a delegate child to a gateway conversation. A session key alone is not
        proof: a delegate created after its parent compressed inherits one. Verify the live route."""
        from gateway.run import _USER_BOUNDARY_END_REASONS
        if row.get("ended_at") and row.get("end_reason") in _USER_BOUNDARY_END_REASONS:
            return True
        routed_id = await self.async_session_store.peek_session_id(row["session_key"])
        if routed_id == session_id:
            return True
        return bool(
            routed_id and row.get("end_reason") == "compression"
            and await self._session_db.get_compression_tip(session_id) == routed_id
        )

    async def _resolve_compression_lineage_target(
        self, session_db: Any, session_entry: SessionEntry, pinned_session_id: str,
    ) -> Optional[str]:
        """Return the live compression tip of ``pinned_session_id`` if the route owns that lineage, else None."""
        try:
            target_session_id = await session_db.get_compression_tip(pinned_session_id)
        except Exception:
            logger.debug("Async-delegation compression-tip lookup failed for %s", pinned_session_id, exc_info=True)
            target_session_id = None
        if not target_session_id or target_session_id == pinned_session_id:
            logger.warning(
                "Async-delegation completion pinned to compressed session %s "
                "without a continuation; dropping injection.", pinned_session_id,
            )
            return None
        try:
            tip_row = await session_db.get_session(target_session_id)
        except Exception:
            tip_row = None
        if tip_row is None or tip_row.get("ended_at"):
            logger.warning(
                "Async-delegation compression continuation %s is %s; dropping injection.",
                target_session_id, "unknown" if tip_row is None else "ended",
            )
            return None
        route_owns_lineage = session_entry.session_id in {pinned_session_id, target_session_id}
        if not route_owns_lineage:
            # Across several rotations, accept a stale route only when its own tip is the same live target.
            try:
                route_row = await session_db.get_session(session_entry.session_id)
                route_tip = (
                    await session_db.get_compression_tip(session_entry.session_id)
                    if route_row is not None
                    and route_row.get("ended_at")
                    and route_row.get("end_reason") == "compression"
                    else None
                )
            except Exception:
                route_tip = None
            route_owns_lineage = route_tip == target_session_id
        if not route_owns_lineage:
            logger.warning(
                "Async-delegation completion for compression lineage %s -> %s "
                "does not own current route %s; dropping injection.",
                pinned_session_id, target_session_id, session_entry.session_id,
            )
            return None
        return target_session_id
