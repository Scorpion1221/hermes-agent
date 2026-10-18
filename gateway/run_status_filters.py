"""Chat-status matchers derived from the agent's status template constants.

``_prepare_gateway_status_message`` (gateway/run.py) uses them to decide which agent statuses reach a
chat. Built from the SAME constants the emit sites format, so rewording a status cannot drift from
its matcher. Imports nothing from ``gateway.run``.
"""

import re

from agent.conversation_compression import (
    COMPACTION_DONE_STATUS, COMPACTION_HEARTBEAT_STATUS, COMPACTION_STATUS, COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE,
    COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE, COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE,
    COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE, IDLE_COMPACTION_STATUS_TEMPLATE,
    PRE_API_COMPRESSION_STATUS_TEMPLATE, PREFLIGHT_COMPRESSION_STATUS_TEMPLATE)
from agent.turn_facade_lease import LEASE_FREE_STATUS, LEASE_WAIT_TICK_STATUS_TEMPLATE


def _status_template_to_regex(template: str) -> str:
    """Compile a compression status template constant into a regex source.

    Literal text is escaped verbatim (wording drift can't diverge from the matcher); ``{field}`` -> numeric."""
    parts = re.split(r"\{[^{}]*\}", template)
    return r"[\d,]+".join(re.escape(part) for part in parts)


# ROUTINE compression progress statuses, derived from the SAME template constants the emit sites format.
# Used ONLY by the opt-in ``compression.progress_notices`` gate in gateway/run.py (#52995) to decide which
# of the noisy statuses matched by _TELEGRAM_NOISY_STATUS_RE are compression progress (deliverable when the
# user opted in) versus unrelated aux/retry chatter (always suppressed on chat surfaces). Failure notices and
# manual /compress feedback never match _TELEGRAM_NOISY_STATUS_RE in the first place, so they are unaffected
# by this gate.
_COMPRESSION_PROGRESS_STATUS_RE = re.compile(
    "|".join(
        _status_template_to_regex(_template)
        for _template in (
            COMPACTION_STATUS, COMPACTION_HEARTBEAT_STATUS, COMPACTION_DONE_STATUS, PRE_API_COMPRESSION_STATUS_TEMPLATE,
            PREFLIGHT_COMPRESSION_STATUS_TEMPLATE, IDLE_COMPACTION_STATUS_TEMPLATE,
            COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE, COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE,
            COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE,
            COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE)),
    re.IGNORECASE)

# Repeats of a durable turn-lease wait (a tick every 15s for up to 30 min, then "session is free"). Chats keep
# only the first notice: adapters without send_or_update_status post every status as a new message.
_LEASE_WAIT_REPEAT_STATUS_RE = re.compile(
    _status_template_to_regex(LEASE_WAIT_TICK_STATUS_TEMPLATE) + "|" + re.escape(LEASE_FREE_STATUS), re.IGNORECASE)
