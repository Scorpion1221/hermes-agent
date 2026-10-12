"""A cron delivery rendered as a notification card (``render_mode: card``).

Cron wraps a job's output as ``Cronjob Response: <name>`` / ``(job_id: <id>)`` / a rule, the output, then a
"To stop or manage this job ..." line (cron/scheduler_delivery.py ``_deliver_result``). That exact wrapper is
the only signal both delivery paths carry: the live adapter path also passes ``job_id`` in metadata, the
standalone path passes nothing. When the text is that wrapper, the card shows the job name in its header,
the output as the body, and the notification time, the job id and how to manage the job in its footer.
Anything else, including ``cron.wrap_response: false``, renders as an ordinary reply.

Only what cron sends is shown: it does not pass the run's status or duration, so the card claims neither.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

# Mirrors the f-string in cron/scheduler_delivery.py. Newline runs may have been collapsed to two by media
# extraction, so the separators accept one or more blank lines.
_CRON_WRAPPER_RE = re.compile(
    r"\ACronjob Response: (?P<name>[^\n]+)\n"
    r"\(job_id: (?P<job_id>[^)\n]*)\)\n"
    r"-------------\n+"
    r"(?P<body>.*?)\n+"
    r'To stop or manage this job, send me a new message \(e\.g\. "stop reminder (?P=name)"\)\.\Z',
    re.DOTALL,
)


@dataclass(frozen=True)
class CronDelivery:
    name: str
    job_id: str
    body: str


def parse_cron_delivery(text: str) -> Optional[CronDelivery]:
    match = _CRON_WRAPPER_RE.match(text or "")
    if not match:
        return None
    return CronDelivery(match.group("name").strip(), match.group("job_id").strip(), match.group("body").strip())


def cron_card(delivery: CronDelivery, body: str, *, header: bool = True, footer: bool = True) -> str:
    """An ``interactive`` payload for one chunk of a cron delivery; long output spans several cards."""
    from agent.i18n import t
    from hermes_time import now

    from .feishu_card_render import render_markdown_for_card

    elements: list[dict] = [{"tag": "markdown", "content": render_markdown_for_card(body)}]
    if footer:
        lines = [t("platform.feishu.cron.notified_at", time=now().strftime("%Y-%m-%d %H:%M:%S %z"))]
        if delivery.job_id:
            lines.append(t("platform.feishu.cron.job_id", job_id=delivery.job_id))
        elements += [
            {"tag": "hr"},
            {"tag": "markdown", "text_size": "notation", "content": "\n".join(lines)},
            {"tag": "markdown", "text_size": "notation", "content": t("platform.feishu.cron.manage_hint")},
        ]
    card: dict = {"schema": "2.0", "config": {"wide_screen_mode": True}, "body": {"elements": elements}}
    if header:
        card["header"] = {
            "template": "blue",
            "title": {"tag": "plain_text", "content": f"⏰ {delivery.name}"},
            "subtitle": {"tag": "plain_text", "content": t("platform.feishu.cron.subtitle")},
        }
    return json.dumps(card, ensure_ascii=False)
