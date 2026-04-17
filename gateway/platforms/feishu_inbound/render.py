from __future__ import annotations

from typing import Iterable, Optional

from .types import FeishuQuotedContext


def render_quoted_context_block(
    quoted_context: Optional[FeishuQuotedContext],
    *,
    found_in_history: bool = False,
    image_analysis: str = "",
) -> str:
    if not quoted_context:
        return ""

    summary = (quoted_context.display_text or "").strip()
    if found_in_history and len(summary) > 200:
        summary = summary[:200]

    lines = ["[Quoted message context]"]
    if quoted_context.kind:
        lines.append(f"type: {quoted_context.kind}")
    if quoted_context.message_id:
        lines.append(f"message_id: {quoted_context.message_id}")
    if quoted_context.sender_name:
        lines.append(f"sender: {quoted_context.sender_name}")
    if summary:
        lines.append(f"summary: {summary}")
    if image_analysis.strip():
        lines.append("image_analysis:")
        lines.append(image_analysis.strip())
    if quoted_context.media_urls:
        lines.append(f"media_count: {len(tuple(quoted_context.media_urls))}")
    lines.append("[/Quoted message context]")
    return "\n".join(lines)
