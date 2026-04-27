from __future__ import annotations

from typing import Iterable, Optional

from .types import FeishuQuotedContext


_QUOTE_CHAIN_MAX_DEPTH = 3


def render_quoted_context_block(
    quoted_context: Optional[FeishuQuotedContext],
    *,
    found_in_history: bool = False,
    image_analysis: str = "",
) -> str:
    if not quoted_context:
        return ""

    lines = ["[Quoted message context]"]
    _render_one(quoted_context, lines, level=0, found_in_history=found_in_history)

    if image_analysis.strip():
        lines.append("image_analysis:")
        lines.append(image_analysis.strip())
    if quoted_context.media_urls:
        lines.append(f"media_count: {len(tuple(quoted_context.media_urls))}")

    # Walk ancestor chain (the message being quoted was itself a reply).
    ancestor = quoted_context.parent
    depth = 1
    while ancestor is not None and depth < _QUOTE_CHAIN_MAX_DEPTH:
        lines.append("")
        lines.append(f"[Ancestor quote, depth={depth}]")
        _render_one(ancestor, lines, level=depth, found_in_history=False)
        ancestor = ancestor.parent
        depth += 1

    lines.append("[/Quoted message context]")
    return "\n".join(lines)


def _render_one(
    ctx: FeishuQuotedContext,
    lines: list[str],
    *,
    level: int,
    found_in_history: bool,
) -> None:
    summary = (ctx.display_text or "").strip()
    if found_in_history and len(summary) > 200:
        summary = summary[:200]
    if ctx.kind:
        lines.append(f"type: {ctx.kind}")
    if ctx.message_id:
        lines.append(f"message_id: {ctx.message_id}")
    if ctx.sender_name:
        lines.append(f"sender: {ctx.sender_name}")
    if summary:
        lines.append(f"summary: {summary}")

