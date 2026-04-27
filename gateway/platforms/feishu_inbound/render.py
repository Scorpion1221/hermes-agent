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

    # Collect the chain oldest → newest so the rendered block reads top-down in
    # chronological order: the root (earliest) message at the top, the
    # immediately-quoted message last. Matches how humans scan a thread.
    chain: list[FeishuQuotedContext] = []
    node: Optional[FeishuQuotedContext] = quoted_context
    while node is not None:
        chain.append(node)
        node = node.parent
    chain.reverse()

    lines = ["[Quoted message context]"]
    total = len(chain)
    for idx, ctx in enumerate(chain):
        # Older → newer: root = 0, most recent (the one being directly quoted)
        # = total - 1. The last one is the actual reply target; everything
        # above it is ancestor context.
        is_root = idx == 0 and total > 1
        is_tail = idx == total - 1
        hops_from_current = total - 1 - idx
        if total == 1:
            header = None
        elif is_root:
            header = f"[Root of quote chain, {hops_from_current} hop(s) above current]"
        elif is_tail:
            header = "[Directly quoted message]"
        else:
            header = f"[Ancestor quote, {hops_from_current} hop(s) above current]"
        if idx > 0:
            lines.append("")
        if header:
            lines.append(header)
        _render_one(ctx, lines, found_in_history=found_in_history and is_tail)

    if image_analysis.strip():
        lines.append("")
        lines.append("image_analysis:")
        lines.append(image_analysis.strip())
    if quoted_context.media_urls:
        lines.append(f"media_count: {len(tuple(quoted_context.media_urls))}")

    lines.append("[/Quoted message context]")
    return "\n".join(lines)


def _render_one(
    ctx: FeishuQuotedContext,
    lines: list[str],
    *,
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

