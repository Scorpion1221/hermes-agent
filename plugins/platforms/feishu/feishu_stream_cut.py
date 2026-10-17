"""Where a long CardKit reply is cut when its live card rolls over while the reply is streaming.

``feishu_stream`` rolls a card over when it gets too old or too long, or when Feishu closed its streaming
window. The sealed card keeps what it showed up to its last paragraph boundary, and the unfinished
paragraph, table or code block continues in the next card. A heading stays with the paragraph it
introduces. Without a paragraph boundary the cut falls on a line boundary, and only a single line longer
than a card is cut mid-line. A cut inside a fenced code block closes the block in the sealed card and
re-opens it, with the same opener line, in the next one; when the block closes right at the cut, the
closing fence stays in the sealed card instead.

Fences are line-leading ``` / ~~~ runs, read the way the card renderer reads them, so a fence quoted
inside a line (a ``grep '```'`` progress line) is not one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .feishu_card_render import _FENCE_OPEN_RE

_HEADING_RE = re.compile(r"^[ ]{0,3}#{1,6}(?:[ \t]|$)")
_FENCE_RUN_RE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})")


@dataclass(frozen=True)
class Cut:
    head: str  # the part of the card's text the sealed card keeps (the card's lead excluded)
    opener: str = ""  # the fence opener line the next card re-opens; "" when the cut is outside code


@dataclass(frozen=True)
class _Line:
    start: int
    end: int  # just past the line ending
    text: str  # without the line ending
    opener: str  # opener line of the code block open after this line; "" outside code
    opener_end: int  # end of that opener line


def _closes(line: str, fence: str) -> bool:
    return bool(re.match(rf"^[ ]{{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$", line))


def _fence(opener: str) -> str:
    match = _FENCE_RUN_RE.match(opener)
    return match.group(1) if match else ""


def _lines(text: str) -> list[_Line]:
    lines: list[_Line] = []
    opener, opener_end, pos = "", 0, 0
    for raw in text.splitlines(keepends=True):
        line, end = raw.rstrip("\r\n"), pos + len(raw)
        if opener:
            if _closes(line, _fence(opener)):
                opener = ""
        elif _FENCE_OPEN_RE.match(line):
            opener, opener_end = line.strip(), end
        lines.append(_Line(pos, end, line, opener, opener_end))
        pos = end
    return lines


def _opener_at(lines: list[_Line], pos: int) -> str:
    """The code block open at ``pos``: after the last line that ends at or before it."""
    opener = ""
    for line in lines:
        if line.end > pos:
            break
        opener = line.opener
    return opener


def _lone_heading_before(lines: list[_Line], index: int) -> bool:
    """The paragraph ending before blank line ``index`` is a single heading line."""
    j = index - 1
    while j >= 0 and not lines[j].text.strip():
        j -= 1
    return j >= 0 and bool(_HEADING_RE.match(lines[j].text)) and (j == 0 or not lines[j - 1].text.strip())


def _paragraph_cut(text: str, lines: list[_Line], start: int, limit: int) -> Optional[int]:
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        if line.end > limit or line.text.strip() or line.opener or not text[start:line.end].strip():
            continue
        if line.start >= start and not _lone_heading_before(lines, index):
            return line.end
    return None


def _line_cut(text: str, lines: list[_Line], start: int, limit: int) -> Optional[int]:
    for line in reversed(lines):
        complete = text[line.end - 1:line.end] == "\n"
        # Never a cut that seals only a code block's opener line.
        if complete and start < line.end <= limit and (not line.opener or line.end > line.opener_end):
            return line.end
    return None


def cut_for_rollover(text: str, start: int, end: int, *, budget: int) -> Cut:
    """Cut a card's ``text`` (its lead, then its part of the reply from ``start``) for a rollover.

    The sealed card keeps at most ``budget`` characters and nothing past ``end`` (what the card showed);
    ``text`` may run on past ``end`` with the newest frame, which only tells where a paragraph or a block
    there ends."""
    lines = _lines(text)
    limit = min(end, start + budget)
    cut = _paragraph_cut(text, lines, start, limit) or _line_cut(text, lines, start, limit) or limit
    opener = _opener_at(lines, cut)
    if opener:
        closing = next((line for line in lines if line.start == cut), None)
        if closing is not None and _closes(closing.text, _fence(opener)):
            cut, opener = closing.end, ""  # the block closes right here: seal its closing fence too
    return Cut(head=text[start:cut], opener=opener)


def closed(text: str) -> str:
    """``text`` with a code block it ends inside closed, so a sealed card renders balanced code."""
    lines = _lines(text)
    if not lines or not lines[-1].opener:
        return text
    return f"{text.rstrip(chr(10))}\n{_fence(lines[-1].opener)}"


def closing_fence(text: str, lead: str) -> str:
    """The line ``text`` starts with when it closes the code block ``lead`` re-opens; "" otherwise."""
    fence = _fence(lead.strip())
    line, newline, _ = text.partition("\n")
    return line + newline if fence and _closes(line, fence) else ""


def open_block(text: str) -> str:
    """The opener line of the code block ``text`` ends inside; "" when it ends outside code."""
    return _opener_at(_lines(text), len(text))
