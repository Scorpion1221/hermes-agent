"""Assistant Markdown rendered as a Feishu Card JSON 2.0 body (outbound, render-only).

Card 2.0's ``markdown`` element renders the full standard set, GFM tables included, which a post's ``md``
element does not render consistently. Two card quirks are repaired here: a top-level heading renders very
large inside a card, so ATX headings go two levels down (``#`` -> ``###``, capped at ``######``); and a strong
span that ends in punctuation and touches the next word (``**Label:**body``) stays literal in CommonMark and
in cards, so it gets a space (``**Label:** body``). Fenced code, indented code, inline code spans and URLs are
never touched.

A mention whose id Feishu cannot resolve makes it refuse the whole card (100290), so an ``<at>`` / ``<person>``
tag with a placeholder id (``ou_?``, ``ou_…``, none) is rendered as plain ``@name``, which is what a post or a
text message shows for such a mention anyway. Mentions inside fenced code stay verbatim.

Render-only: callers pass the raw assistant text and never store the result back into conversation or
streaming state (rendering it again would push the headings down twice). The streaming card path renders its
card bodies through ``render_markdown_for_card`` too.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterator

_FENCE_OPEN_RE = re.compile(r"^[ ]{0,3}(?P<fence>`{3,}|~{3,}).*$")
_ATX_HEADING_RE = re.compile(r"^(?P<indent>[ ]{0,3})(?P<marks>#{1,6})(?P<rest>(?:[ \t]+.*)?)$")
_INLINE_CODE_RE = re.compile(r"(?<!\\)(?P<fence>`+)(?!`)[^\r\n]*?(?P=fence)(?!`)")
_STRONG_BOUNDARY_RE = re.compile(
    r"(?<![*\\])\*\*(?!\*)"
    r"(?P<label>[^`\\*\r\n]*?)(?P<last>[^`\\*\s\r\n])\*\*"
    r"(?=[^\W])"
)
_MENTION_TAG_RE = re.compile(
    r"<(?P<tag>at|person)\b(?P<attrs>[^>]*?)(?:/>|>(?P<inner>.*?)</(?P=tag)\s*>)", re.IGNORECASE | re.DOTALL,
)
_MENTION_ID_RE = re.compile(r"\b(?:id|ids|user_id|open_id|email)\s*=\s*[\"']?(?P<value>[^\"'\s>]*)", re.IGNORECASE)
# Ids Feishu can resolve: an open/union id, ``all``, a user_id, an email. Anything else is a placeholder.
_MENTION_ID_VALUE_RE = re.compile(
    r"^(?:all|o[un]_[A-Za-z0-9]{8,}|(?!o[un]_)[A-Za-z0-9][A-Za-z0-9_-]{3,}|[^@\s]+@[^@\s]+\.[A-Za-z]{2,})$"
)
HEADING_LEVELS_DOWN = 2


def _split_line_ending(line: str) -> tuple[str, str]:
    for ending in ("\r\n", "\n", "\r"):
        if line.endswith(ending):
            return line[: -len(ending)], ending
    return line, ""


def _lines(content: str) -> Iterator[tuple[str, str, bool]]:
    """``(line, ending, is_code)`` for each line; a fence line and every line inside a fence is code."""
    fence = ""
    for raw_line in content.splitlines(keepends=True):
        line, ending = _split_line_ending(raw_line)
        if fence:
            if re.match(rf"^[ ]{{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$", line):
                fence = ""
            yield line, ending, True
            continue
        opening = _FENCE_OPEN_RE.match(line)
        if opening:
            fence = opening.group("fence")
        yield line, ending, bool(opening)


def _code_spans(line: str) -> list[tuple[int, int]]:
    return [match.span() for match in _INLINE_CODE_RE.finditer(line)]


def _in_spans(spans: list[tuple[int, int]], position: int) -> bool:
    return any(start <= position < end for start, end in spans)


def _inside_url_token(line: str, position: int) -> bool:
    token_start = position
    while token_start > 0 and not line[token_start - 1].isspace():
        token_start -= 1
    token_prefix = line[token_start:position].lower()
    return "://" in token_prefix or token_prefix.startswith("www.")


def _space_strong_boundaries(line: str) -> str:
    """Separate a punctuation-ended ``**`` span from the word glued to it, outside code and URLs."""
    if line.startswith(("    ", "\t")):  # indented code
        return line
    spans = _code_spans(line)
    pieces: list[str] = []
    cursor = 0
    for match in _STRONG_BOUNDARY_RE.finditer(line):
        if (not unicodedata.category(match.group("last")).startswith("P")
                or _in_spans(spans, match.start()) or _inside_url_token(line, match.start())):
            continue
        pieces += [line[cursor:match.end()], " "]
        cursor = match.end()
    return "".join(pieces) + line[cursor:] if pieces else line


def _downshift_heading(line: str) -> str:
    match = _ATX_HEADING_RE.match(line)
    if not match:
        return line
    level = min(len(match.group("marks")) + HEADING_LEVELS_DOWN, 6)
    return f"{match.group('indent')}{'#' * level}{match.group('rest')}"


def _mention_is_plausible(attrs: str) -> bool:
    ids = [match.group("value") for match in _MENTION_ID_RE.finditer(attrs or "")]
    return bool(ids) and all(_MENTION_ID_VALUE_RE.match(value) for raw in ids for value in raw.split(","))


def _mentions_as_text(line: str, *, only_invalid: bool) -> str:
    def _plain(match: re.Match[str]) -> str:
        if only_invalid and _mention_is_plausible(match.group("attrs")):
            return match.group(0)
        name = (match.group("inner") or "").strip()
        return f"@{name}" if name else ""

    return _MENTION_TAG_RE.sub(_plain, line)


def has_card_mentions(content: str) -> bool:
    return bool(content) and _MENTION_TAG_RE.search(content) is not None


def strip_card_mentions(content: str, *, only_invalid: bool = False) -> str:
    """``<at>`` / ``<person>`` mention tags outside fenced code as plain ``@name`` text.

    An inline code span is not spared: whether Feishu resolves a tag there is not known, and a refused
    one rejects the whole card.

    ``only_invalid`` keeps a mention whose id looks real and rewrites only placeholders; without it every
    mention goes, which is the retry after Feishu refused one of them (100290).
    """
    return "".join(
        line + ending if code else _mentions_as_text(line, only_invalid=only_invalid) + ending
        for line, ending, code in _lines(content)
    )


def render_markdown_for_card(content: str) -> str:
    """Raw assistant Markdown as a card's ``markdown`` element shows it best (see the module docstring)."""
    rendered: list[str] = []
    for line, ending, code in _lines(content):
        if not code:
            line = _space_strong_boundaries(_downshift_heading(_mentions_as_text(line, only_invalid=True)))
        rendered.append(line + ending)
    return "".join(rendered)


def markdown_card(content: str) -> str:
    """An ``interactive`` message body: a Card JSON 2.0 card with one ``markdown`` element of ``content``."""
    element = {"tag": "markdown", "content": render_markdown_for_card(content)}
    return json.dumps({"schema": "2.0", "body": {"elements": [element]}}, ensure_ascii=False)
