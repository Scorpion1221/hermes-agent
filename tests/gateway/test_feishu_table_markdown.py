"""Tests for the fork's Feishu Card 2.0 markdown payload construction.

Reproduces the bug tracked in hermes-agent issue #52786:
All fork messages use a Card 2.0 ``markdown`` element.  Tables, plain text,
and mixed markdown must stay on that one path without losing content.

These tests guard the fix.  They invoke the real adapter via the project's
plugin-loader helper so that no ``sys.path`` / ``sys.modules`` games are
needed.
"""

from __future__ import annotations

import json

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_adapter = load_plugin_adapter("feishu")


def _call_build_outbound_payload(content: str) -> tuple[str, str]:
    """Invoke ``_build_outbound_payload`` on a bare adapter instance.

    ``_build_outbound_payload`` is a method that only uses module-level
    helpers (``_MARKDOWN_TABLE_RE``, ``_MARKDOWN_HINT_RE``,
    ``_build_markdown_post_payload``) and never touches ``self.*``, so a bare
    object is sufficient.
    """
    inst = object.__new__(_adapter.FeishuAdapter)
    return inst._build_outbound_payload(content)


def _markdown_texts_from_card(payload_str: str) -> list[str]:
    """Return Card 2.0 markdown element contents."""
    payload = json.loads(payload_str)
    assert payload.get("schema") == "2.0"
    return [
        element.get("content", "")
        for element in payload.get("body", {}).get("elements", [])
        if isinstance(element, dict) and element.get("tag") == "markdown"
    ]


def test_markdown_table_uses_card_not_text():
    """Regression test for issue #52786 (and its older sibling #23938).

    A message whose only markdown is a table must take the ``post`` path,
    not be downgraded to plain text.
    """
    content = (
        "| col A | col B |\n"
        "| ----- | ----- |\n"
        "| 1     | 2     |"
    )
    msg_type, payload_str = _call_build_outbound_payload(content)
    assert msg_type == "interactive"
    md_texts = _markdown_texts_from_card(payload_str)
    assert md_texts, f"card payload must include a markdown element; got {payload_str!r}"
    joined = "".join(md_texts)
    assert "col A" in joined and "|" in joined, (
        "table text was lost or reformatted when switching from text to post"
    )


def test_plain_text_without_markdown_still_uses_card():
    content = "just a plain sentence with no markup"
    msg_type, payload = _call_build_outbound_payload(content)
    assert msg_type == "interactive"
    assert _markdown_texts_from_card(payload) == [content]


def test_existing_markdown_heading_still_uses_card():
    """Sanity: the existing ``post`` path (heading / list / code / bold /
    link) must still work after the table downgrade is removed."""
    msg_type, payload_str = _call_build_outbound_payload("# hello world\n")
    assert msg_type == "interactive"
    md_texts = _markdown_texts_from_card(payload_str)
    assert md_texts, f"expected at least one md element; got {payload_str!r}"
    assert any("hello world" in t for t in md_texts), (
        f"expected 'hello world' in md elements; got {md_texts!r}"
    )


def test_table_combined_with_other_markdown_does_not_downgrade():
    """A message that mixes a table with surrounding markdown must also
    take the ``post`` path.

    The old ``_MARKDOWN_TABLE_RE`` branch returned ``text`` unconditionally
    and stripped all the surrounding markdown formatting, so a Feishu
    reader saw literal pipes and lost the prose framing the table.
    """
    content = (
        "Here is the data:\n\n"
        "| col A | col B |\n"
        "| ----- | ----- |\n"
        "| 1     | 2     |\n\n"
        "Let me know."
    )
    msg_type, payload_str = _call_build_outbound_payload(content)
    assert msg_type == "interactive"
    md_texts = _markdown_texts_from_card(payload_str)
    joined = "\n".join(md_texts)
    assert "Here is the data" in joined, (
        "leading prose was lost when downgrading a mixed-table message"
    )
    assert "col A" in joined, "table header was lost"
    assert "Let me know" in joined, "trailing prose was lost"
