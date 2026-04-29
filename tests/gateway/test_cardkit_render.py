"""CardKit json_card renderer tests.

Fixtures under tests/gateway/fixtures/cardkit/ are real raw body.content
strings captured from Feishu's im.v1.message.get with
card_msg_content_type=raw_card_content. Keeping real samples around gives us
coverage over the full AST schema (plain_text with textStyle, heading, table,
code_block, blockquote, nested markdown containers, ...) without needing to
trigger a live reproduction against Feishu.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway.platforms.feishu_inbound.parse import (
    normalize_feishu_message,
    render_cardkit_body,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "cardkit"


def _load(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def _render(raw: str) -> str:
    return normalize_feishu_message(message_type="interactive", raw_content=raw).text_content


class TestCardKitRichTableHeading:
    """Mixed paragraph + code_block + heading + table + code_span."""

    def setup_method(self) -> None:
        self.rendered = _render(_load("rich_table_heading.json"))

    def test_bold_inline_attaches_to_text(self):
        assert "**完整原始结构**" in self.rendered

    def test_heading_is_standalone_block(self):
        assert "\n##### 观察分析\n" in self.rendered

    def test_code_block_fenced(self):
        assert "```\ntype: interactive" in self.rendered
        assert "\n```" in self.rendered

    def test_table_rendered_as_markdown(self):
        assert "| 段落 | 推测原始元素 |" in self.rendered
        assert "| --- | --- |" in self.rendered

    def test_code_span_backticked(self):
        assert "`能！Sir，这条引用我`" in self.rendered

    def test_no_element_ids_leak(self):
        assert "_2_0" not in self.rendered
        assert "_2_1" not in self.rendered

    def test_no_textalign_leak(self):
        assert "\nleft\n" not in self.rendered
        assert "left\n\n" not in self.rendered


class TestCardKitBlockquoteTable:
    """Blockquote blocks + markdown table + heading."""

    def setup_method(self) -> None:
        self.rendered = _render(_load("blockquote_table.json"))

    def test_blockquotes_standalone(self):
        lines = self.rendered.splitlines()
        bq_lines = [l for l in lines if l.startswith("> ")]
        assert len(bq_lines) >= 3

    def test_terminal_blocks_isolated(self):
        for line in self.rendered.splitlines():
            if "terminal:" in line:
                assert line.startswith("> "), f"terminal call not in blockquote: {line!r}"

    def test_heading_lines(self):
        assert "##### 🚀 SUP-65 已上线" in self.rendered
        assert "##### 🔄 接下来 CEO 会做什么" in self.rendered.split("\n", 200)[-1] or any(
            "##### 🔄" in l for l in self.rendered.splitlines()
        )

    def test_table_rows_rendered(self):
        assert "| **Identifier** | **SUP-65** |" in self.rendered


class TestCardKitSyntheticElements:
    """Unit-level coverage for per-tag renderers using synthetic trees."""

    def _wrap(self, elements):
        return json.dumps({
            "json_card": json.dumps({
                "body": {"property": {"elements": elements}}
            })
        })

    def test_ordered_list(self):
        raw = self._wrap([{
            "tag": "ordered_list",
            "property": {"elements": [
                {"tag": "plain_text", "property": {"content": "first"}},
                {"tag": "plain_text", "property": {"content": "second"}},
            ]},
        }])
        assert _render(raw) == "1. first\n2. second"

    def test_bullet_list(self):
        raw = self._wrap([{
            "tag": "bullet_list",
            "property": {"elements": [
                {"tag": "plain_text", "property": {"content": "a"}},
                {"tag": "plain_text", "property": {"content": "b"}},
            ]},
        }])
        assert _render(raw) == "- a\n- b"

    def test_hr(self):
        raw = self._wrap([
            {"tag": "plain_text", "property": {"content": "above"}},
            {"tag": "hr", "property": {}},
            {"tag": "plain_text", "property": {"content": "below"}},
        ])
        assert _render(raw) == "above\n\n---\n\nbelow"

    def test_link(self):
        raw = self._wrap([{
            "tag": "a",
            "property": {"content": "docs", "href": "https://example.com"},
        }])
        assert _render(raw) == "[docs](https://example.com)"

    def test_mention_user(self):
        raw = self._wrap([{
            "tag": "at",
            "property": {"name": "Alice", "user_id": "u_1"},
        }])
        assert _render(raw) == "@Alice"

    def test_mention_all(self):
        raw = self._wrap([{
            "tag": "at",
            "property": {"id": "all"},
        }])
        assert _render(raw) == "@全体成员"

    def test_text_tag(self):
        raw = self._wrap([{
            "tag": "text_tag",
            "property": {"text": {"tag": "plain_text", "content": "URGENT"}},
        }])
        assert _render(raw) == "[URGENT]"

    def test_image_inline(self):
        raw = self._wrap([{
            "tag": "image",
            "property": {"alt": "pic", "img_key": "img_abc"},
        }])
        assert _render(raw) == "![pic](img_abc)"

    def test_font_wrapper_strips_color(self):
        raw = self._wrap([{
            "tag": "font",
            "property": {"elements": [
                {"tag": "plain_text", "property": {"content": "red text"}},
            ]},
        }])
        assert _render(raw) == "red text"

    def test_strikethrough(self):
        raw = self._wrap([{
            "tag": "plain_text",
            "property": {"content": "deleted", "textStyle": {"attributes": ["strikethrough"]}},
        }])
        assert _render(raw) == "~~deleted~~"

    def test_unknown_tag_recurses(self):
        raw = self._wrap([{
            "tag": "future_widget",
            "property": {"elements": [
                {"tag": "plain_text", "property": {"content": "visible"}},
            ]},
        }])
        assert _render(raw) == "visible"


class TestCardKitFallback:
    """Non-json_card payloads still flow through the legacy walker."""

    def test_card_id_stub_returns_placeholder(self):
        raw = json.dumps({"type": "card", "data": {"card_id": "XYZ"}})
        assert _render(raw) == "[Interactive message]"

    def test_card_2_markdown(self):
        raw = json.dumps({
            "schema": "2.0",
            "body": {"elements": [{"tag": "markdown", "content": "Hello **world**"}]},
        })
        assert "Hello **world**" in _render(raw)


def test_render_cardkit_body_empty_input():
    assert render_cardkit_body(None) == ""
    assert render_cardkit_body({}) == ""
    assert render_cardkit_body({"body": {}}) == ""
