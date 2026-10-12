"""Assistant Markdown as a Feishu Card 2.0 ``markdown`` element: what gets rewritten and what never does.

Pure input -> output behaviour of ``feishu_card_render``: ATX headings go two levels down (capped at
``######``), a punctuation-ended strong span touching a word gets a space, mentions with placeholder ids
become plain ``@name``, and fenced code, indented code, inline code spans and URLs are left alone. Rendering
is render-only and deliberately not idempotent, so nothing here renders a result a second time. All names,
ids and texts are synthetic.
"""

from __future__ import annotations

import json

import pytest

from plugins.platforms.feishu.feishu_card_render import (
    has_card_mentions,
    markdown_card,
    render_markdown_for_card,
    strip_card_mentions,
)

PLAUSIBLE_ID = "ou_0123456789abcdef"


# ── headings ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("level", "rendered_level"),
    [(1, 3), (2, 4), (3, 5), (4, 6), (5, 6), (6, 6)],
)
def test_heading_goes_two_levels_down_capped_at_six(level, rendered_level):
    assert render_markdown_for_card(f"{'#' * level} Title") == f"{'#' * rendered_level} Title"


def test_heading_keeps_its_indent_and_text():
    assert render_markdown_for_card("  ## Plan **now**") == "  #### Plan **now**"


@pytest.mark.parametrize("line", ["#hashtag", "#!shebang", "####### seven marks", "text # not a heading"])
def test_line_that_is_not_an_atx_heading_is_untouched(line):
    assert render_markdown_for_card(line) == line


def test_heading_only_rewrites_heading_lines():
    assert render_markdown_for_card("# One\nbody\n## Two\n") == "### One\nbody\n#### Two\n"


# ── strong span boundaries ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("**Label:**body", "**Label:** body"),
        ("**Done!**Next", "**Done!** Next"),
        ("- **Step 1.**Install it", "- **Step 1.** Install it"),
        ("**A:**x and **B:**y", "**A:** x and **B:** y"),
        ("see https://example.com/page and **Label:**body", "see https://example.com/page and **Label:** body"),
    ],
)
def test_strong_span_ending_in_punctuation_gets_a_space_before_the_word(source, expected):
    assert render_markdown_for_card(source) == expected


def test_strong_span_ending_in_cjk_punctuation_gets_a_space():
    assert render_markdown_for_card("**标签：**正文") == "**标签：** 正文"


@pytest.mark.parametrize(
    "line",
    [
        "**Label**body",  # ends in a letter: not the punctuation quirk
        "**Label:** body",  # already separated
        "**Label:**",  # nothing follows
        "**Label:**\n",
        "**bold**, then text",  # the word after is not glued to the span
        "**标签：** 正文",
    ],
)
def test_strong_span_that_needs_no_space_is_untouched(line):
    assert render_markdown_for_card(line) == line


def test_strong_span_inside_inline_code_is_untouched():
    assert render_markdown_for_card("`**Label:**body` and **Other:**x") == "`**Label:**body` and **Other:** x"


def test_strong_span_inside_a_double_backtick_code_span_is_untouched():
    assert render_markdown_for_card("``a **Label:**body`` end") == "``a **Label:**body`` end"


def test_strong_span_inside_a_url_is_untouched():
    url = "https://example.com/a**b:**c"
    assert render_markdown_for_card(url) == url
    assert render_markdown_for_card(f"[link]({url}) and **Label:**body") == f"[link]({url}) and **Label:** body"


@pytest.mark.parametrize("prefix", ["    ", "\t"])
def test_indented_code_is_untouched(prefix):
    code = f"{prefix}# comment\n{prefix}**Label:**body\n"
    assert render_markdown_for_card(code) == code


# ── fenced code ─────────────────────────────────────────────────────────────

CODE_BODY = '# shell comment\n**Label:**body\n<at id="ou_?">Alice</at>\n'


def test_backtick_fence_content_is_untouched_and_text_around_it_is_rendered():
    source = f"# Before\n```python\n{CODE_BODY}```\n# After\n**Done:**ok\n"
    assert render_markdown_for_card(source) == f"### Before\n```python\n{CODE_BODY}```\n### After\n**Done:** ok\n"


def test_tilde_fence_content_is_untouched_and_a_backtick_line_does_not_close_it():
    source = f"~~~\n{CODE_BODY}```\n# still code\n~~~\n# After\n"
    assert render_markdown_for_card(source) == f"~~~\n{CODE_BODY}```\n# still code\n~~~\n### After\n"


def test_four_backtick_fence_is_closed_only_by_four_or_more():
    source = "````\n```\n# inside\n```\n````\n# After\n"
    assert render_markdown_for_card(source) == "````\n```\n# inside\n```\n````\n### After\n"
    assert render_markdown_for_card("````\n# a\n`````\n# b\n") == "````\n# a\n`````\n### b\n"


def test_unclosed_fence_keeps_everything_after_it_as_code():
    source = "```\n# never closed\n**Label:**body\n"
    assert render_markdown_for_card(source) == source


# ── line endings ────────────────────────────────────────────────────────────


def test_crlf_line_endings_are_preserved():
    source = "# Title\r\n**Label:**body\r\n```\r\n# code\r\n```\r\n# After\r\n"
    assert render_markdown_for_card(source) == "### Title\r\n**Label:** body\r\n```\r\n# code\r\n```\r\n### After\r\n"


def test_mixed_and_missing_final_line_endings_are_preserved():
    assert render_markdown_for_card("# One\r\n## Two\n### Three") == "### One\r\n#### Two\n##### Three"


# ── mentions ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "tag",
    [
        '<at id="ou_?">Alice</at>',
        '<at id="ou_…">Alice</at>',
        '<at id="ou_xxx">Alice</at>',
        '<at id="">Alice</at>',
        "<at>Alice</at>",
        "<person id='ou_?'>Alice</person>",
        '<person user_id="">Alice</person>',
    ],
)
def test_mention_with_a_placeholder_id_becomes_plain_name(tag):
    assert render_markdown_for_card(f"Hi {tag}, look") == "Hi @Alice, look"
    assert strip_card_mentions(f"Hi {tag}, look", only_invalid=True) == "Hi @Alice, look"


@pytest.mark.parametrize(
    "tag",
    [
        f'<at id="{PLAUSIBLE_ID}">Alice</at>',
        f'<at id="on_{PLAUSIBLE_ID[3:]}">Alice</at>',
        '<at id="all">everyone</at>',
        '<at user_id="u_12345">Alice</at>',
        '<at email="alice@example.com">Alice</at>',
        f'<person id="{PLAUSIBLE_ID}"></person>',
    ],
)
def test_mention_with_a_plausible_id_is_kept(tag):
    assert render_markdown_for_card(f"Hi {tag}, look") == f"Hi {tag}, look"
    assert strip_card_mentions(f"Hi {tag}, look", only_invalid=True) == f"Hi {tag}, look"


def test_only_invalid_rewrites_the_placeholder_and_keeps_the_real_mention_on_one_line():
    keep = f'<at id="{PLAUSIBLE_ID}">Bob</at>'
    source = f'<at id="ou_?">Alice</at> and {keep}'
    assert strip_card_mentions(source, only_invalid=True) == f"@Alice and {keep}"
    assert render_markdown_for_card(source) == f"@Alice and {keep}"


def test_strip_card_mentions_without_only_invalid_converts_every_mention():
    source = (
        f'<at id="{PLAUSIBLE_ID}">Alice</at> <person id="{PLAUSIBLE_ID}">Bob</person> '
        '<at id="all">everyone</at> <at id="ou_?">Carol</at>'
    )
    assert strip_card_mentions(source) == "@Alice @Bob @everyone @Carol"


def test_strip_card_mentions_does_not_spare_inline_code():
    assert strip_card_mentions('`<at id="ou_?">Alice</at>`') == "`@Alice`"


@pytest.mark.parametrize("only_invalid", [False, True])
def test_mention_inside_fenced_code_is_untouched(only_invalid):
    fenced = '```\n<at id="ou_?">Alice</at>\n<person>Bob</person>\n```\n'
    assert strip_card_mentions(fenced, only_invalid=only_invalid) == fenced
    assert strip_card_mentions(f'<at id="ou_?">Alice</at>\r\n{fenced}', only_invalid=only_invalid) == (
        f"@Alice\r\n{fenced}"
    )
    assert render_markdown_for_card(fenced) == fenced


def test_mention_rewrite_and_heading_rewrite_compose_on_the_same_line():
    assert render_markdown_for_card('# Ping <at id="ou_?">Alice</at>') == "### Ping @Alice"


# ── card body ───────────────────────────────────────────────────────────────


def test_markdown_card_is_a_card_2_0_body_with_one_rendered_markdown_element():
    source = '# Title\n**Label:**body\n<at id="ou_?">Alice</at>\n```\n# code\n```\n'
    card = json.loads(markdown_card(source))
    assert card == {
        "schema": "2.0",
        "body": {"elements": [{"tag": "markdown", "content": render_markdown_for_card(source)}]},
    }
    assert card["body"]["elements"][0]["content"] == "### Title\n**Label:** body\n@Alice\n```\n# code\n```\n"


def test_markdown_card_keeps_non_ascii_text_unescaped():
    payload = markdown_card("# 标题\n**标签：**正文")
    assert "标题" in payload and "\\u" not in payload
    assert json.loads(payload)["body"]["elements"][0]["content"] == "### 标题\n**标签：** 正文"


# --- review fixes: delimiter pairing, fence info strings, multi-line mentions ---


@pytest.mark.parametrize("text", [
    "**状态**：**已完成**",
    "**苹果**、**香蕉**",
    "| **状态**：**完成** |",
    "**x**:**y**z",
])
def test_adjacent_strong_spans_keep_both_bold(text):
    """A closing ``**`` is never reused as an opener, so the second span is left intact."""
    assert render_markdown_for_card(text) == text


def test_every_glued_label_on_a_line_gets_its_space():
    assert render_markdown_for_card("**a:**b and **c:**d") == "**a:** b and **c:** d"


def test_a_one_line_code_span_with_backticks_does_not_open_a_fence():
    assert render_markdown_for_card("```js x``` then\n# H") == "```js x``` then\n### H"


def test_a_mention_tag_spanning_lines_is_rewritten_and_detected_consistently():
    text = '<at id="ou_abc12345678">\nAlice</at> hi'

    assert has_card_mentions(text)
    assert strip_card_mentions(text) == "@Alice hi"


def test_a_mention_only_inside_a_fence_does_not_count_as_a_mention():
    assert not has_card_mentions('```\n<at id="ou_?">x</at>\n```')
