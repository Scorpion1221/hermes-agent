from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter, _build_card_v2_payload
from gateway.platforms.feishu_inbound.cardkit import (
    STREAMING_ELEMENT_ID,
    CardKitState,
    build_card_id_message_content,
    build_final_card_body,
    build_streaming_card_body,
    create_streaming_card,
    render_markdown_for_card,
    set_card_streaming_mode,
    stream_card_element,
    update_card,
)


def test_build_streaming_card_body_has_streaming_mode_and_element_id():
    body = build_streaming_card_body()
    assert body["config"]["streaming_mode"] is True
    element_ids = [e.get("element_id") for e in body["body"]["elements"]]
    assert STREAMING_ELEMENT_ID in element_ids


def test_build_final_card_body_disables_streaming():
    body = build_final_card_body("hello world")
    assert body["config"]["streaming_mode"] is False
    assert body["body"]["elements"][0]["content"] == "hello world"


def test_build_final_card_body_downshifts_markdown_headings():
    body = build_final_card_body(
        "# H1\n## H2\n### H3\n#### H4\n- **A｜先止血：**将外层 timeout 对齐"
    )
    assert body["body"]["elements"][0]["content"] == (
        "### H1\n#### H2\n##### H3\n###### H4\n"
        "- **A｜先止血：** 将外层 timeout 对齐"
    )


def test_render_markdown_for_card_downshifts_markdown_headings():
    rendered = render_markdown_for_card("# H1\n## H2\n### H3\n#### H4")
    assert rendered == "### H1\n#### H2\n##### H3\n###### H4"


def test_render_markdown_for_card_spaces_cardkit_strong_boundary():
    rendered = render_markdown_for_card(
        "- **A｜先止血：**将外层 timeout 对齐\n"
        "1. **B｜再优化：**取消前置 gate\n"
        "业务日志。**建议前端处理：**收到 Deck 详情 404 后停止请求\n"
        "- **已有空格：** 保持不变\n"
        "- **正常加粗**，继续说明\n"
        "[docs](https://example.com/a**b**c)\n"
        "https://example.com/a**b:**c\n"
        "x**2**nd\n"
        "`**inline:**code`\n"
        "    - **copyable:**command\n"
        "2. **paragraph-continuation:**body\n"
        "1234567890. **not-a-list:**body\n"
        "- **Use `foo**bar`：**body\n"
        "- **literal\\**body"
    )
    assert rendered == (
        "- **A｜先止血：** 将外层 timeout 对齐\n"
        "1. **B｜再优化：** 取消前置 gate\n"
        "业务日志。**建议前端处理：** 收到 Deck 详情 404 后停止请求\n"
        "- **已有空格：** 保持不变\n"
        "- **正常加粗**，继续说明\n"
        "[docs](https://example.com/a**b**c)\n"
        "https://example.com/a**b:**c\n"
        "x**2**nd\n"
        "`**inline:**code`\n"
        "    - **copyable:**command\n"
        "2. **paragraph-continuation:** body\n"
        "1234567890. **not-a-list:** body\n"
        "- **Use `foo**bar`：**body\n"
        "- **literal\\**body"
    )


def test_cardkit_strong_boundary_fix_ignores_fenced_code_blocks():
    rendered = render_markdown_for_card(
        "```md\n- **A｜先止血：**将外层 timeout 对齐\n```\n"
        "- **A｜先止血：**将外层 timeout 对齐"
    )
    assert rendered == (
        "```md\n- **A｜先止血：**将外层 timeout 对齐\n```\n"
        "- **A｜先止血：** 将外层 timeout 对齐"
    )


def test_cardkit_strong_boundary_fix_does_not_extend_unmatched_code_span():
    rendered = render_markdown_for_card(
        "`unmatched code span\n\n- **A｜先止血：**将外层 timeout 对齐"
    )
    assert rendered == (
        "`unmatched code span\n\n- **A｜先止血：** 将外层 timeout 对齐"
    )


def test_unmatched_code_span_does_not_steal_later_markdown_block():
    rendered = render_markdown_for_card(
        "`typo\n\n"
        "# Heading\n\n"
        "- **A｜先止血：**将外层 timeout 对齐\n\n"
        "Later `code` here."
    )
    assert rendered == (
        "`typo\n\n"
        "### Heading\n\n"
        "- **A｜先止血：** 将外层 timeout 对齐\n\n"
        "Later `code` here."
    )


def test_build_card_v2_payload_downshifts_markdown_headings_before_send():
    payload = json.loads(
        _build_card_v2_payload("# H1\n## H2\n- **A｜先止血：**将外层 timeout 对齐")
    )
    assert payload["body"]["elements"][0]["content"] == (
        "### H1\n#### H2\n- **A｜先止血：** 将外层 timeout 对齐"
    )


def test_card_heading_downshift_ignores_fenced_code_blocks():
    payload = json.loads(_build_card_v2_payload("```md\n# code\n```\n# Real"))
    assert payload["body"]["elements"][0]["content"] == "```md\n# code\n```\n### Real"


def test_build_card_id_message_content_format():
    content = build_card_id_message_content("card_abc123")
    parsed = json.loads(content)
    assert parsed == {"type": "card", "data": {"card_id": "card_abc123"}}


@pytest.mark.asyncio
async def test_create_streaming_card_returns_card_id():
    mock_resp = SimpleNamespace(
        success=lambda: True,
        data=SimpleNamespace(card_id="ck_test_123"),
    )
    client = SimpleNamespace(
        cardkit=SimpleNamespace(v1=SimpleNamespace(
            card=SimpleNamespace(create=lambda req: mock_resp),
        )),
    )
    card_id = await create_streaming_card(client)
    assert card_id == "ck_test_123"


@pytest.mark.asyncio
async def test_create_streaming_card_returns_none_on_failure():
    mock_resp = SimpleNamespace(success=lambda: False, code=500, msg="error")
    client = SimpleNamespace(
        cardkit=SimpleNamespace(v1=SimpleNamespace(
            card=SimpleNamespace(create=lambda req: mock_resp),
        )),
    )
    card_id = await create_streaming_card(client)
    assert card_id is None


@pytest.mark.asyncio
async def test_stream_card_element_returns_true_on_success():
    calls = []
    mock_resp = SimpleNamespace(success=lambda: True)
    client = SimpleNamespace(
        cardkit=SimpleNamespace(v1=SimpleNamespace(
            card_element=SimpleNamespace(content=lambda req: calls.append(req) or mock_resp),
        )),
    )
    ok = await stream_card_element(
        client, card_id="ck_1", element_id=STREAMING_ELEMENT_ID,
        content="# H1\n## H2\n- **A｜先止血：**将外层 timeout 对齐",
        sequence=1,
    )
    assert ok is True
    assert calls[0].request_body.content == (
        "### H1\n#### H2\n- **A｜先止血：** 将外层 timeout 对齐"
    )


@pytest.mark.asyncio
async def test_stream_card_element_silently_skips_rate_limit():
    mock_resp = SimpleNamespace(success=lambda: False, code=230020, msg="rate limit")
    client = SimpleNamespace(
        cardkit=SimpleNamespace(v1=SimpleNamespace(
            card_element=SimpleNamespace(content=lambda req: mock_resp),
        )),
    )
    ok = await stream_card_element(
        client, card_id="ck_1", element_id=STREAMING_ELEMENT_ID,
        content="hello", sequence=1,
    )
    assert ok is True


@pytest.mark.asyncio
async def test_update_card_sends_final_body():
    calls = []
    def mock_update(req):
        calls.append(req)
        return SimpleNamespace(success=lambda: True)

    client = SimpleNamespace(
        cardkit=SimpleNamespace(v1=SimpleNamespace(
            card=SimpleNamespace(update=mock_update),
        )),
    )
    body = build_final_card_body("done")
    ok = await update_card(client, card_id="ck_1", card_body=body, sequence=5)
    assert ok is True
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_set_card_streaming_mode_toggle():
    calls = []
    def mock_settings(req):
        calls.append(req)
        return SimpleNamespace(success=lambda: True)

    client = SimpleNamespace(
        cardkit=SimpleNamespace(v1=SimpleNamespace(
            card=SimpleNamespace(settings=mock_settings),
        )),
    )
    ok = await set_card_streaming_mode(client, card_id="ck_1", enabled=False, sequence=10)
    assert ok is True
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_finalize_closes_stream_before_replacing_final_card():
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = object()
    state = CardKitState(
        card_id="ck_1",
        message_id="om_1",
        sequence=7,
        started_at=time.time() - 2.2,
    )
    adapter._streaming_cards["om_1"] = state
    calls = []

    async def close_stream(_client, **kwargs):
        calls.append(("close", kwargs))
        return True

    async def update_final_card(_client, **kwargs):
        calls.append(("update", kwargs))
        return True

    with (
        patch(
            "plugins.platforms.feishu.adapter.set_card_streaming_mode",
            side_effect=close_stream,
        ),
        patch(
            "plugins.platforms.feishu.adapter.cardkit_update_card",
            side_effect=update_final_card,
        ),
    ):
        finalized = await adapter.finalize_streaming_message("om_1", "done")

    assert finalized is True
    assert [name for name, _kwargs in calls] == ["close", "update"]
    assert calls[0][1]["sequence"] == 8
    assert calls[1][1]["sequence"] == 9
    footer = calls[1][1]["card_body"]["body"]["elements"][-1]["content"]
    assert footer.startswith("已完成 · 耗时 2.")
    assert "om_1" not in adapter._streaming_cards


@pytest.mark.asyncio
async def test_finalize_failure_is_reported_and_state_is_retained():
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = object()
    state = CardKitState(
        card_id="ck_1",
        message_id="om_1",
        sequence=3,
        started_at=time.time(),
    )
    adapter._streaming_cards["om_1"] = state

    with (
        patch(
            "plugins.platforms.feishu.adapter.set_card_streaming_mode",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "plugins.platforms.feishu.adapter.cardkit_update_card",
            new=AsyncMock(return_value=False),
        ),
        pytest.raises(RuntimeError, match="final card update failed"),
    ):
        await adapter.finalize_streaming_message("om_1", "done")

    assert adapter._streaming_cards["om_1"] is state


@pytest.mark.asyncio
async def test_stop_all_closes_stream_before_replacing_stopped_card():
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = object()
    state = CardKitState(
        card_id="ck_1",
        message_id="om_1",
        sequence=4,
        started_at=time.time() - 1.0,
        last_content="partial",
    )
    adapter._streaming_cards["om_1"] = state
    calls = []

    async def close_stream(_client, **kwargs):
        calls.append(("close", kwargs))
        return True

    async def update_stopped_card(_client, **kwargs):
        calls.append(("update", kwargs))
        return True

    with (
        patch(
            "plugins.platforms.feishu.adapter.set_card_streaming_mode",
            side_effect=close_stream,
        ),
        patch(
            "plugins.platforms.feishu.adapter.cardkit_update_card",
            side_effect=update_stopped_card,
        ),
    ):
        await adapter.stop_all_streaming_cards()

    assert [name for name, _kwargs in calls] == ["close", "update"]
    assert calls[0][1]["sequence"] == 5
    assert calls[1][1]["sequence"] == 6
    footer = calls[1][1]["card_body"]["body"]["elements"][-1]["content"]
    assert footer.startswith("已停止 · 耗时 1.")
    assert state.stopped is True


def test_cardkit_state_defaults():
    state = CardKitState(card_id="ck_1", message_id="om_1")
    assert state.sequence == 1
    assert state.element_id == STREAMING_ELEMENT_ID
    assert state.failed is False
