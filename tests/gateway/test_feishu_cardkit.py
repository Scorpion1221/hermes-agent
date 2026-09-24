from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
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
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.feishu.adapter import FeishuAdapter, _build_card_v2_payload


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


def test_cardkit_streaming_uses_native_element_limit():
    adapter = FeishuAdapter(
        PlatformConfig(extra={"streaming_transport": "cardkit"})
    )
    assert adapter.streaming_overflow_limit() == 30_000

    regular_adapter = FeishuAdapter(PlatformConfig())
    assert regular_adapter.streaming_overflow_limit() is None


@pytest.mark.asyncio
async def test_cardkit_reply_over_legacy_message_limit_stays_in_one_card():
    adapter = FeishuAdapter(
        PlatformConfig(extra={"streaming_transport": "cardkit"})
    )
    adapter.send = AsyncMock(
        return_value=SendResult(success=True, message_id="om_1")
    )
    adapter.edit_message = AsyncMock(
        return_value=SendResult(success=True, message_id="om_1")
    )
    adapter.finalize_streaming_message = AsyncMock(return_value=True)
    adapter.on_streaming_message_complete = AsyncMock()

    consumer = GatewayStreamConsumer(
        adapter,
        "oc_chat",
        StreamConsumerConfig(edit_interval=999, buffer_threshold=999),
        metadata={"streaming": True},
    )
    long_report = "# Dogfooding report\n\n" + ("report line\n" * 1_000)
    assert 8_000 < len(long_report) < 30_000

    consumer.on_delta(long_report)
    consumer.finish()
    await consumer.run()

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.kwargs["content"] == long_report
    adapter.finalize_streaming_message.assert_awaited_once_with(
        "om_1", long_report
    )
    adapter.on_streaming_message_complete.assert_awaited_once_with("om_1")


@pytest.mark.asyncio
async def test_successful_cardkit_finalize_suppresses_generic_fallback_after_edit_failure():
    """A final CardKit replace is authoritative even if the last stream tick failed."""
    adapter = FeishuAdapter(
        PlatformConfig(extra={"streaming_transport": "cardkit"})
    )
    adapter.send = AsyncMock(
        return_value=SendResult(success=True, message_id="duplicate")
    )
    adapter.edit_message = AsyncMock(
        return_value=SendResult(success=False, error="CardKit stream failed")
    )
    adapter.finalize_streaming_message = AsyncMock(return_value=True)
    adapter.on_streaming_message_complete = AsyncMock()

    consumer = GatewayStreamConsumer(
        adapter,
        "oc_chat",
        StreamConsumerConfig(edit_interval=999, buffer_threshold=999),
        metadata={"streaming": True},
    )
    # Reproduce a stream whose preview is already visible. The last incremental
    # update fails, but Feishu's final card replacement still succeeds with the
    # complete response.
    consumer._message_id = "om_1"
    consumer._already_sent = True
    consumer._has_visible_delivery = True
    consumer._last_sent_text = "partial"
    final_report = "partial final dogfooding report"
    consumer.on_delta(final_report)
    consumer.finish()

    await consumer.run()

    adapter.finalize_streaming_message.assert_awaited_once_with(
        "om_1", final_report
    )
    adapter.send.assert_not_awaited()
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True
    adapter.on_streaming_message_complete.assert_awaited_once_with("om_1")


# ── Streaming-window expiry and card rollover (adapter side) ───────────────


def _content_client(responses):
    """Fake lark client whose card_element.content replies from ``responses``."""
    calls = []

    def content(req):
        calls.append(req)
        return responses[min(len(calls), len(responses)) - 1]

    client = SimpleNamespace(
        cardkit=SimpleNamespace(v1=SimpleNamespace(card_element=SimpleNamespace(content=content))),
    )
    return client, calls


def _ok():
    return SimpleNamespace(success=lambda: True)


def _fail(code, msg="error"):
    return SimpleNamespace(success=lambda: False, code=code, msg=msg)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [200850, 300309])
async def test_closed_streaming_window_is_reported_without_disabling_the_card(code):
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client, calls = _content_client([_fail(code, "card streaming timeout")])
    state = CardKitState(card_id="ck_1", message_id="om_1", sequence=5, created_at=time.time() - 600)
    adapter._streaming_cards["om_1"] = state

    first = await adapter.edit_message("oc_chat", "om_1", "more text")
    second = await adapter.edit_message("oc_chat", "om_1", "even more text")

    for result in (first, second):
        assert result.success is False
        assert result.raw_response["cardkit_stream_expired"] is True
    assert state.expired is True
    # Not ``failed``: that would reroute later edits to the IM update API.
    assert state.failed is False
    # Known-closed windows are not retried against the API.
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_sequence_conflict_retries_once_with_the_next_number():
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client, calls = _content_client([_fail(300317, "sequence number compare failed"), _ok()])
    state = CardKitState(card_id="ck_1", message_id="om_1", sequence=5)
    adapter._streaming_cards["om_1"] = state

    result = await adapter.edit_message("oc_chat", "om_1", "text")

    assert result.success is True
    assert [c.request_body.sequence for c in calls] == [6, 7]
    assert state.failed is False


@pytest.mark.asyncio
async def test_other_stream_failures_still_mark_the_card_failed():
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client, _calls = _content_client([_fail(99991, "internal error")])
    state = CardKitState(card_id="ck_1", message_id="om_1", sequence=5)
    adapter._streaming_cards["om_1"] = state

    result = await adapter.edit_message("oc_chat", "om_1", "text")

    assert result.success is False
    assert not result.raw_response
    assert state.failed is True


@pytest.mark.asyncio
async def test_rollover_seal_uses_verbatim_footer_and_card_stays_updatable():
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = object()
    state = CardKitState(card_id="ck_1", message_id="om_1", sequence=7, started_at=time.time() - 300)
    adapter._streaming_cards["om_1"] = state
    calls = []

    async def close_stream(_client, **kwargs):
        calls.append(("close", kwargs))
        return True

    async def update_card_body(_client, **kwargs):
        calls.append(("update", kwargs))
        return True

    with (
        patch("plugins.platforms.feishu.adapter.set_card_streaming_mode", side_effect=close_stream),
        patch("plugins.platforms.feishu.adapter.cardkit_update_card", side_effect=update_card_body),
    ):
        sealed = await adapter.finalize_streaming_message(
            "om_1", "part one", footer="⏳ 已运行 5 分钟 · 继续 ↓",
        )
        restatused = await adapter.update_sealed_streaming_message(
            "om_1", footer="⏳ 已运行 8 分钟 · 仍在处理…",
        )
        completed = await adapter.update_sealed_streaming_message("om_1", "part one")

    assert (sealed, restatused, completed) == (True, True, True)
    assert "om_1" not in adapter._streaming_cards
    bodies = [kwargs for name, kwargs in calls if name == "update"]
    assert [b["sequence"] for b in bodies] == [9, 10, 11]
    footers = [b["card_body"]["body"]["elements"][-1]["content"] for b in bodies]
    assert footers[0] == "⏳ 已运行 5 分钟 · 继续 ↓"
    assert footers[1] == "⏳ 已运行 8 分钟 · 仍在处理…"
    assert footers[2].startswith("已完成 · 耗时 5m")
    # The sealed text is reused for footer-only updates.
    assert all(b["card_body"]["body"]["elements"][0]["content"] == "part one" for b in bodies)


@pytest.mark.asyncio
async def test_expired_card_seals_even_if_settings_call_is_rejected():
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = object()
    state = CardKitState(card_id="ck_1", message_id="om_1", sequence=7, expired=True)
    adapter._streaming_cards["om_1"] = state

    with (
        patch("plugins.platforms.feishu.adapter.set_card_streaming_mode", new=AsyncMock(return_value=False)),
        patch("plugins.platforms.feishu.adapter.cardkit_update_card", new=AsyncMock(return_value=True)),
    ):
        assert await adapter.finalize_streaming_message("om_1", "content", footer="继续 ↓") is True


@pytest.mark.asyncio
async def test_continuation_card_reports_the_whole_reply_elapsed_time():
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = object()
    started = time.time() - 700
    state = CardKitState(
        card_id="ck_2", message_id="om_2", sequence=3,
        started_at=time.time() - 30, elapsed_origin=started,
    )
    adapter._streaming_cards["om_2"] = state
    update = AsyncMock(return_value=True)

    with (
        patch("plugins.platforms.feishu.adapter.set_card_streaming_mode", new=AsyncMock(return_value=True)),
        patch("plugins.platforms.feishu.adapter.cardkit_update_card", new=update),
    ):
        await adapter.finalize_streaming_message("om_2", "the end")

    footer = update.await_args.kwargs["card_body"]["body"]["elements"][-1]["content"]
    assert footer.startswith("已完成 · 耗时 11m")


@pytest.mark.asyncio
async def test_rate_limited_frame_is_reported_as_skipped():
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client, _calls = _content_client([_fail(230020, "rate limited")])
    state = CardKitState(card_id="ck_1", message_id="om_1", sequence=5)
    adapter._streaming_cards["om_1"] = state

    result = await adapter.edit_message("oc_chat", "om_1", "text")

    assert result.success is True
    assert result.raw_response == {"cardkit_rate_limited": True}
    assert state.failed is False and state.expired is False
