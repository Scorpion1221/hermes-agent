from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.feishu import _build_card_v2_payload
from gateway.platforms.feishu_inbound.cardkit import (
    STREAMING_ELEMENT_ID,
    CardKitState,
    build_card_id_message_content,
    build_final_card_body,
    build_streaming_card_body,
    create_streaming_card,
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
    body = build_final_card_body("# H1\n## H2\n### H3")
    assert body["body"]["elements"][0]["content"] == "### H1\n#### H2\n##### H3"


def test_build_card_v2_payload_downshifts_markdown_headings_before_send():
    payload = json.loads(_build_card_v2_payload("# H1\n## H2\nplain"))
    assert payload["body"]["elements"][0]["content"] == "### H1\n#### H2\nplain"


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
    mock_resp = SimpleNamespace(success=lambda: True)
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


def test_cardkit_state_defaults():
    state = CardKitState(card_id="ck_1", message_id="om_1")
    assert state.sequence == 1
    assert state.element_id == STREAMING_ELEMENT_ID
    assert state.failed is False
