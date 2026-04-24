"""Tests for the extracted Feishu inbound parse/normalize helpers."""

from __future__ import annotations

import json
from dataclasses import asdict

from gateway.platforms.feishu import normalize_feishu_message as legacy_normalize_feishu_message
from gateway.platforms.feishu_inbound.lookup import build_resource_descriptors
from gateway.platforms.feishu_inbound.parse import normalize_feishu_message


def _assert_matches_legacy(*, message_type: str, payload: dict) -> None:
    raw_content = json.dumps(payload, ensure_ascii=False)
    normalized = normalize_feishu_message(message_type=message_type, raw_content=raw_content)
    legacy = legacy_normalize_feishu_message(message_type=message_type, raw_content=raw_content)
    assert asdict(normalized) == asdict(legacy)


def test_text_normalization_strips_placeholder_mentions_and_whitespace():
    payload = {"text": "Hello @_user_1  \r\n   world"}
    _assert_matches_legacy(message_type="text", payload=payload)

    normalized = normalize_feishu_message(
        message_type="text",
        raw_content=json.dumps(payload, ensure_ascii=False),
    )
    assert normalized.text_content == "Hello\nworld"
    assert normalized.relation_kind == "plain"


def test_post_normalization_collects_mentions_and_resources():
    payload = {
        "zh_cn": {
            "title": "Deploy update",
            "content": [[
                {"tag": "text", "text": "See "},
                {"tag": "at", "user_name": "Alice", "open_id": "ou_alice"},
                {"tag": "text", "text": " "},
                {"tag": "img", "image_key": "img_123", "text": "diagram"},
                {"tag": "text", "text": " "},
                {"tag": "file", "file_key": "file_456", "file_name": "notes.pdf"},
            ]],
        }
    }
    _assert_matches_legacy(message_type="post", payload=payload)

    normalized = normalize_feishu_message(
        message_type="post",
        raw_content=json.dumps(payload, ensure_ascii=False),
    )
    assert normalized.text_content == "Deploy update\nSee @Alice [Image: diagram] [Attachment: notes.pdf]"
    assert normalized.mentioned_ids == ["ou_alice"]

    resources = build_resource_descriptors(normalized)
    assert [(resource.type, resource.file_key, resource.file_name) for resource in resources] == [
        ("image", "img_123", ""),
        ("file", "file_456", "notes.pdf"),
    ]


def test_image_normalization_preserves_alt_text_and_photo_resource():
    payload = {"image_key": "img_789", "alt": "Architecture diagram"}
    _assert_matches_legacy(message_type="image", payload=payload)

    normalized = normalize_feishu_message(
        message_type="image",
        raw_content=json.dumps(payload, ensure_ascii=False),
    )
    assert normalized.text_content == "Architecture diagram"
    assert normalized.preferred_message_type == "photo"
    assert normalized.image_keys == ["img_789"]

    resources = build_resource_descriptors(normalized)
    assert [(resource.type, resource.file_key) for resource in resources] == [("image", "img_789")]


def test_file_normalization_preserves_placeholder_metadata_and_file_resource():
    payload = {"file_key": "file_123", "file_name": "runbook.pdf"}
    _assert_matches_legacy(message_type="file", payload=payload)

    normalized = normalize_feishu_message(
        message_type="file",
        raw_content=json.dumps(payload, ensure_ascii=False),
    )
    assert normalized.text_content == ""
    assert normalized.preferred_message_type == "document"
    assert normalized.metadata == {"placeholder_text": "[Attachment: runbook.pdf]"}

    resources = build_resource_descriptors(normalized)
    assert [(resource.type, resource.file_key, resource.file_name) for resource in resources] == [
        ("file", "file_123", "runbook.pdf"),
    ]


def test_interactive_normalization_preserves_title_body_and_actions():
    payload = {
        "card": {
            "header": {"title": {"tag": "plain_text", "content": "Build Failed"}},
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": "Service: payments-api"}},
                {"tag": "div", "text": {"tag": "plain_text", "content": "Branch: main"}},
                {
                    "tag": "action",
                    "actions": [
                        {"tag": "button", "text": {"tag": "plain_text", "content": "View Logs"}},
                        {"tag": "button", "text": {"tag": "plain_text", "content": "Retry"}},
                    ],
                },
            ],
        }
    }
    _assert_matches_legacy(message_type="interactive", payload=payload)

    normalized = normalize_feishu_message(
        message_type="interactive",
        raw_content=json.dumps(payload, ensure_ascii=False),
    )
    assert normalized.relation_kind == "interactive"
    assert normalized.text_content == (
        "Build Failed\n"
        "Service: payments-api\n"
        "Branch: main\n"
        "View Logs\n"
        "Retry\n"
        "Actions: View Logs, Retry"
    )


def test_template_variable_card_normalization_filters_urls_and_ids():
    payload = {
        "type": "template",
        "data": {
            "template_id": "ctp_xxx",
            "template_variable": {
                "alert_title": "Build Failed",
                "service": "payments-api",
                "branch": "main",
                "build_id": "build_123",
                "log_url": "https://example.com/logs",
                "details": {"service_copy": "payments-api"},
            },
        },
    }
    _assert_matches_legacy(message_type="card", payload=payload)

    normalized = normalize_feishu_message(
        message_type="card",
        raw_content=json.dumps(payload, ensure_ascii=False),
    )
    assert normalized.relation_kind == "interactive"
    assert normalized.text_content == "Build Failed\npayments-api\nmain"


def test_share_chat_normalization_exposes_summary_and_metadata():
    payload = {"chat_id": "oc_chat_shared", "chat_name": "Backend Guild"}
    _assert_matches_legacy(message_type="share_chat", payload=payload)

    normalized = normalize_feishu_message(
        message_type="share_chat",
        raw_content=json.dumps(payload, ensure_ascii=False),
    )
    assert normalized.relation_kind == "share_chat"
    assert normalized.text_content == "Shared chat: Backend Guild\nChat ID: oc_chat_shared"
    assert normalized.metadata == {"chat_id": "oc_chat_shared", "chat_name": "Backend Guild"}


def test_merge_forward_normalization_preserves_summary_lines():
    payload = {
        "title": "Sprint recap",
        "messages": [
            {"sender_name": "Alice", "text": "Please review PR-128"},
            {
                "sender_name": "Bob",
                "message_type": "post",
                "content": {
                    "en_us": {
                        "content": [[{"tag": "text", "text": "Ship it"}]],
                    }
                },
            },
        ],
    }
    _assert_matches_legacy(message_type="merge_forward", payload=payload)

    normalized = normalize_feishu_message(
        message_type="merge_forward",
        raw_content=json.dumps(payload, ensure_ascii=False),
    )
    assert normalized.relation_kind == "merge_forward"
    assert normalized.text_content == "Sprint recap\n- Alice: Please review PR-128\n- Bob: Ship it"
    assert normalized.metadata == {"entry_count": 2, "title": "Sprint recap"}
