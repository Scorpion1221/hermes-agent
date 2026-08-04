from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import MessageType
from gateway.platforms.feishu_inbound.bridge import (
    FeishuExtractedContent,
    FeishuSenderProfile,
    build_extracted_content,
    build_feishu_inbound_content_bridge,
    build_feishu_message_event,
    build_feishu_sender_profile,
    build_feishu_reply_context_bridge,
    build_message_event,
    build_reply_context,
    coerce_command_message_type,
    extract_text_from_raw_content,
    resolve_feishu_source_chat_type,
    resolve_message_context_type,
    resolve_normalized_message_type,
    resolve_reply_to_message_id,
    should_ignore_extracted_content,
)
from gateway.platforms.feishu_inbound.lookup import build_feishu_message_context
from gateway.platforms.feishu_inbound.parse import normalize_feishu_message
from gateway.platforms.feishu_inbound.types import FeishuMessageContext, FeishuQuotedContext


def test_resolve_reply_to_message_id_prefers_parent_then_root_then_upper():
    assert resolve_reply_to_message_id(parent_id="om_parent", root_id="om_root", upper_message_id="om_upper") == "om_parent"
    assert resolve_reply_to_message_id(parent_id="", root_id="om_root", upper_message_id="om_upper") == "om_root"
    assert resolve_reply_to_message_id(parent_id=None, root_id="", upper_message_id="om_upper") == "om_upper"
    assert resolve_reply_to_message_id(parent_id=None, root_id=None, upper_message_id=None) is None


def test_resolve_feishu_source_chat_type_prefers_chat_info_type_then_event_type():
    assert resolve_feishu_source_chat_type(chat_info={"type": "forum"}, event_chat_type="group") == "forum"
    assert resolve_feishu_source_chat_type(chat_info={"type": "group"}, event_chat_type="p2p") == "group"
    assert resolve_feishu_source_chat_type(chat_info={"type": "dm"}, event_chat_type="p2p") == "dm"
    assert resolve_feishu_source_chat_type(chat_info={"type": ""}, event_chat_type="group") == "group"


@pytest.mark.asyncio
async def test_build_feishu_sender_profile_uses_resolved_display_name_and_preserves_ids():
    sender_id = SimpleNamespace(open_id="ou_user", user_id="u_user", union_id="on_union")
    profile = await build_feishu_sender_profile(
        sender_id=sender_id,
        resolve_display_name=AsyncMock(return_value="Alice"),
    )

    assert profile == FeishuSenderProfile(
        user_id="ou_user",
        user_name="Alice",
        user_id_alt="on_union",
        auth_user_ids=("u_user", "ou_user", "on_union"),
    )


def test_extract_text_from_raw_content_uses_placeholders_and_never_stringifies_null():
    assert (
        extract_text_from_raw_content(
            msg_type="file",
            raw_content='{"file_key":"file_1","file_name":"runbook.pdf"}',
        )
        == "[Attachment: runbook.pdf]"
    )
    assert extract_text_from_raw_content(msg_type="text", raw_content="null") is None
    assert (
        extract_text_from_raw_content(
            msg_type="card",
            raw_content=json.dumps(
                {
                    "type": "template",
                    "data": {
                        "template_id": "ctp_xxx",
                        "template_variable": {
                            "alert_title": "Build Failed",
                            "service": "payments-api",
                            "branch": "main",
                            "log_url": "https://example.com/logs",
                        },
                    },
                }
            ),
        )
        == "Build Failed\npayments-api\nmain"
    )


def test_message_type_resolution_helpers_respect_preferred_type_and_downloaded_media_types():
    normalized = normalize_feishu_message(
        message_type="file",
        raw_content='{"file_key":"file_1","file_name":"voice-note.m4a"}',
    )
    assert resolve_normalized_message_type(normalized, ["audio/m4a"]) == MessageType.AUDIO

    context = FeishuMessageContext(
        message_id="om_ctx",
        content="[Attachment]",
        content_type="file",
        preferred_message_type="document",
    )
    assert resolve_message_context_type(context, ["image/png"]) == MessageType.PHOTO
    assert resolve_message_context_type(context, []) == MessageType.DOCUMENT


def test_build_extracted_content_expands_merge_forward_items_via_context():
    response_items = [
        SimpleNamespace(
            message_id="om_merge",
            msg_type="merge_forward",
            body=SimpleNamespace(content='{"title":"Forwarded"}'),
        ),
        SimpleNamespace(
            message_id="om_text",
            upper_message_id="om_merge",
            msg_type="text",
            body=SimpleNamespace(content='{"text":"Investigating"}'),
            sender=SimpleNamespace(id="ou_alice"),
            create_time="1",
        ),
        SimpleNamespace(
            message_id="om_image",
            upper_message_id="om_merge",
            msg_type="image",
            body=SimpleNamespace(content='{"image_key":"img_nested"}'),
            sender=SimpleNamespace(id="ou_bob"),
            create_time="2",
        ),
    ]
    context = build_feishu_message_context(
        message_id="om_merge",
        message_type="merge_forward",
        raw_content='{"title":"Forwarded"}',
        response_items=response_items,
        resolve_sender_name_sync=lambda sender_id: {"ou_alice": "Alice", "ou_bob": "Bob"}.get(sender_id),
    )

    extracted = build_extracted_content(raw_message_type="merge_forward", context=context)

    assert extracted.text == "- Alice: Investigating\n- Bob: [Image]"
    assert extracted.message_type == MessageType.TEXT
    assert extracted.message_context is context
    assert extracted.message_context.relation_kind == "merge_forward"
    assert extracted.message_context.metadata == {"expanded_from_items": True}
    assert not should_ignore_extracted_content(extracted)


def test_build_extracted_content_clears_binary_placeholders_and_reinjects_document_text():
    context = build_feishu_message_context(
        message_id="om_file",
        message_type="file",
        raw_content='{"file_key":"file_1","file_name":"notes.txt"}',
    )

    extracted = build_extracted_content(
        raw_message_type="file",
        context=context,
        media_urls=["/tmp/notes.txt"],
        media_types=["text/plain"],
        injected_text="[Content of notes.txt]:\nhello world",
    )

    assert extracted.text == "[Content of notes.txt]:\nhello world"
    assert extracted.message_type == MessageType.DOCUMENT

    media_only = build_extracted_content(
        raw_message_type="image",
        context=build_feishu_message_context(
            message_id="om_image",
            message_type="image",
            raw_content='{"image_key":"img_1"}',
        ),
    )
    assert media_only.text == ""
    assert media_only.message_type == MessageType.PHOTO


def test_build_message_event_copies_reply_quote_and_command_fields():
    extracted = FeishuExtractedContent(
        text="/reset session",
        message_type=MessageType.TEXT,
        media_urls=("/tmp/new.png",),
        media_types=("image/png",),
        raw_message_type="text",
    )
    quoted_context = FeishuQuotedContext(
        message_id="om_root",
        kind="merge_forward",
        text="Original text",
        summary="Alice: Original text",
        sender_name="Alice",
        media_urls=("/tmp/original.png",),
        media_types=("image/png",),
        stable_ref="feishu:om_root",
        metadata={"expanded_from_items": True},
    )
    reply_context = build_reply_context(root_id="om_root", upper_message_id="om_upper", quoted_context=quoted_context)
    source = SimpleNamespace(thread_id="thread-1")
    timestamp = datetime(2026, 4, 18, 12, 0, 0)

    event = build_message_event(
        extracted=extracted,
        source=source,
        raw_message={"event": "payload"},
        message_id="om_reply",
        reply_context=reply_context,
        timestamp=timestamp,
    )

    assert event.text == "/reset session"
    assert event.message_type == MessageType.COMMAND
    assert event.source is source
    assert event.raw_message == {"event": "payload"}
    assert event.message_id == "om_reply"
    assert event.media_urls == ["/tmp/new.png"]
    assert event.media_types == ["image/png"]
    assert event.reply_to_message_id == "om_root"
    assert event.reply_to_text == "Alice: Original text"
    assert event.reply_to_media_urls == ["/tmp/original.png"]
    assert event.reply_to_media_types == ["image/png"]
    assert event.quoted_context is quoted_context
    assert event.timestamp == timestamp


def test_build_feishu_reply_context_bridge_propagates_root_reply_and_quoted_media():
    message = SimpleNamespace(
        message_id="om_reply_bridge",
        parent_id=None,
        root_id="om_root",
        upper_message_id="om_upper",
    )
    quoted_context = FeishuQuotedContext(
        message_id="om_root",
        kind="card",
        text="Quoted body",
        summary="",
        sender_name="Alice",
        media_urls=("/tmp/original.png",),
        media_types=("image/png",),
        stable_ref="feishu:om_root",
        metadata={"expanded_from_items": True},
    )

    reply_context = build_feishu_reply_context_bridge(message=message, quoted_context=quoted_context)

    assert reply_context.reply_to_message_id == "om_root"
    assert reply_context.reply_to_text == "Quoted body"
    assert reply_context.reply_to_media_urls == ("/tmp/original.png",)
    assert reply_context.reply_to_media_types == ("image/png",)
    assert reply_context.quoted_context is quoted_context


@pytest.mark.parametrize(
    ("text", "message_type", "expected"),
    [
        ("/reset session", MessageType.TEXT, MessageType.COMMAND),
        ("/reset session", MessageType.DOCUMENT, MessageType.DOCUMENT),
        (" /reset session", MessageType.TEXT, MessageType.TEXT),
    ],
)
def test_coerce_command_message_type_only_promotes_leading_slash_text(text, message_type, expected):
    assert coerce_command_message_type(text=text, message_type=message_type) == expected


@pytest.mark.parametrize(
    (
        "message_type",
        "raw_content",
        "media_urls",
        "media_types",
        "injected_text",
        "expected_text",
        "expected_type",
    ),
    [
        (
            "file",
            '{"file_key":"file_1","file_name":"notes.txt"}',
            ["/tmp/notes.txt"],
            ["text/plain"],
            "[Content of notes.txt]:\nhello world",
            "[Content of notes.txt]:\nhello world",
            MessageType.DOCUMENT,
        ),
        (
            "audio",
            '{"file_key":"audio_1","file_name":"voice.m4a"}',
            ["/tmp/voice.m4a"],
            ["audio/m4a"],
            "Transcript: hello world",
            "",
            MessageType.VOICE,
        ),
        (
            "image",
            '{"image_key":"img_1"}',
            ["/tmp/image.png"],
            ["image/png"],
            "OCR text that should stay detached",
            "",
            MessageType.PHOTO,
        ),
        (
            "file",
            '{"file_key":"file_2","file_name":"bundle.zip"}',
            ["/tmp/part-1.zip", "/tmp/part-2.zip"],
            ["application/zip", "application/zip"],
            "Combined contents",
            "",
            MessageType.DOCUMENT,
        ),
    ],
)
def test_build_feishu_inbound_content_bridge_text_injection_respects_shape_and_type(
    message_type,
    raw_content,
    media_urls,
    media_types,
    injected_text,
    expected_text,
    expected_type,
):
    message = SimpleNamespace(
        message_id=f"om_{message_type}",
        message_type=message_type,
        content=raw_content,
        chat_id="oc_chat",
        chat_type="p2p",
        root_id=None,
        parent_id=None,
        thread_id=None,
    )

    extracted = build_feishu_inbound_content_bridge(
        message=message,
        media_urls=media_urls,
        media_types=media_types,
        injected_text=injected_text,
    )

    assert extracted.text == expected_text
    assert extracted.message_type == expected_type
    assert extracted.media_urls == tuple(media_urls)
    assert extracted.media_types == tuple(media_types)


def test_build_feishu_message_event_propagates_quoted_context_via_bridge_helpers():
    message = SimpleNamespace(
        message_id="om_reply_bridge",
        parent_id=None,
        root_id=None,
        upper_message_id="om_upper",
    )
    quoted_context = FeishuQuotedContext(
        message_id="om_upper",
        kind="plain",
        text="Original text",
        media_urls=("/tmp/original.png",),
        media_types=("image/png",),
    )
    reply_context = build_feishu_reply_context_bridge(message=message, quoted_context=quoted_context)

    event = build_feishu_message_event(
        data={"event": "payload"},
        message=message,
        source=SimpleNamespace(thread_id="thread-2"),
        inbound_content=FeishuExtractedContent(text="Thanks", message_type=MessageType.TEXT),
        reply_context=reply_context,
    )

    assert reply_context.reply_to_message_id == "om_upper"
    assert reply_context.reply_to_text == "Original text"
    assert event.message_type == MessageType.TEXT
    assert event.reply_to_message_id == "om_upper"
    assert event.reply_to_text == "Original text"
    assert event.reply_to_media_urls == ["/tmp/original.png"]
    assert event.reply_to_media_types == ["image/png"]
    assert event.quoted_context is quoted_context
