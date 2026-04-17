from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

from gateway.platforms.base import MessageType
from gateway.platforms.feishu_inbound.bridge import (
    FeishuExtractedContent,
    build_extracted_content,
    build_message_event,
    build_reply_context,
    extract_text_from_raw_content,
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
