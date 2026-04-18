import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.feishu_inbound.lookup import (
    build_feishu_message_context,
    build_feishu_quoted_context,
    build_resource_descriptors,
    extract_message_items,
)
from gateway.platforms.feishu_inbound.render import render_quoted_context_block
from gateway.platforms.feishu_inbound.types import FeishuQuotedContext, FeishuResourceDescriptor


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (SimpleNamespace(data=SimpleNamespace(items=["first", "second"])), ["first", "second"]),
        (SimpleNamespace(data=SimpleNamespace(items=("tuple-item",))), ["tuple-item"]),
        (SimpleNamespace(data=SimpleNamespace(items=None)), []),
        (SimpleNamespace(data=None), []),
        (SimpleNamespace(), []),
    ],
)
def test_extract_message_items_handles_lists_iterables_and_missing_data(response, expected):
    assert extract_message_items(response) == expected


def test_build_resource_descriptors_collects_images_and_maps_unknown_media_types():
    normalized = SimpleNamespace(
        image_keys=["img_primary", "img_secondary"],
        media_refs=[
            SimpleNamespace(file_key="file_pdf", file_name="report.pdf", resource_type="file"),
            SimpleNamespace(file_key="video_clip", file_name="demo.mp4", resource_type="video"),
            SimpleNamespace(file_key="mystery_blob", file_name="payload.bin", resource_type="custom"),
            SimpleNamespace(file_key="", file_name="ignored.txt", resource_type="file"),
        ],
    )

    assert build_resource_descriptors(normalized) == (
        FeishuResourceDescriptor(type="image", file_key="img_primary"),
        FeishuResourceDescriptor(type="image", file_key="img_secondary"),
        FeishuResourceDescriptor(type="file", file_key="file_pdf", file_name="report.pdf"),
        FeishuResourceDescriptor(type="video", file_key="video_clip", file_name="demo.mp4"),
        FeishuResourceDescriptor(type="file", file_key="mystery_blob", file_name="payload.bin"),
    )


def test_build_feishu_message_context_prefers_placeholder_text_then_type_fallback():
    file_context = build_feishu_message_context(
        message_id="om_file",
        message_type="file",
        raw_content='{"file_key":"file_1","file_name":"runbook.pdf"}',
    )
    image_context = build_feishu_message_context(
        message_id="om_image",
        message_type="image",
        raw_content='{"image_key":"img_1"}',
    )

    assert file_context.content == "[Attachment: runbook.pdf]"
    assert file_context.preferred_message_type == "document"
    assert file_context.relation_kind == "file"
    assert file_context.resource_descriptors == (
        FeishuResourceDescriptor(type="file", file_key="file_1", file_name="runbook.pdf"),
    )

    assert image_context.content == "[Image]"
    assert image_context.preferred_message_type == "photo"
    assert image_context.relation_kind == "image"
    assert image_context.resource_descriptors == (
        FeishuResourceDescriptor(type="image", file_key="img_1"),
    )


def test_build_feishu_message_context_expands_merge_forward_items_and_preserves_resources():
    response_items = [
        SimpleNamespace(
            message_id="om_merge",
            msg_type="merge_forward",
            body=SimpleNamespace(content='{"title":"Forwarded"}'),
        ),
        SimpleNamespace(
            message_id="om_post",
            upper_message_id="om_merge",
            msg_type="post",
            body=SimpleNamespace(content='{"en_us":{"content":[[{"tag":"text","text":"ETA 10 min"}]]}}'),
            sender=SimpleNamespace(id="ou_bob"),
            create_time="1",
        ),
        SimpleNamespace(
            message_id="om_text",
            upper_message_id="om_merge",
            msg_type="text",
            body=SimpleNamespace(content='{"text":"Investigating"}'),
            sender=SimpleNamespace(id="ou_alice"),
            create_time="2",
        ),
        SimpleNamespace(
            message_id="om_image",
            upper_message_id="om_merge",
            msg_type="image",
            body=SimpleNamespace(content='{"image_key":"img_nested"}'),
            sender=SimpleNamespace(id="ou_bob"),
            create_time="3",
        ),
    ]

    context = build_feishu_message_context(
        message_id="om_merge",
        message_type="merge_forward",
        raw_content='{"title":"Forwarded"}',
        response_items=response_items,
        resolve_sender_name_sync=lambda sender_id: {"ou_alice": "Alice", "ou_bob": "Bob"}.get(sender_id),
    )

    assert context.content == "- Bob: ETA 10 min\n- Alice: Investigating\n- Bob: [Image]"
    assert context.content_type == "merge_forward"
    assert context.preferred_message_type == "text"
    assert context.relation_kind == "merge_forward"
    assert context.resource_descriptors == (
        FeishuResourceDescriptor(type="image", file_key="img_nested"),
    )
    assert context.metadata == {"expanded_from_items": True}


def test_build_feishu_message_context_prefers_embedded_sender_name_over_sender_id_when_lookup_missing():
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
            sender_name="Alice Embedded",
            create_time="1",
        ),
    ]

    context = build_feishu_message_context(
        message_id="om_merge",
        message_type="merge_forward",
        raw_content='{"title":"Forwarded"}',
        response_items=response_items,
        resolve_sender_name_sync=lambda _sender_id: None,
    )

    assert context.content == "- Alice Embedded: Investigating"


def test_build_feishu_message_context_hides_opaque_sender_ids_when_merge_forward_lookup_missing():
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
            sender=SimpleNamespace(id="ou_hidden_sender"),
            create_time="1",
        ),
    ]

    context = build_feishu_message_context(
        message_id="om_merge",
        message_type="merge_forward",
        raw_content='{"title":"Forwarded"}',
        response_items=response_items,
        resolve_sender_name_sync=lambda _sender_id: None,
    )

    assert context.content == "- Investigating"
    assert "ou_hidden_sender" not in context.content


def test_build_feishu_quoted_context_downloads_expanded_merge_forward_resources():
    response_items = [
        SimpleNamespace(
            message_id="om_merge",
            msg_type="merge_forward",
            body=SimpleNamespace(content='{"title":"Forwarded"}'),
            sender=SimpleNamespace(id="ou_forwarder"),
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
    download_resources = AsyncMock(return_value=(["/tmp/merge-forward.png"], ["image/png"]))
    resolve_sender_name = AsyncMock(return_value="Forward Bot")

    quoted = asyncio.run(
        build_feishu_quoted_context(
            message_id="om_merge",
            response_items=response_items,
            download_resources=download_resources,
            resolve_sender_name=resolve_sender_name,
            resolve_sender_name_sync=lambda sender_id: {"ou_alice": "Alice", "ou_bob": "Bob"}.get(sender_id),
        )
    )

    assert quoted == FeishuQuotedContext(
        message_id="om_merge",
        kind="merge_forward",
        text="- Alice: Investigating\n- Bob: [Image]",
        summary="Forward Bot: - Alice: Investigating\n- Bob: [Image]",
        sender_name="Forward Bot",
        media_urls=("/tmp/merge-forward.png",),
        media_types=("image/png",),
        stable_ref="feishu:om_merge",
        metadata={"expanded_from_items": True},
    )
    download_resources.assert_awaited_once_with(
        "om_merge",
        (FeishuResourceDescriptor(type="image", file_key="img_nested"),),
    )


def test_build_feishu_quoted_context_uses_embedded_sender_name_when_lookup_missing():
    response_items = [
        SimpleNamespace(
            message_id="om_plain",
            msg_type="text",
            body=SimpleNamespace(content='{"text":"Hello"}'),
            sender=SimpleNamespace(id="ou_alice"),
            sender_name="Alice Embedded",
        ),
    ]

    quoted = asyncio.run(
        build_feishu_quoted_context(
            message_id="om_plain",
            response_items=response_items,
            download_resources=AsyncMock(return_value=([], [])),
            resolve_sender_name=AsyncMock(return_value=None),
            resolve_sender_name_sync=lambda _sender_id: None,
        )
    )

    assert quoted.sender_name == "Alice Embedded"
    assert quoted.summary == "Alice Embedded: Hello"


def test_build_feishu_quoted_context_hides_opaque_sender_ids_in_merge_forward_summary_and_render():
    response_items = [
        SimpleNamespace(
            message_id="om_merge",
            msg_type="merge_forward",
            body=SimpleNamespace(content='{"title":"Forwarded"}'),
            sender=SimpleNamespace(id="ou_forwarder"),
        ),
        SimpleNamespace(
            message_id="om_text",
            upper_message_id="om_merge",
            msg_type="text",
            body=SimpleNamespace(content='{"text":"Investigating"}'),
            sender=SimpleNamespace(id="ou_hidden_sender"),
            create_time="1",
        ),
    ]

    quoted = asyncio.run(
        build_feishu_quoted_context(
            message_id="om_merge",
            response_items=response_items,
            download_resources=AsyncMock(return_value=([], [])),
            resolve_sender_name=AsyncMock(return_value=None),
            resolve_sender_name_sync=lambda _sender_id: None,
        )
    )
    rendered = render_quoted_context_block(quoted)

    assert quoted.sender_name == ""
    assert quoted.summary == "- Investigating"
    assert "ou_forwarder" not in quoted.summary
    assert "ou_hidden_sender" not in quoted.summary
    assert "sender:" not in rendered
    assert "summary: - Investigating" in rendered
    assert "ou_forwarder" not in rendered
    assert "ou_hidden_sender" not in rendered


def test_render_quoted_context_block_includes_history_truncation_image_analysis_and_media_count():
    quoted = FeishuQuotedContext(
        message_id="om_quote",
        kind="merge_forward",
        summary="x" * 205,
        sender_name="Alice",
        media_urls=("/tmp/one.png", "/tmp/two.png"),
    )

    rendered = render_quoted_context_block(
        quoted,
        found_in_history=True,
        image_analysis="  detected screenshot with annotations  ",
    )

    assert rendered == "\n".join(
        [
            "[Quoted message context]",
            "type: merge_forward",
            "message_id: om_quote",
            "sender: Alice",
            f"summary: {'x' * 200}",
            "image_analysis:",
            "detected screenshot with annotations",
            "media_count: 2",
            "[/Quoted message context]",
        ]
    )
