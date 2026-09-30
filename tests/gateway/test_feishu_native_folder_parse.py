"""Native IM folder keys must reach the resource downloader as folders."""

from __future__ import annotations

import json

import pytest

from gateway.platforms.base import MessageType
from gateway.platforms.feishu_inbound.bridge import (
    build_extracted_content,
    extract_text_from_raw_content,
    should_ignore_extracted_content,
)
from gateway.platforms.feishu_inbound.lookup import (
    build_feishu_message_context,
    build_resource_descriptors,
)
from gateway.platforms.feishu_inbound.parse import normalize_feishu_message
from gateway.platforms.feishu_inbound.types import FeishuResourceDescriptor
from plugins.platforms.feishu.adapter import normalize_feishu_message as adapter_normalize


@pytest.fixture(params=[normalize_feishu_message, adapter_normalize])
def normalize(request):
    return lambda payload: request.param(message_type="post", raw_content=json.dumps(payload))


@pytest.mark.parametrize("shape", ["server", "locale", "wrapped", "wrapped_files"])
def test_post_native_folder_preserves_resource_kind_through_normalization(normalize, shape):
    # Captured im.message.get shape: native folders live in post.files, not
    # in rich-text rows, and their file_key is a container rather than a file.
    post = {
        "title": "",
        "content": [[{"tag": "text", "text": "Read root.txt and nested/child.txt."}]],
        "content_v2": [[{"tag": "md", "text": "Read root.txt and nested/child.txt."}]],
    }
    files = [{"file_key": "file_folder", "file_name": "source", "is_folder": True}]
    payload = {**post, "files": files}
    if shape != "server":
        files = [{"key": "file_folder", "name": "source", "is_folder": True}]
        payload = {"zh_cn": post, "files": files}
        if shape == "wrapped":
            payload = {"post": {"zh_cn": post}, "files": files}
        elif shape == "wrapped_files":
            payload = {"post": payload}

    normalized = normalize(payload)

    assert normalized.text_content == "Read root.txt and nested/child.txt.\n[Folder attachment: source]"
    assert [(ref.resource_type, ref.file_key, ref.file_name) for ref in normalized.media_refs] == [
        ("folder", "file_folder", "source"),
    ]
    assert build_resource_descriptors(normalized) == (
        FeishuResourceDescriptor(type="folder", file_key="file_folder", file_name="source"),
    )


def test_attachment_only_post_keeps_native_folder_and_deduplicates_key(normalize):
    normalized = normalize({
        "content": [],
        "files": [
            {"file_key": "file_folder", "file_name": "source", "is_folder": True},
            {"key": "file_folder", "name": "source", "is_folder": True},
        ],
    })
    assert normalized.text_content == "[Folder attachment: source]"
    assert build_resource_descriptors(normalized) == (
        FeishuResourceDescriptor(type="folder", file_key="file_folder", file_name="source"),
    )


def test_post_native_folder_can_coexist_with_file_and_image(normalize):
    normalized = normalize({
        "content": [[{"tag": "img", "image_key": "image_1"}]],
        "files": [
            {"file_key": "file_folder", "file_name": "source", "is_folder": True},
            {"file_key": "file_zip", "file_name": "source.zip", "is_folder": False},
        ],
    })
    assert [(ref.type, ref.file_key) for ref in build_resource_descriptors(normalized)] == [
        ("image", "image_1"), ("folder", "file_folder"), ("file", "file_zip"),
    ]


@pytest.mark.parametrize("key", [None, "", " ", 42])
def test_folder_without_valid_key_does_not_create_download_resource(normalize, key):
    normalized = normalize({
        "content": [[{"tag": "text", "text": "Read the folder."}]],
        "files": [{"file_key": key, "file_name": "source", "is_folder": True}],
    })
    assert build_resource_descriptors(normalized) == ()


def test_message_lookup_preserves_native_folder_descriptor():
    context = build_feishu_message_context(
        message_id="om_folder",
        message_type="post",
        raw_content=json.dumps({
            "content": [],
            "files": [{"file_key": "file_folder", "file_name": "source", "is_folder": True}],
        }),
    )
    assert context.content == "[Folder attachment: source]"
    assert context.resource_descriptors == (
        FeishuResourceDescriptor(type="folder", file_key="file_folder", file_name="source"),
    )


@pytest.mark.parametrize("normalize", [normalize_feishu_message, adapter_normalize])
def test_standalone_folder_message_preserves_document_preference_and_resource(normalize):
    # Official larksuite/cli shortcuts/im/convert_lib/folder_test.go uses
    # msg_type=folder with only file_key/file_name (no is_folder attribute).
    normalized = normalize(
        message_type="folder",
        raw_content='{"file_key":"fld_root","file_name":"Docs"}',
    )
    assert normalized.raw_type == normalized.relation_kind == "folder"
    assert normalized.preferred_message_type == "document"
    assert normalized.metadata == {"placeholder_text": "[Folder attachment: Docs]"}
    assert build_resource_descriptors(normalized) == (
        FeishuResourceDescriptor(type="folder", file_key="fld_root", file_name="Docs"),
    )


def test_standalone_folder_bridge_exposes_manifest_and_leaf_as_documents():
    raw_content = '{"file_key":"fld_root","file_name":"Docs"}'
    context = build_feishu_message_context(
        message_id="om_folder", message_type="folder", raw_content=raw_content,
    )
    extracted = build_extracted_content(
        raw_message_type="folder",
        context=context,
        media_urls=["/tmp/hermes-cache/feishu-folder-manifest.json", "/tmp/hermes-cache/root.txt"],
        media_types=["application/json", "text/plain"],
    )
    assert extracted.message_type is MessageType.DOCUMENT
    assert extracted.text == "[Folder attachment: Docs]"
    assert extracted.media_urls == ("/tmp/hermes-cache/feishu-folder-manifest.json", "/tmp/hermes-cache/root.txt")
    assert extracted.media_types == ("application/json", "text/plain")
    assert not should_ignore_extracted_content(extracted)
    assert extract_text_from_raw_content(msg_type="folder", raw_content=raw_content) == extracted.text
