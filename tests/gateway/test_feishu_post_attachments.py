"""Attachment-zone posts must retain text and downloadable message resources."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from gateway.platforms.feishu_inbound import media_index
from gateway.platforms.feishu_inbound.lookup import build_resource_descriptors
from gateway.platforms.feishu_inbound.parse import normalize_feishu_message
from plugins.platforms.feishu.adapter import (
    FeishuAdapter,
    normalize_feishu_message as legacy_normalize_feishu_message,
)


@pytest.fixture(params=[normalize_feishu_message, legacy_normalize_feishu_message])
def normalize(request):
    return lambda payload: request.param(message_type="post", raw_content=json.dumps(payload))


@pytest.mark.parametrize("shape", ["server", "locale", "wrapped", "wrapped_files"])
def test_post_attachment_zone_preserves_text_and_zip_resource(normalize, shape):
    # The direct server shape was observed via im.message.get after a
    # --markdown + --attachment message; content_v2 is not a second message.
    post = {
        "title": "",
        "content": [[{"tag": "text", "text": "Analyze this archive."}]],
        "content_v2": [[{"tag": "md", "text": "Analyze this archive."}]],
    }
    if shape == "server":
        payload = {**post, "files": [{"file_key": "file_zip", "file_name": "archive.zip", "is_folder": False}]}
    else:
        files = [{"key": "file_zip", "name": "archive.zip", "is_folder": False}]
        payload = {"zh_cn": post, "files": files}
        if shape == "wrapped":
            payload = {"post": {"zh_cn": post}, "files": files}
        elif shape == "wrapped_files":
            payload = {"post": payload}

    result = normalize(payload)
    assert result.text_content == "Analyze this archive.\n[Attachment: archive.zip]"
    assert [(ref.file_key, ref.file_name) for ref in result.media_refs] == [("file_zip", "archive.zip")]
    assert [(ref.type, ref.file_key, ref.file_name) for ref in build_resource_descriptors(result)] == [
        ("file", "file_zip", "archive.zip"),
    ]


def test_post_inline_image_and_attachment_zone_zip_both_survive(normalize):
    result = normalize({
        "content": [[{"tag": "text", "text": "Review "}, {"tag": "img", "image_key": "img_1"}]],
        "files": [{"file_key": "file_zip", "file_name": "archive.zip"}],
    })
    assert result.text_content == "Review [Image]\n[Attachment: archive.zip]"
    assert [(ref.type, ref.file_key) for ref in build_resource_descriptors(result)] == [
        ("image", "img_1"), ("file", "file_zip"),
    ]


@pytest.mark.parametrize("file_name", ["archive.zip", ""])
def test_attachment_only_post_and_unnamed_file_keep_the_resource(normalize, file_name):
    result = normalize({
        "content": [],
        "files": [{"file_key": "file_zip", "file_name": file_name}],
    })
    assert result.text_content == (f"[Attachment: {file_name}]" if file_name else "[Attachment]")
    assert [(ref.file_key, ref.file_name) for ref in build_resource_descriptors(result)] == [
        ("file_zip", file_name),
    ]


def test_post_inline_and_attachment_zone_duplicate_downloads_only_once(normalize):
    result = normalize({
        "content": [[{"tag": "file", "file_key": "file_zip", "file_name": "archive.zip"}]],
        "files": [
            {"key": "file_zip", "name": "archive.zip"},
            {"file_key": "file_zip", "file_name": "archive.zip"},
        ],
    })
    assert result.text_content == "[Attachment: archive.zip]"
    assert len(build_resource_descriptors(result)) == 1


def test_post_folder_attachment_is_explicitly_not_downloaded(normalize):
    result = normalize({
        "content": [[{"tag": "text", "text": "Review this folder."}]],
        "files": [{"file_key": "file_folder", "file_name": "source", "is_folder": True}],
    })
    assert result.text_content.startswith("Review this folder.\n")
    assert "source" in result.text_content
    assert "not downloaded" in result.text_content
    assert build_resource_descriptors(result) == ()


def test_malformed_post_attachment_zone_does_not_create_invalid_resources(normalize):
    result = normalize({
        "content": [[{"tag": "text", "text": "Keep this text."}]],
        "files": [None, "file_bad", {}, {"key": None}, {"file_key": " "}, {"name": "missing-key.zip"}],
    })
    assert result.text_content == "Keep this text."
    assert build_resource_descriptors(result) == ()


@pytest.mark.asyncio
async def test_post_zip_extraction_uses_existing_sdk_download_and_safe_cache(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(media_index, "_INDEX_PATH", home / "cache" / "feishu_media_index.json")
    adapter = FeishuAdapter(PlatformConfig())
    requests = []
    zip_data = io.BytesIO()
    with zipfile.ZipFile(zip_data, "w") as archive:
        archive.writestr("probe.txt", "Verification code: local-attachment-test")
    data = zip_data.getvalue()

    def download(request):
        requests.append(request)
        return SimpleNamespace(
            success=lambda: True, file=io.BytesIO(data), file_name="",
            raw=SimpleNamespace(headers={"Content-Type": "application/zip"}),
        )

    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(
        message_resource=SimpleNamespace(get=download),
    )))
    message = SimpleNamespace(
        message_id="om_post_zip", message_type="post", mentions=[],
        content=json.dumps({
            "title": "",
            "content": [[{"tag": "text", "text": "Analyze this archive."}]],
            "files": [{"file_key": "file_zip", "file_name": "../../archive.zip", "is_folder": False}],
        }),
    )
    try:
        text, kind, paths, types, _mentions = await adapter._extract_message_content(message)
    finally:
        adapter._shutdown_sdk_executor()

    assert text == "Analyze this archive.\n[Attachment: ../../archive.zip]"
    assert kind is MessageType.TEXT
    assert types == ["application/zip"]
    assert len(paths) == len(requests) == 1
    assert requests[0].paths == {"message_id": "om_post_zip", "file_key": "file_zip"}
    assert dict(requests[0].queries)["type"] == "file"
    cached = Path(paths[0])
    assert cached.resolve().is_relative_to((home / "cache" / "documents").resolve())
    assert cached.name.endswith("_archive.zip")
    assert cached.read_bytes() == data
    assert media_index.get_feishu_media_index_entry("om_post_zip", "file_zip").cached_path == str(cached)
    assert not (home / "gateway_state.json").exists()
