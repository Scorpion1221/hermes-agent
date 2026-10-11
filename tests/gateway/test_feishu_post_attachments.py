"""The attachment zone of a Feishu rich-text post: the files the composer sends beside the text.

Received posts name an entry ``{"file_key", "file_name", "is_folder"}``; the send API (and payloads
built from it) spell it ``{"key", "name"}``; a locale-wrapped post can carry ``files`` beside the
locales instead of inside one. Every spelling must reach the resource download. All keys, names and
texts are synthetic.
"""

from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace

import pytest

POST = {"title": "", "content": [[{"tag": "text", "text": "Analyze this archive."}]]}


def normalize(payload):
    from plugins.platforms.feishu.adapter import normalize_feishu_message

    return normalize_feishu_message(message_type="post", raw_content=json.dumps(payload))


@pytest.mark.parametrize(
    "payload",
    [
        {**POST, "files": [{"file_key": "file_zip", "file_name": "archive.zip", "is_folder": False}]},
        {**POST, "files": [{"key": "file_zip", "name": "archive.zip"}]},
        {"zh_cn": POST, "files": [{"key": "file_zip", "name": "archive.zip"}]},
        {"en_us": {**POST, "files": [{"key": "file_zip", "name": "archive.zip"}]}},
        {"post": {"zh_cn": POST}, "files": [{"file_key": "file_zip", "file_name": "archive.zip"}]},
        {"post": {"zh_cn": POST, "files": [{"key": "file_zip", "name": "archive.zip"}]}},
    ],
    ids=["received", "send-api-aliases", "outer-files", "inner-files", "wrapped-outer", "wrapped-inner"],
)
def test_every_attachment_zone_spelling_keeps_the_caption_and_the_file(payload):
    normalized = normalize(payload)

    assert normalized.text_content == "Analyze this archive.\n[Attachment: archive.zip]"
    assert [(ref.file_key, ref.file_name, ref.resource_type) for ref in normalized.media_refs] == [
        ("file_zip", "archive.zip", "file"),
    ]


def test_a_file_named_in_the_text_and_in_both_attachment_zones_is_one_resource():
    normalized = normalize({
        "zh_cn": {
            "content": [[{"tag": "file", "file_key": "file_zip", "file_name": "archive.zip"}]],
            "files": [{"file_key": "file_zip", "file_name": "archive.zip"}],
        },
        "files": [{"key": "file_zip", "name": "archive.zip"}, {"key": "file_csv", "name": "rows.csv"}],
    })

    assert normalized.text_content == "[Attachment: archive.zip]\n[Attachment: rows.csv]"
    assert [ref.file_key for ref in normalized.media_refs] == ["file_zip", "file_csv"]


def test_malformed_attachment_zone_entries_leave_the_text_and_no_resource():
    normalized = normalize({
        "zh_cn": POST,
        "files": [None, "file_bad", {}, {"key": None}, {"file_key": " "}, {"key": 42}, {"name": "no-key.zip"}],
    })

    assert normalized.text_content == "Analyze this archive."
    assert normalized.media_refs == []


def test_an_outer_attachment_zone_file_is_downloaded_from_the_message():
    from gateway.config import PlatformConfig
    from plugins.platforms.feishu.adapter import FeishuAdapter

    requests = []

    def get_resource(request):
        requests.append(request)
        return SimpleNamespace(
            success=lambda: True, file=io.BytesIO(b"col_a,col_b\n1,2\n"), file_name=None,
            raw=SimpleNamespace(headers={"Content-Type": "text/csv"}),
        )

    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(
        message_resource=SimpleNamespace(get=get_resource),
    )))
    message = SimpleNamespace(
        message_id="om_post", message_type="post", mentions=None,
        content=json.dumps({"zh_cn": POST, "files": [{"key": "file_csv", "name": "rows.csv"}]}),
    )

    text, _kind, media_urls, media_types, _inlined, _mentions = asyncio.run(adapter._extract_message_content(message))

    assert text == "Analyze this archive.\n[Attachment: rows.csv]"
    assert [(r.message_id, r.file_key, r.type) for r in requests] == [("om_post", "file_csv", "file")]
    assert media_types == ["text/csv"] and media_urls[0].endswith("_rows.csv")
