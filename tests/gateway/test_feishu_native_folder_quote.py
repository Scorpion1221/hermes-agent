"""Quoted native folders expose their downloaded manifest to the agent."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.feishu_inbound import media_index
from gateway.platforms.feishu_inbound.lookup import build_feishu_quoted_context
from gateway.platforms.feishu_inbound.render import render_quoted_context_block
from gateway.platforms.feishu_inbound.types import FeishuQuotedContext
from plugins.platforms.feishu import adapter as feishu
from tools.credential_files import to_agent_visible_cache_path


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(media_index, "_INDEX_PATH", home / "cache" / "feishu_media_index.json")
    instance = feishu.FeishuAdapter(PlatformConfig())

    def enumerate_folder(request):
        assert request.paths == {"file_key": "folder_root"}
        assert dict(request.queries)["srcid"] == "om_folder"
        return SimpleNamespace(success=lambda: True, raw=SimpleNamespace(content=json.dumps({
            "code": 0, "data": {"items": [{
                "file_key": "file_child", "name": "child.txt", "is_folder": False,
            }]},
        }).encode()))

    def download(request):
        assert request.paths == {"message_id": "om_folder", "file_key": "file_child"}
        return SimpleNamespace(
            success=lambda: True, file=io.BytesIO(b"folder-child-content"),
            raw=SimpleNamespace(headers={"Content-Type": "text/plain"}),
        )

    instance._client = SimpleNamespace(
        request=enumerate_folder,
        im=SimpleNamespace(v1=SimpleNamespace(message_resource=SimpleNamespace(get=download))),
    )
    yield instance
    instance._shutdown_sdk_executor()
    assert not (home / "gateway_state.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["post", "folder"])
@pytest.mark.parametrize("backend", ["local", "docker", "ssh"])
async def test_quote_downloads_folder_and_renders_readable_manifest(adapter, monkeypatch, kind, backend):
    monkeypatch.setenv("TERMINAL_ENV", backend)
    content = {"file_key": "folder_root", "file_name": "source"}
    if kind == "post":
        content = {
            "content": [[{"tag": "text", "text": "Long quoted description. " * 20}]],
            "files": [{**content, "is_folder": True}],
        }

    async def download_resources(message_id, descriptors):
        return await adapter._download_feishu_resource_descriptors(
            message_id=message_id, descriptors=descriptors,
        )

    context = await build_feishu_quoted_context(
        message_id="om_folder",
        response_items=[{
            "message_id": "om_folder", "msg_type": kind,
            "body": {"content": json.dumps(content)},
        }],
        download_resources=download_resources,
    )

    manifest_path = context.media_urls[0]
    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest["status"] == "complete"
    assert Path(context.media_urls[1]).read_bytes() == b"folder-child-content"
    assert context.metadata["folder_manifest_paths"] == [manifest_path]
    assert manifest["files"][0]["local_path"] == to_agent_visible_cache_path(context.media_urls[1])
    rendered = render_quoted_context_block(context, found_in_history=True)
    visible_path = to_agent_visible_cache_path(manifest_path)
    assert f"folder_manifest: {visible_path}" in rendered
    if kind == "post":
        summary_line = next(line for line in rendered.splitlines() if line.startswith("summary: "))
        assert len(summary_line.removeprefix("summary: ")) == 200
    if backend != "local":
        assert manifest_path not in rendered
    # Ancestor folders also remain readable when the direct reply is plain text.
    descendant = FeishuQuotedContext(message_id="om_reply", kind="plain", text="Next step", parent=context)
    assert f"folder_manifest: {visible_path}" in render_quoted_context_block(descendant)


@pytest.mark.asyncio
async def test_ordinary_file_quote_does_not_gain_folder_metadata():
    async def download_resources(_message_id, _descriptors):
        # A user-uploaded filename alone must not turn a file into a folder.
        return ["/tmp/doc_x_feishu-folder-manifest.json"], ["application/json"]

    context = await build_feishu_quoted_context(
        message_id="om_file",
        response_items=[{
            "message_id": "om_file", "msg_type": "file",
            "body": {"content": '{"file_key":"file_leaf","file_name":"feishu-folder-manifest.json"}'},
        }],
        download_resources=download_resources,
    )
    assert "folder_manifest_paths" not in context.metadata
    assert "folder_manifest:" not in render_quoted_context_block(context)
