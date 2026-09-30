"""Native folder posts traverse real SDK builders and the local attachment cache."""

from __future__ import annotations

import io
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from gateway.platforms.feishu_inbound import media_index
from plugins.platforms.feishu import adapter as feishu


def leaf(key, name, data=b"content", **extra):
    return {
        "file_key": key, "name": name, "is_folder": False,
        "mimetype": "text/plain; charset=utf-8", "size": len(data), **extra,
    }


def directory(key, name, children):
    return {"file_key": key, "name": name, "is_folder": True, "children": children}


class FolderTransport:
    """Only the SDK transport is fake; parsing, requests and caching are real."""

    def __init__(self, adapter, home):
        self.adapter = adapter
        self.home = home
        self.payload = {"code": 0, "data": {"items": []}}
        self.bodies = {}
        self.folder_requests = []
        self.resource_requests = []
        adapter._client = SimpleNamespace(
            request=self.enumerate,
            im=SimpleNamespace(v1=SimpleNamespace(
                message_resource=SimpleNamespace(get=self.download),
            )),
        )

    def enumerate(self, request):
        self.folder_requests.append(request)
        return SimpleNamespace(
            success=lambda: self.payload.get("code") == 0,
            code=self.payload.get("code"), msg=self.payload.get("msg", ""),
            raw=SimpleNamespace(content=json.dumps(self.payload).encode()),
        )

    def download(self, request):
        self.resource_requests.append(request)
        body = self.bodies.get(request.paths["file_key"])
        if body is None:
            return SimpleNamespace(success=lambda: False, code=999, msg="not available")
        return SimpleNamespace(
            success=lambda: True, file=io.BytesIO(body), file_name="",
            raw=SimpleNamespace(headers={"Content-Type": "text/plain; charset=utf-8"}),
        )

    def set_items(self, items):
        self.payload = {"code": 0, "data": {"items": items}}

    async def extract(self, name="source", *, standalone=False):
        content = {"file_key": "file_folder", "file_name": name} if standalone else {
            "title": "",
            "content": [[{"tag": "text", "text": "Read the attached folder."}]],
            "files": [{"file_key": "file_folder", "file_name": name, "is_folder": True}],
        }
        message = SimpleNamespace(
            message_id="om_native_folder", message_type="folder" if standalone else "post", mentions=[],
            content=json.dumps(content),
        )
        text, kind, paths, types, _mentions = await self.adapter._extract_message_content(message)
        if standalone:
            assert kind is MessageType.DOCUMENT
        else:
            assert "Read the attached folder." in text
            assert kind is MessageType.TEXT
        assert paths, "A manifest must explain even empty or failed folders to the agent"
        assert types[0] == "application/json"
        assert len(paths) == len(types)
        for path in paths:
            assert Path(path).resolve().is_relative_to((self.home / "cache").resolve())
        manifest = json.loads(Path(paths[0]).read_text())
        assert isinstance(manifest["files"], list)
        assert isinstance(manifest["directories"], list)
        assert isinstance(manifest["errors"], list)
        return manifest, paths

    @property
    def downloaded_keys(self):
        return [request.paths["file_key"] for request in self.resource_requests]


@pytest.fixture
def folder(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(media_index, "_INDEX_PATH", home / "cache" / "feishu_media_index.json")
    adapter = feishu.FeishuAdapter(PlatformConfig())
    transport = FolderTransport(adapter, home)
    yield transport
    adapter._shutdown_sdk_executor()
    assert not (home / "gateway_state.json").exists()


@pytest.mark.asyncio
async def test_nested_folder_preserves_paths_and_duplicate_basenames(folder):
    folder.bodies = {"file_root": b"root-code", "file_nested": b"nested-code"}
    folder.set_items([
        directory("folder_nested", "nested", [leaf("file_nested", "same.txt", b"nested-code")]),
        leaf("file_root", "same.txt", b"root-code"),
    ])
    manifest, paths = await folder.extract()

    assert manifest["status"] == "complete"
    assert manifest["folder_name"] == "source"
    assert not manifest["errors"]
    assert "source/nested" in manifest["directories"]
    files = {entry["relative_path"]: entry for entry in manifest["files"]}
    assert set(files) == {"source/same.txt", "source/nested/same.txt"}
    assert Path(files["source/same.txt"]["local_path"]).read_bytes() == b"root-code"
    assert Path(files["source/nested/same.txt"]["local_path"]).read_bytes() == b"nested-code"
    assert len(set(paths)) == 3
    assert set(paths[1:]) == {entry["local_path"] for entry in files.values()}
    assert all(entry["content_type"] == "text/plain" for entry in files.values())

    request, = folder.folder_requests
    assert request.http_method == feishu.HttpMethod.GET
    assert request.token_types == {feishu.AccessTokenType.TENANT}
    assert request.uri == "/open-apis/im/v1/files/:file_key/folder"
    assert request.paths == {"file_key": "file_folder"}
    assert dict(request.queries) == {"srctype": "message", "srcid": "om_native_folder", "recursive": "true"}
    for request in folder.resource_requests:
        assert request.paths["message_id"] == "om_native_folder"
        assert dict(request.queries)["type"] == "file"


@pytest.mark.asyncio
async def test_duplicate_file_keys_download_only_once(folder):
    folder.bodies = {"file_a": b"same-content"}
    folder.set_items([
        leaf("file_a", "one.txt", b"same-content"),
        directory("nested", "nested", [leaf("file_a", "two.txt", b"same-content")]),
    ])
    manifest, paths = await folder.extract()
    assert manifest["status"] == "complete"
    assert folder.downloaded_keys == ["file_a"]
    assert len(paths) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("items", [[], [directory("empty", "empty", [])]])
async def test_empty_folder_returns_a_complete_manifest(folder, items):
    folder.set_items(items)
    manifest, paths = await folder.extract()
    assert manifest["status"] == "complete"
    assert not manifest["errors"]
    assert manifest["files"] == []
    assert len(paths) == 1
    assert folder.downloaded_keys == []
    if items:
        assert "source/empty" in manifest["directories"]


@pytest.mark.asyncio
async def test_failed_leaf_retains_good_sibling_and_reports_partial(folder):
    folder.bodies = {"file_good": b"readable"}
    folder.set_items([leaf("file_bad", "bad.txt"), leaf("file_good", "good.txt", b"readable")])
    manifest, paths = await folder.extract()
    assert manifest["status"] == "partial"
    assert manifest["errors"]
    assert len(paths) == 2
    files = {entry["relative_path"]: entry for entry in manifest["files"]}
    assert files["source/bad.txt"]["status"] == "failed"
    assert not files["source/bad.txt"].get("local_path")
    assert Path(files["source/good.txt"]["local_path"]).read_bytes() == b"readable"


@pytest.mark.asyncio
async def test_folder_api_failure_has_explicit_failed_manifest(folder):
    folder.payload = {"code": 999, "msg": "no permission"}
    manifest, paths = await folder.extract()
    assert manifest["status"] == "failed"
    assert manifest["errors"]
    assert manifest["files"] == []
    assert len(paths) == 1
    assert folder.downloaded_keys == []


@pytest.mark.asyncio
@pytest.mark.parametrize("items", [None, "not-a-tree", [None, {}, "invalid"], [directory("nested", "nested", {})]])
async def test_malformed_tree_is_visible_instead_of_silently_complete(folder, items):
    folder.set_items(items)
    manifest, paths = await folder.extract()
    assert manifest["status"] in {"partial", "failed"}
    assert manifest["errors"]
    assert len(paths) == 1
    assert folder.downloaded_keys == []


@pytest.mark.asyncio
async def test_declared_oversize_skips_resource_request(folder, monkeypatch):
    monkeypatch.setattr(feishu, "get_inbound_media_max_bytes", lambda: 10)
    folder.set_items([leaf("huge", "huge.txt", size=11)])
    manifest, paths = await folder.extract()
    assert manifest["status"] != "complete"
    assert manifest["errors"]
    assert folder.downloaded_keys == []
    assert len(paths) == 1


@pytest.mark.asyncio
async def test_actual_bytes_cannot_bypass_declared_size(folder, monkeypatch):
    monkeypatch.setattr(feishu, "get_inbound_media_max_bytes", lambda: 10)
    folder.bodies = {"lying": b"X" * 11}
    folder.set_items([leaf("lying", "lying.txt", size=1)])
    manifest, paths = await folder.extract()
    assert manifest["status"] != "complete"
    assert manifest["errors"]
    assert folder.downloaded_keys == ["lying"]
    assert len(paths) == 1
    assert media_index.get_feishu_media_index_entry("om_native_folder", "lying") is None


@pytest.mark.asyncio
async def test_aggregate_budget_cannot_be_bypassed_by_small_files(folder, monkeypatch):
    monkeypatch.setattr(feishu, "get_inbound_media_max_bytes", lambda: 10)
    folder.bodies = {"first": b"A" * 6, "second": b"B" * 6}
    folder.set_items([leaf("first", "first.txt", size=6), leaf("second", "second.txt", size=6)])
    manifest, paths = await folder.extract()
    assert manifest["status"] == "partial"
    assert manifest["errors"]
    assert folder.downloaded_keys == ["first"]
    assert len(paths) == 2
    assert sum(Path(path).stat().st_size for path in paths[1:]) <= 10


@pytest.mark.asyncio
async def test_cached_file_size_is_rechecked_against_remaining_budget(folder, monkeypatch):
    folder.bodies = {"cached": b"X" * 11}
    folder.set_items([leaf("cached", "cached.txt", size=1)])
    first, _paths = await folder.extract()
    assert first["status"] == "complete"
    monkeypatch.setattr(feishu, "get_inbound_media_max_bytes", lambda: 10)
    manifest, paths = await folder.extract()
    assert manifest["status"] != "complete"
    assert manifest["errors"]
    assert len(paths) == 1
    assert folder.downloaded_keys == ["cached"], "The cached resource must be rejected without another GET"


@pytest.mark.asyncio
async def test_untrusted_names_cannot_escape_cache_or_manifest_root(folder):
    folder.bodies = {"unsafe": b"safe-content"}
    folder.set_items([directory("nested", "../nested", [leaf("unsafe", "../../outside.txt")])])
    manifest, paths = await folder.extract("../../source")
    assert manifest["status"] == "complete"
    for path in paths:
        assert Path(path).parent == folder.home / "cache" / "documents"
    for path in manifest["directories"] + [entry["relative_path"] for entry in manifest["files"]]:
        assert not PurePosixPath(path).is_absolute()
        assert ".." not in PurePosixPath(path).parts
    assert Path(paths[1]).read_bytes() == b"safe-content"


@pytest.mark.asyncio
async def test_file_count_limit_leaves_explicit_partial_result(folder, monkeypatch):
    monkeypatch.setattr(feishu, "_FEISHU_FOLDER_MAX_FILES", 1)
    folder.bodies = {"one": b"1", "two": b"2"}
    folder.set_items([leaf("one", "one.txt", b"1"), leaf("two", "two.txt", b"2")])
    manifest, paths = await folder.extract()
    assert manifest["status"] == "partial"
    assert manifest["errors"]
    assert folder.downloaded_keys == ["one"]
    assert len(paths) == 2


@pytest.mark.asyncio
async def test_tree_entry_limit_also_bounds_directories(folder, monkeypatch):
    monkeypatch.setattr(feishu, "_FEISHU_FOLDER_MAX_ENTRIES", 2)
    folder.set_items([directory(f"dir-{n}", f"dir-{n}", []) for n in range(4)])
    manifest, paths = await folder.extract()
    assert manifest["status"] != "complete"
    assert manifest["errors"]
    assert "source/dir-3" not in manifest["directories"]
    assert len(paths) == 1


@pytest.mark.asyncio
async def test_depth_limit_prevents_deep_leaf_download(folder, monkeypatch):
    monkeypatch.setattr(feishu, "_FEISHU_FOLDER_MAX_DEPTH", 2)
    folder.bodies = {"deep": b"deep"}
    tree = leaf("deep", "deep.txt")
    for n in range(5):
        tree = directory(f"dir-{n}", f"dir-{n}", [tree])
    folder.set_items([tree])
    manifest, paths = await folder.extract()
    assert manifest["status"] != "complete"
    assert manifest["errors"]
    assert folder.downloaded_keys == []
    assert len(paths) == 1


@pytest.mark.asyncio
async def test_standalone_folder_extracts_manifest_and_leaf_as_document(folder):
    folder.bodies = {"file_a": b"native-folder-content"}
    folder.set_items([leaf("file_a", "root.txt", b"native-folder-content")])
    manifest, paths = await folder.extract(standalone=True)
    assert manifest["status"] == "complete"
    assert manifest["files"][0]["relative_path"] == "source/root.txt"
    assert Path(paths[1]).read_bytes() == b"native-folder-content"


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [{"has_more": True}, {"page_token": "next-page"}])
async def test_incomplete_listing_is_not_reported_complete(folder, metadata):
    folder.bodies = {"file_a": b"first-page-content"}
    folder.set_items([leaf("file_a", "first.txt", b"first-page-content")])
    folder.payload["data"].update(metadata)
    manifest, paths = await folder.extract()
    assert manifest["status"] == "partial"
    assert manifest["errors"]
    assert Path(paths[1]).read_bytes() == b"first-page-content"


@pytest.mark.asyncio
async def test_missing_children_are_not_reported_as_empty_success(folder):
    item = directory("nested", "nested", [])
    item["children_count"] = 1
    folder.set_items([item])
    manifest, paths = await folder.extract()
    assert manifest["status"] != "complete"
    assert manifest["errors"]
    assert len(paths) == 1


@pytest.mark.asyncio
async def test_exact_budget_exhaustion_does_not_disable_cap_for_next_file(folder, monkeypatch):
    monkeypatch.setattr(feishu, "get_inbound_media_max_bytes", lambda: 10)
    folder.bodies = {"first": b"A" * 10, "second": b"B"}
    folder.set_items([leaf("first", "first.txt", size=10), leaf("second", "second.txt", size=1)])
    manifest, paths = await folder.extract()
    assert manifest["status"] == "partial"
    assert manifest["errors"]
    assert folder.downloaded_keys == ["first"]
    assert len(paths) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("byte_limit", [0, -1])
async def test_disabled_byte_cap_preserves_folder_downloads(folder, monkeypatch, byte_limit):
    monkeypatch.setattr(feishu, "get_inbound_media_max_bytes", lambda: byte_limit)
    folder.bodies = {"file_a": b"uncapped"}
    folder.set_items([leaf("file_a", "root.txt", b"uncapped")])
    manifest, paths = await folder.extract()
    assert manifest["status"] == "complete"
    assert Path(paths[1]).read_bytes() == b"uncapped"


@pytest.mark.asyncio
async def test_listing_size_limit_rejects_tree_before_downloading(folder):
    folder.bodies = {"file_a": b"content"}
    folder.set_items([leaf("file_a", "huge-name" + "x" * (4 * 1024 * 1024))])
    manifest, paths = await folder.extract()
    assert manifest["status"] == "failed"
    assert manifest["errors"]
    assert folder.downloaded_keys == []
    assert len(paths) == 1


@pytest.mark.asyncio
async def test_zero_byte_file_is_preserved_as_a_real_empty_attachment(folder):
    folder.bodies = {"empty_file": b""}
    folder.set_items([leaf("empty_file", ".gitkeep", b"")])
    manifest, paths = await folder.extract()
    assert manifest["status"] == "complete"
    assert not manifest["errors"]
    assert manifest["files"][0]["status"] == "downloaded"
    assert manifest["files"][0]["relative_path"] == "source/.gitkeep"
    assert Path(paths[1]).is_file()
    assert Path(paths[1]).read_bytes() == b""


@pytest.mark.asyncio
async def test_missing_binary_response_is_not_treated_as_an_empty_file(folder):
    folder.set_items([leaf("missing", "missing.txt", b"")])

    def missing_body(request):
        folder.resource_requests.append(request)
        return SimpleNamespace(success=lambda: True, raw=SimpleNamespace(headers={}))

    folder.adapter._client.im.v1.message_resource.get = missing_body
    manifest, paths = await folder.extract()
    assert manifest["status"] != "complete"
    assert manifest["errors"]
    assert manifest["files"][0]["status"] == "failed"
    assert len(paths) == 1
    assert media_index.get_feishu_media_index_entry("om_native_folder", "missing") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend,agent_home", [("docker", "/root/.hermes"), ("ssh", "~/.hermes")])
async def test_manifest_leaf_paths_match_gateway_agent_visible_cache_paths(folder, monkeypatch, backend, agent_home):
    from tools.credential_files import to_agent_visible_cache_path

    monkeypatch.setenv("TERMINAL_ENV", backend)
    folder.bodies = {"file_a": b"remote-readable-content"}
    folder.set_items([leaf("file_a", "root.txt", b"remote-readable-content")])
    manifest, paths = await folder.extract()

    assert manifest["status"] == "complete"
    host_leaf = paths[1]
    agent_leaf = manifest["files"][0]["local_path"]
    assert Path(host_leaf).read_bytes() == b"remote-readable-content"
    assert agent_leaf == f"{agent_home}/cache/documents/{Path(host_leaf).name}"
    # gateway.run uses this same real mapper for each host media URL when
    # composing document context notes. The manifest must not embed a
    # different host-only path once those notes reach a remote terminal.
    assert agent_leaf == to_agent_visible_cache_path(host_leaf)
    assert agent_leaf != host_leaf
    assert to_agent_visible_cache_path(paths[0]) == f"{agent_home}/cache/documents/{Path(paths[0]).name}"
    assert media_index.get_feishu_media_index_entry("om_native_folder", "file_a").cached_path == host_leaf
