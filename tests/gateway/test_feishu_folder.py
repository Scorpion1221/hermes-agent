"""Feishu folders as the agent sees them: a native folder message, a folder in a post, a quoted folder.

Feishu lists a folder with ``GET im/v1/files/{folder_key}/folder?srctype=message&srcid={message_id}&recursive=true``
(a tree of ``{file_key, name, is_folder, mimetype, size, children}``) and serves each file through the
carrying message's resource endpoint. The client is faked at that boundary (the raw tenant GET and
``im.v1.message_resource.get``); caching, the size cap and path translation are the real ones. All
keys, names and contents are synthetic.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from urllib.parse import unquote

import pytest

from tests.gateway.test_feishu_fetched_content import (  # noqa: F401  (autouse SDK fakes)
    FakeFeishu,
    _fake_lark_requests,
    delivered_turns,
    dispatched_event,
    inbound_event,
    inbound_message,
    item,
    make_adapter,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def leaf(key, name, data=b"content", size=None, **extra):
    return {"file_key": key, "name": name, "is_folder": False, "mimetype": "text/plain", "size": len(data) if size is None else size, **extra}


def folder_node(key, name, children):
    return {"file_key": key, "name": name, "is_folder": True, "size": 0, "children": children}


class FolderFeishu(FakeFeishu):
    """``trees``: folder key -> listed items (or a whole response body via ``bodies``); ``files``: key -> bytes."""

    def __init__(self, trees=None, files=None, *, bodies=None, types=None, messages=None):
        super().__init__(messages or {})
        self.trees, self.files, self.bodies, self.types = trees or {}, files or {}, bodies or {}, types or {}
        self.listings = []

    def request(self, request):  # raw tenant GET: the folder listing
        self.listings.append(request)
        folder_key = unquote(PurePosixPath(request.uri).parent.name)  # Feishu decodes the path segment
        body = self.bodies.get(folder_key, {"code": 0, "data": {"all_count": 0, "items": self.trees.get(folder_key, [])}})
        return SimpleNamespace(raw=SimpleNamespace(content=body if isinstance(body, bytes) else json.dumps(body).encode()))

    def _get_resource(self, request):
        self.resource_gets.append(request)
        data = self.files.get(request.file_key)
        if data is None:
            return SimpleNamespace(success=lambda: False, code=234003, msg="File not in msg.")
        content_type = self.types.get(request.file_key, "text/plain; charset=utf-8")
        return SimpleNamespace(
            success=lambda: True, file=None if data is ... else io.BytesIO(data), file_name=None,
            raw=SimpleNamespace(headers={"Content-Type": content_type}),
        )

    @property
    def downloaded(self):
        return [request.file_key for request in self.resource_gets]


def folder_message(message_id="om_folder", key="fld_root", name="project"):
    return inbound_message(message_id, "folder", json.dumps({"file_key": key, "file_name": name}))


def send(client, message):
    return dispatched_event(make_adapter(client), message)


def listing_of(event):
    # listing is always the first media_url; its path also appears as the "Full listing" line
    listing_path = event.media_urls[0]
    return json.loads(Path(listing_path).read_text(encoding="utf-8"))


def files_urls(event_or_content):
    """media_urls without the leading listing JSON (first entry per folder)."""
    urls = event_or_content.media_urls
    return [u for u in urls if not u.endswith(".folder.json")]


def tree_path(event, name):
    """The cached path the tree names for file ``name``."""
    line = next(line for line in event.text.splitlines() if line.strip().startswith(f"- {name} → "))
    return line.split(" → ", 1)[1]


# --- native folder message ---

def test_a_folder_message_lists_its_tree_and_attaches_its_files_from_the_carrying_message():
    from gateway.platforms.base import get_document_cache_dir

    client = FolderFeishu(
        trees={"fld_root": [
            folder_node("fld_docs", "docs", [leaf("fil_guide", "guide.md", b"# guide"), leaf("fil_notes_a", "notes.txt", b"docs notes")]),
            leaf("fil_notes_b", "notes.txt", b"top notes"),
            folder_node("fld_empty", "empty", []),
        ]},
        files={"fil_guide": b"# guide", "fil_notes_a": b"docs notes", "fil_notes_b": b"top notes"},
    )

    event = send(client, folder_message())

    request, = client.listings
    assert request.uri == "/open-apis/im/v1/files/fld_root/folder"
    assert dict(request.queries) == {"srctype": "message", "srcid": "om_folder", "recursive": "true"}
    assert [(r.message_id, r.type) for r in client.resource_gets] == [("om_folder", "file")] * 3
    assert event.message_type.value == "text"  # so the note Feishu sends behind it joins its turn
    file_urls = files_urls(event)
    assert event.text.splitlines()[:6] == [
        'Contents of folder "project" (3 files, 2 subfolders):',
        "- docs/",
        f"  - guide.md → {file_urls[0]}",
        f"  - notes.txt → {file_urls[1]}",
        f"- notes.txt → {file_urls[2]}",
        "- empty/",
    ]
    assert event.text.splitlines()[-1].startswith("Full listing (JSON): ")
    assert [Path(path).read_bytes() for path in file_urls] == [b"# guide", b"docs notes", b"top notes"]
    assert all(Path(path).parent == get_document_cache_dir() for path in event.media_urls)
    assert event.media_types == ["text/plain", "text/markdown", "text/plain", "text/plain"]
    assert event.media_text_inlined == [False, False, False, False]
    listing = listing_of(event)
    assert listing["folder"] == "project" and listing["notices"] == []
    assert [(e["path"], e["type"], e.get("status")) for e in listing["entries"]] == [
        ("docs", "folder", None), ("docs/guide.md", "file", "downloaded"), ("docs/notes.txt", "file", "downloaded"),
        ("notes.txt", "file", "downloaded"), ("empty", "folder", None),
    ]
    listing_dir = Path(event.media_urls[0]).parent  # local_path is relative to the listing's directory
    assert [str(listing_dir / e["local_path"]) for e in listing["entries"] if e["type"] == "file"] == file_urls


def test_an_empty_folder_says_so_and_downloads_nothing():
    client = FolderFeishu(trees={"fld_root": []})

    event = send(client, folder_message())

    assert event.text == 'Folder "project" is empty.'
    assert event.media_urls == [] and client.resource_gets == []


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ({"code": 234040, "msg": "The message is invisible to the operator."}, "Feishu error 234040"),
        ({"code": 0, "data": {"items": "not-a-tree"}}, "Feishu returned an unreadable listing"),
        (b"<html>gateway timeout</html>", "Feishu returned an unreadable listing"),
    ],
    ids=["api-error", "malformed-items", "not-json"],
)
def test_a_folder_that_cannot_be_listed_says_why_instead_of_vanishing(body, reason):
    client = FolderFeishu(bodies={"fld_root": body})

    event = send(client, folder_message())

    assert event.text == f'Folder "project" could not be read ({reason}); none of its files were downloaded.'
    assert event.media_urls == [] and client.resource_gets == []


def test_an_oversized_listing_is_refused_before_it_is_parsed(monkeypatch):
    import plugins.platforms.feishu.feishu_folder as feishu_folder

    monkeypatch.setattr(feishu_folder, "_MAX_LISTING_BYTES", 64)
    client = FolderFeishu(trees={"fld_root": [leaf("fil_a", "a.txt"), leaf("fil_b", "b.txt")]}, files={"fil_a": b"a"})

    event = send(client, folder_message())

    assert "could not be read (its listing is too large)" in event.text
    assert client.resource_gets == []


# --- failures and bounds stay visible ---

def test_a_failed_file_is_named_and_its_siblings_still_arrive():
    client = FolderFeishu(
        trees={"fld_root": [leaf("fil_gone", "gone.txt"), leaf("fil_nobody", "nobody.txt"), leaf("fil_ok", "ok.txt", b"ok"),
                            leaf("fil_empty", ".gitkeep", b"")]},
        files={"fil_ok": b"ok", "fil_nobody": ..., "fil_empty": b""},
    )

    event = send(client, folder_message())

    assert "- gone.txt (download failed)" in event.text
    assert "- nobody.txt (download failed)" in event.text  # a success without a body is not an empty file
    assert Path(tree_path(event, "ok.txt")).read_bytes() == b"ok"
    assert Path(tree_path(event, ".gitkeep")).read_bytes() == b""  # an empty file is a real file
    assert len(files_urls(event)) == 2
    statuses = {e["path"]: e["status"] for e in listing_of(event)["entries"]}
    assert statuses == {"gone.txt": "download failed", "nobody.txt": "download failed", "ok.txt": "downloaded", ".gitkeep": "downloaded"}


def test_the_size_cap_skips_files_listed_too_large_and_catches_listed_sizes_that_lie(monkeypatch):
    monkeypatch.setattr("gateway.platforms.base.get_inbound_media_max_bytes", lambda: 10)
    client = FolderFeishu(
        trees={"fld_root": [leaf("fil_big", "big.bin", size=11), leaf("fil_liar", "liar.txt", size=1), leaf("fil_fits", "fits.txt", b"12345")]},
        files={"fil_big": b"x" * 11, "fil_liar": b"y" * 11, "fil_fits": b"12345"},
    )

    event = send(client, folder_message())

    assert client.downloaded.count("fil_big") == 0  # declared too large: never requested
    assert "- big.bin (not downloaded: over the inbound media size limit for the folder)" in event.text
    assert "- liar.txt (not downloaded: over the inbound media size limit for the folder)" in event.text
    assert [Path(path).read_bytes() for path in files_urls(event)] == [b"12345"]


def test_the_size_cap_is_shared_by_the_files_of_every_folder_in_a_message(monkeypatch):
    monkeypatch.setattr("gateway.platforms.base.get_inbound_media_max_bytes", lambda: 10)
    post = {"zh_cn": {"content": [[{"tag": "text", "text": "both"}]]}, "files": [
        {"key": "fld_one", "name": "one", "is_folder": True}, {"key": "fld_two", "name": "two", "is_folder": True},
    ]}
    client = FolderFeishu(
        trees={"fld_one": [leaf("fil_a", "a.txt", b"a" * 6)], "fld_two": [leaf("fil_b", "b.txt", b"b" * 6)]},
        files={"fil_a": b"a" * 6, "fil_b": b"b" * 6},
    )

    event = send(client, inbound_message("om_post", "post", json.dumps(post)))

    assert sum(Path(path).stat().st_size for path in files_urls(event)) <= 10
    assert len(files_urls(event)) == 1
    assert "- b.txt (not downloaded: over the inbound media size limit for the folder)" in event.text


@pytest.mark.parametrize("cap", [0, -1])
def test_a_disabled_size_cap_downloads_everything(monkeypatch, cap):
    monkeypatch.setattr("gateway.platforms.base.get_inbound_media_max_bytes", lambda: cap)
    client = FolderFeishu(trees={"fld_root": [leaf("fil_a", "a.txt", b"a" * 50)]}, files={"fil_a": b"a" * 50})

    event = send(client, folder_message())

    assert [Path(path).read_bytes() for path in files_urls(event)] == [b"a" * 50]


def test_entry_depth_and_file_bounds_say_what_they_left_out(monkeypatch):
    import plugins.platforms.feishu.feishu_folder as feishu_folder

    monkeypatch.setattr(feishu_folder, "MAX_ENTRIES", 6)
    monkeypatch.setattr(feishu_folder, "MAX_DEPTH", 2)
    monkeypatch.setattr(feishu_folder, "MAX_FILES", 2)
    deep = folder_node("fld_l1", "l1", [folder_node("fld_l2", "l2", [leaf("fil_deep", "deep.txt")])])
    flat = [leaf(f"fil_{n}", f"{n}.txt", str(n).encode()) for n in range(6)]
    client = FolderFeishu(trees={"fld_root": [deep, *flat]}, files={f"fil_{n}": str(n).encode() for n in range(6)} | {"fil_deep": b"d"})

    event = send(client, folder_message())

    listed = [e["path"] for e in listing_of(event)["entries"]]
    assert listed == ["l1", "l1/l2", "0.txt", "1.txt", "2.txt", "3.txt"]  # deep.txt is past the depth bound
    assert len(client.downloaded) == 2
    assert "- 2.txt (not downloaded: file limit reached)" in event.text
    assert "The folder has more than 6 entries; the rest are not listed." in event.text
    assert "Entries nested more than 2 levels deep are not listed." in event.text


def test_attachments_are_capped_and_audio_is_listed_but_never_attached(monkeypatch):
    import plugins.platforms.feishu.feishu_folder as feishu_folder

    monkeypatch.setattr(feishu_folder, "MAX_MEDIA", 2)
    client = FolderFeishu(
        trees={"fld_root": [leaf("fil_voice", "memo.ogg", b"OggS"), *(leaf(f"fil_{n}", f"{n}.txt", b"t") for n in range(3))]},
        files={"fil_voice": b"OggS", **{f"fil_{n}": b"t" for n in range(3)}},
        types={"fil_voice": "audio/ogg"},
    )

    event = send(client, folder_message())

    assert len(client.resource_gets) == 4  # every file is downloaded and named in the tree
    assert os.path.isfile(tree_path(event, "memo.ogg"))
    assert "audio/ogg" not in event.media_types  # voice is cached but never attached
    assert len(files_urls(event)) == 1  # listing takes 1 of the 2 MAX_MEDIA slots; 1 text file takes the other


def test_a_file_named_as_an_image_that_is_not_one_is_still_kept_as_a_file():
    client = FolderFeishu(trees={"fld_root": [leaf("fil_fake", "photo.png", b"not an image")]},
                          files={"fil_fake": b"not an image"}, types={"fil_fake": "image/png"})

    event = send(client, folder_message())

    assert Path(tree_path(event, "photo.png")).read_bytes() == b"not an image"
    assert event.media_types[1:] == ["application/octet-stream"]


def test_malformed_entries_are_reported_and_the_readable_ones_still_arrive():
    incomplete = folder_node("fld_partial", "partial", [])
    incomplete["children_count"] = 3
    client = FolderFeishu(
        trees={"fld_root": [None, {"name": "no-key.txt"}, leaf("fil_a", "a.txt", b"a"), incomplete,
                            folder_node("fld_bad", "bad", {"not": "a list"})]},
        files={"fil_a": b"a"},
    )

    event = send(client, folder_message())

    assert client.downloaded == ["fil_a"]
    listing = listing_of(event)
    assert [e["path"] for e in listing["entries"]] == ["a.txt", "partial", "bad"]
    assert "Some entries of the folder could not be read and are not listed." in event.text


def test_a_key_listed_twice_is_listed_and_downloaded_once_even_a_folder_listed_inside_itself():
    def looping(levels):  # a folder listed again inside itself, as a listing that loops would show it
        children = [leaf("fil_b", "b.txt", b"b")]
        return folder_node("fld_loop", "loop", children + ([looping(levels - 1)] if levels else []))

    client = FolderFeishu(
        trees={"fld_root": [leaf("fil_a", "a.txt", b"a"), looping(5), leaf("fil_a", "a-again.txt", b"a")]},
        files={"fil_a": b"a", "fil_b": b"b"},
    )

    event = send(client, folder_message())

    assert sorted(client.downloaded) == ["fil_a", "fil_b"]
    listing = listing_of(event)
    assert [e["path"] for e in listing["entries"]] == ["a.txt", "loop", "loop/b.txt"]
    assert listing["notices"] == []
    assert event.text.splitlines()[0] == 'Contents of folder "project" (2 files, 1 subfolder):'
    assert [Path(path).read_bytes() for path in files_urls(event)] == [b"a", b"b"]


def test_a_folder_key_is_sent_as_one_encoded_path_segment():
    key = "fld/../x?y=1#z"
    client = FolderFeishu(trees={key: [leaf("fil_a", "a.txt", b"a")]}, files={"fil_a": b"a"})

    event = send(client, folder_message(key=key))

    request, = client.listings
    assert request.uri == "/open-apis/im/v1/files/fld%2F..%2Fx%3Fy%3D1%23z/folder"
    assert dict(request.queries) == {"srctype": "message", "srcid": "om_folder", "recursive": "true"}
    assert [Path(path).read_bytes() for path in files_urls(event)] == [b"a"]


@pytest.mark.parametrize("marker", [{"has_more": True}, {"page_token": "next"}])
def test_a_listing_feishu_marks_incomplete_is_not_presented_as_complete(marker):
    client = FolderFeishu(bodies={"fld_root": {"code": 0, "data": {"items": [leaf("fil_a", "a.txt", b"a")], **marker}}},
                          files={"fil_a": b"a"})

    event = send(client, folder_message())

    assert "Feishu listed only part of the folder." in event.text
    assert len(files_urls(event)) == 1


def test_hostile_names_never_leave_the_cache_or_become_path_segments():
    from gateway.platforms.base import get_document_cache_dir

    client = FolderFeishu(
        trees={"fld_root": [folder_node("fld_up", "..", [leaf("fil_a", "../../../outside.txt", b"a")]),
                            folder_node("fld_sep", "x/../y", [leaf("fil_b", "..\\..\\evil.txt", b"b")]),
                            leaf("fil_c", "\x00\x1f", b"c")]},
        files={"fil_a": b"a", "fil_b": b"b", "fil_c": b"c"},
    )

    event = send(client, folder_message(name="../../root"))

    assert len(files_urls(event)) == 3
    assert all(Path(path).resolve().parent == get_document_cache_dir().resolve() for path in files_urls(event))
    listing = listing_of(event)
    assert listing["folder"] == ".._.._root"
    for entry in listing["entries"]:
        p = PurePosixPath(entry["path"])
        assert not p.is_absolute() and ".." not in p.parts and "." not in p.parts


def test_tree_and_listing_name_the_files_where_a_sandboxed_agent_sees_them(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    client = FolderFeishu(trees={"fld_root": [leaf("fil_a", "a.txt", b"a")]}, files={"fil_a": b"a"})

    event = send(client, folder_message())

    file_paths = files_urls(event)
    host_path = file_paths[0]
    agent_path = f"/root/.hermes/cache/documents/{Path(host_path).name}"
    assert tree_path(event, "a.txt") == agent_path
    listing_line = next(line for line in event.text.splitlines() if line.startswith("Full listing (JSON): "))
    assert listing_line.startswith("Full listing (JSON): /root/.hermes/cache/documents/")
    listing_agent_path = listing_line.removeprefix("Full listing (JSON): ")
    entry_path = listing_of(event)["entries"][0]["local_path"]  # relative: valid wherever the cache is mounted
    assert str(PurePosixPath(listing_agent_path).parent / entry_path) == agent_path


# --- folders inside a post ---

def test_a_folder_in_a_posts_attachment_zone_follows_the_caption_and_is_listed_not_downloaded_as_a_file():
    post = {"zh_cn": {"title": "", "content": [[{"tag": "text", "text": "Review this folder."}]]},
            "files": [{"key": "fld_src", "name": "src", "is_folder": True}, {"key": "fil_zip", "name": "src.zip"}]}
    client = FolderFeishu(trees={"fld_src": [leaf("fil_main", "main.py", b"print(1)")]},
                          files={"fil_zip": b"PK\x03\x04", "fil_main": b"print(1)"}, types={"fil_zip": "application/zip"})

    event = send(client, inbound_message("om_post", "post", json.dumps(post)))

    assert "fld_src" not in client.downloaded  # a folder key is a container, not a resource
    assert sorted(client.downloaded) == ["fil_main", "fil_zip"]
    assert event.text.startswith('Review this folder.\n[Folder: src]\n[Attachment: src.zip]\n\nContents of folder "src"')
    assert Path(tree_path(event, "main.py")).read_bytes() == b"print(1)"
    # media order: file attachment from post, then listing, then folder file
    assert event.media_types[0] == "application/zip"
    assert "text/plain" in event.media_types


# --- quoted folders ---

def quote_client(tree_items, files, **kwargs):
    folder_item = item("om_folder", "folder", {"file_key": "fld_root", "file_name": "project"}, upper=None)
    folder_item.parent_id = ""
    return FolderFeishu(trees={"fld_root": tree_items}, files=files, messages={"om_folder": [folder_item]}, **kwargs)


def reply(client, *, thread=False):
    message = inbound_message("om_reply", "text", json.dumps({"text": "what is in here?"}), parent_id="om_folder")
    if thread:
        message.thread_id, message.root_id = "omt_topic", "om_folder"
    return send(client, message)


def test_a_reply_to_a_folder_quotes_its_tree_and_attaches_its_files():
    client = quote_client([folder_node("fld_docs", "docs", [leaf("fil_a", "a.txt", b"quoted")])], {"fil_a": b"quoted"})

    event = reply(client)

    assert event.text == "what is in here?"
    assert 'Contents of folder "project" (1 file, 1 subfolder)' in event.reply_to_text
    assert '- docs/' in event.reply_to_text
    assert '- a.txt → ' in event.reply_to_text
    # listing takes slot 0; file takes the remaining slot
    txt_paths = [u for u in event.media_urls if not u.endswith(".folder.json")]
    assert len(txt_paths) == 1 and Path(txt_paths[0]).read_bytes() == b"quoted"
    assert [(r.message_id, r.file_key) for r in client.resource_gets] == [("om_folder", "fil_a")]


def test_a_quoted_folder_downloads_only_the_files_the_quote_may_attach():
    from plugins.platforms.feishu.feishu_quote import MAX_QUOTED_MEDIA

    count = MAX_QUOTED_MEDIA + 3
    client = quote_client([leaf(f"fil_{n}", f"{n}.txt", b"q") for n in range(count)], {f"fil_{n}": b"q" for n in range(count)})

    event = reply(client)

    # listing takes 1 of MAX_QUOTED_MEDIA slots; MAX_QUOTED_MEDIA - 1 files fit
    expected_files = MAX_QUOTED_MEDIA - 1
    assert len(client.resource_gets) == expected_files
    file_urls = [u for u in event.media_urls if not u.endswith(".folder.json")]
    assert len(file_urls) == expected_files
    assert f"- {count - 1}.txt (not downloaded: file limit reached)" in event.reply_to_text


def test_a_message_posted_in_a_folders_topic_names_the_folder_without_listing_it():
    client = quote_client([leaf("fil_a", "a.txt", b"a")], {"fil_a": b"a"})

    event = reply(client, thread=True)

    assert event.reply_to_text == "[Folder: project]"
    assert client.listings == [] and event.media_urls == []


def test_a_quoted_folder_that_cannot_be_listed_says_so_in_the_quote():
    client = quote_client([], {}, bodies={"fld_root": {"code": 234040, "msg": "invisible"}})

    event = reply(client)

    assert event.reply_to_text == (
        '[Folder: project]\n\nFolder "project" could not be read (Feishu error 234040); none of its files were downloaded.'
    )


def test_a_folder_message_forwarded_inside_a_merge_forward_reads_as_its_name():
    from plugins.platforms.feishu.adapter import normalize_feishu_message
    from plugins.platforms.feishu.feishu_forward import display_text

    normalized = normalize_feishu_message(message_type="folder", raw_content=json.dumps({"file_key": "fld_x", "file_name": "assets"}))

    assert display_text(normalized) == "[Folder: assets]"
    assert normalized.media_refs == [] and [ref.file_key for ref in normalized.folder_refs] == ["fld_x"]


def test_concurrent_downloads_stay_within_the_folder_size_cap(monkeypatch):
    import plugins.platforms.feishu.feishu_folder as feishu_folder

    monkeypatch.setattr("gateway.platforms.base.get_inbound_media_max_bytes", lambda: 20)
    monkeypatch.setattr(feishu_folder, "DOWNLOAD_CONCURRENCY", 8)
    files = {f"fil_{n}": b"z" * 6 for n in range(8)}
    client = FolderFeishu(trees={"fld_root": [leaf(key, f"{key}.txt", data) for key, data in files.items()]}, files=files)

    event = send(client, folder_message())

    assert sum(Path(path).stat().st_size for path in files_urls(event)) <= 20
    assert len(files_urls(event)) == 3


def test_a_quoted_image_without_a_folder_is_attached_and_nothing_is_listed():
    image_item = item("om_image", "image", '{"image_key": "img_chart"}', upper=None)
    image_item.parent_id = ""
    client = FolderFeishu(files={"img_chart": PNG}, types={"img_chart": "image/png"}, messages={"om_image": [image_item]})
    message = inbound_message("om_reply", "text", json.dumps({"text": "what is this?"}), parent_id="om_image")

    event = dispatched_event(make_adapter(client), message)

    assert client.listings == []
    assert event.reply_to_text == "[Image]"
    assert event.media_types == ["image/png"]


def _on_event_loop_thread():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def test_a_folder_is_listed_downloaded_and_cached_off_the_event_loop(monkeypatch):
    """Listing, downloads and every cache write (file, not-an-image fallback, listing) run on worker threads."""
    import gateway.platforms.base as base

    calls = []

    class WatchedFolderFeishu(FolderFeishu):
        def request(self, request):
            calls.append(("listing", _on_event_loop_thread()))
            return super().request(request)

        def _get_resource(self, request):
            calls.append(("download", _on_event_loop_thread()))
            return super()._get_resource(request)

    def watched(name):
        real = getattr(base, name)

        def write(*args, **kwargs):
            calls.append((name, _on_event_loop_thread()))
            return real(*args, **kwargs)

        return write

    for name in ("cache_media_bytes", "cache_document_from_bytes"):
        monkeypatch.setattr(base, name, watched(name))
    client = WatchedFolderFeishu(
        trees={"fld_root": [leaf("fil_a", "a.txt", b"a"), leaf("fil_fake", "photo.png", b"not an image")]},
        files={"fil_a": b"a", "fil_fake": b"not an image"}, types={"fil_fake": "image/png"},
    )

    event = send(client, folder_message())

    assert [Path(path).read_bytes() for path in files_urls(event)] == [b"a", b"not an image"]
    assert {name for name, _ in calls} == {"listing", "download", "cache_media_bytes", "cache_document_from_bytes"}
    assert [name for name, on_loop in calls if on_loop] == []


# --- a folder and the note behind it ---

class SlowFolderFeishu(FolderFeishu):
    """File downloads take ``delay`` seconds each (on the adapter's SDK executor thread)."""

    def __init__(self, *args, delay, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay = delay

    def _get_resource(self, request):
        time.sleep(self.delay)
        return super()._get_resource(request)


def test_a_note_sent_behind_a_folder_joins_its_turn_folder_first():
    """Feishu sends a folder without a caption, so "folder, then instruction" is the normal use."""
    files = {f"fil_{n}": f"file {n}".encode() for n in range(4)}
    client = SlowFolderFeishu(
        trees={"fld_root": [leaf(key, f"{key}.txt", data) for key, data in files.items()]}, files=files, delay=0.2,
    )

    turns = delivered_turns(
        make_adapter(client),
        inbound_event("om_folder", "folder", json.dumps({"file_key": "fld_root", "file_name": "project"})),
        inbound_event("om_note", "text", json.dumps({"text": "please review these"})),
    )

    (turn,) = turns
    assert turn.startswith('Contents of folder "project" (4 files, 0 subfolders):\n')
    assert turn.endswith("\nplease review these")


def test_folder_downloads_stop_at_the_time_budget_and_say_which_files_were_cut_off(monkeypatch):
    import threading

    import plugins.platforms.feishu.feishu_forward_order as forward_order

    class StalledFolderFeishu(FolderFeishu):
        released = threading.Event()

        def _get_resource(self, request):
            if request.file_key in {"fil_1", "fil_4"}:
                self.released.wait(10)  # on the adapter's SDK executor thread
            return super()._get_resource(request)

    monkeypatch.setattr(forward_order, "DOWNLOAD_BUDGET_SECONDS", 1.0)
    files = {f"fil_{n}": b"x" for n in range(6)}
    client = StalledFolderFeishu(trees={"fld_root": [leaf(key, f"{key}.txt", data) for key, data in files.items()]}, files=files)

    start = time.monotonic()
    try:
        event = send(client, folder_message())
    finally:
        client.released.set()
    elapsed = time.monotonic() - start

    assert elapsed < 5  # stopped at the budget, not after the stalled 10s downloads
    statuses = {e["path"]: e["status"] for e in listing_of(event)["entries"]}
    assert statuses == {f"fil_{n}.txt": "not downloaded: time limit reached" if n in (1, 4) else "downloaded" for n in range(6)}
    assert len(files_urls(event)) == 4
    assert "- fil_1.txt (not downloaded: time limit reached)" in event.text


def test_a_long_folder_tree_does_not_push_its_note_into_a_turn_of_its_own():
    from plugins.platforms.feishu.adapter import _DEFAULT_TEXT_BATCH_MAX_CHARS

    long_name = "n" * 120
    count = _DEFAULT_TEXT_BATCH_MAX_CHARS // len(long_name) + 1
    files = {f"fil_{n}": b"x" for n in range(count)}
    client = FolderFeishu(trees={"fld_root": [leaf(key, f"{long_name}{key}.txt") for key in files]}, files=files)

    turns = delivered_turns(
        make_adapter(client),
        inbound_event("om_folder", "folder", json.dumps({"file_key": "fld_root", "file_name": "project"})),
        inbound_event("om_note", "text", json.dumps({"text": "please review these"})),
    )

    assert len(turns) == 1 and len(turns[0]) > _DEFAULT_TEXT_BATCH_MAX_CHARS
    assert turns[0].endswith("\nplease review these")

