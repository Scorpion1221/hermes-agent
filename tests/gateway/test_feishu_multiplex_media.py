"""Feishu attachments on a gateway serving several profiles (``gateway.multiplex_profiles``).

The adapter caches files under the launch home before the event is routed; the routed turn's real
``rehome_inbound_media`` then MOVES the event's attachments into the routed profile's cache and
rewrites their raw paths in the text. Real temp homes A (launch, local backend) and B (routed,
docker backend), real ``_profile_runtime_scope``, multiplex on. The Feishu client is faked at the SDK
boundary (shared harness); all names, keys and contents are synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import hermes_yaml as yaml

from agent import secret_scope as ss
from gateway.run import _profile_runtime_scope
from gateway.run_inbound import GatewayInboundMixin
from gateway.run_inbound_media import rehome_inbound_media
from tests.gateway.test_feishu_fetched_content import (  # noqa: F401  (autouse SDK fakes)
    _fake_lark_requests,
    dispatched_event,
    make_adapter,
)
from tests.gateway.test_feishu_folder import FolderFeishu, folder_message, leaf
from tests.gateway.test_feishu_quote_chain import ChainFeishu, inbound, msg

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    a = tmp_path / ".hermes"
    b = a / "profiles" / "b"
    b.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(a))
    for backend, home in (("local", a), ("docker", b)):
        (home / "config.yaml").write_text(yaml.safe_dump({"terminal": {"backend": backend}}), encoding="utf-8")
    ss.set_multiplex_active(True)
    yield a, b
    ss.set_multiplex_active(False)


def launch_cache_files(home: Path) -> list[Path]:
    return sorted(p for p in (home / "cache").rglob("*") if p.is_file() and "profiles" not in p.parts)


def test_a_folder_routed_to_another_profile_leaves_nothing_behind_and_its_listing_still_resolves(two_homes):
    a, b = two_homes
    client = FolderFeishu(
        trees={"fld_root": [leaf("fil_a", "a.txt", b"aaa"), leaf("fil_voice", "memo.ogg", b"OggS"), leaf("fil_b", "b.txt", b"bbb")]},
        files={"fil_a": b"aaa", "fil_voice": b"OggS", "fil_b": b"bbb"}, types={"fil_voice": "audio/ogg"},
    )
    event = dispatched_event(make_adapter(client), folder_message())

    with _profile_runtime_scope(b):  # the routed turn
        rehome_inbound_media(event)
        notes = GatewayInboundMixin._prepend_inbound_document_notes(event, event.text)

    assert launch_cache_files(a) == []  # no file of B's turn stays in A's cache
    listing_path = Path(next(url for url in event.media_urls if url.endswith(".folder.json")))
    assert listing_path.is_relative_to(b / "cache")
    listing = json.loads(listing_path.read_text(encoding="utf-8"))
    downloaded = {e["path"]: listing_path.parent / e["local_path"] for e in listing["entries"] if e.get("local_path")}
    assert sorted(downloaded) == ["a.txt", "b.txt"] and all(path.is_file() for path in downloaded.values())
    assert sorted(map(str, downloaded.values())) == sorted(u for u in event.media_urls if not u.endswith(".json"))
    memo = next(e for e in listing["entries"] if e["path"] == "memo.ogg")
    assert "local_path" not in memo and memo["status"] == "not kept: never attached"
    # The tree names each file as the gateway's note for that attachment does, in B's sandbox form.
    for path in downloaded.values():
        assert f"→ attached as {path.name}" in event.text
        assert f"/root/.hermes/cache/documents/{path.name}" in notes
    assert str(a) not in event.text


def test_a_folder_with_more_files_than_attachment_room_downloads_only_those_it_attaches(two_homes):
    """Not just "nothing left behind": a file that could never be attached is never requested from Feishu."""
    from plugins.platforms.feishu.feishu_folder import MAX_MEDIA

    a, b = two_homes
    count = MAX_MEDIA + 4
    files = {f"fil_{n:02d}": f"file {n}".encode() for n in range(count)}
    client = FolderFeishu(trees={"fld_root": [leaf(key, f"{key}.txt", data) for key, data in files.items()]}, files=files)
    event = dispatched_event(make_adapter(client), folder_message())

    room = MAX_MEDIA - 1  # the listing takes the first slot
    assert sorted(client.downloaded) == list(files)[:room]  # a few download at once, in any order
    assert f"- fil_{count - 1:02d}.txt (not downloaded: file limit reached)" in event.text
    with _profile_runtime_scope(b):
        rehome_inbound_media(event)
    attached = [Path(url) for url in event.media_urls if not url.endswith(".folder.json")]
    assert [path.read_bytes() for path in attached] == list(files.values())[:room]
    assert launch_cache_files(a) == []


def test_two_events_quoting_one_image_each_carry_their_own_file_to_their_profile(two_homes):
    a, b = two_homes
    client = ChainFeishu({"om_img": [msg("om_img", {"image_key": "img_chart"}, msg_type="image")]})
    adapter = make_adapter(client)
    first = dispatched_event(adapter, inbound("om_r1", {"text": "this?"}, parent="om_img").event.message)
    second = dispatched_event(adapter, inbound("om_r2", {"text": "and this?"}, parent="om_img").event.message)

    with _profile_runtime_scope(b):
        rehome_inbound_media(first)
        rehome_inbound_media(second)

    assert [len(e.media_urls) for e in (first, second)] == [1, 1]
    assert first.media_urls != second.media_urls
    for event in (first, second):
        path = Path(event.media_urls[0])
        assert path.is_relative_to(b / "cache") and path.read_bytes().startswith(b"\x89PNG")
    assert launch_cache_files(a) == []


def test_without_multiplexing_replies_quoting_one_image_still_share_one_download(tmp_path):
    client = ChainFeishu({"om_img": [msg("om_img", {"image_key": "img_chart"}, msg_type="image")]})
    adapter = make_adapter(client)

    first = dispatched_event(adapter, inbound("om_r1", {"text": "this?"}, parent="om_img").event.message)
    second = dispatched_event(adapter, inbound("om_r2", {"text": "and this?"}, parent="om_img").event.message)

    assert first.media_urls == second.media_urls and len(client.resource_gets) == 1
