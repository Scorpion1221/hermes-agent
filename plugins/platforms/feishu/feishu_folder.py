"""Feishu folders: a native ``folder`` message, or a folder in a rich-text post's attachment zone.

A folder's key is a container, not a downloadable file. ``GET im/v1/files/{folder_key}/folder``
with ``srctype=message&srcid={message_id}&recursive=true`` (both source params required) lists it as
a tree: ``data.items`` of ``{file_key, name, is_folder, mimetype, size, children: [...]}``. Each file
then downloads like any message file, through the message that carries the folder
(``im/v1/messages/{message_id}/resources/{file_key}?type=file``).

Files land flat in the media cache (upstream's ``cache_media_bytes``: documents as ``doc_<id>_<name>``,
images, audio and video in their own caches): the cache sweep prunes only the files at the top of a
cache directory, so a materialized tree would never be cleaned up, and a remote name never becomes a
local path component. The hierarchy lives in the text (an indented tree naming each
downloaded file's path) and in a JSON listing file beside the files. Listing and downloads are
bounded (entries, depth, files, and both ``gateway.max_inbound_media_bytes`` and a download time
budget for all the folders of a message together); whatever a bound or a failure leaves out is said
in the text. Up to ``max_media``
files are attached to the event, never audio (attached, it would be transcribed as the user's own
voice); none is inlined. A quoted folder downloads only the files it may attach (``feishu_quote``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

MAX_ENTRIES = 200  # files and subfolders listed per folder
MAX_DEPTH = 10  # nesting levels listed below the folder itself
MAX_FILES = 100  # files downloaded per folder
MAX_MEDIA = 10  # files attached to the event, all folders of a message together
MAX_TREE_LINES = 60  # tree lines in the text; the listing file has every entry
DOWNLOAD_CONCURRENCY = 4
_MAX_LISTING_BYTES = 4 * 1024 * 1024
_UNSAFE_NAME_CHARS = re.compile(r"[\x00-\x1f\x7f/\\]")


@dataclass
class FolderContent:
    text: str = ""
    media_urls: list[str] = field(default_factory=list)
    media_types: list[str] = field(default_factory=list)


@dataclass
class _Entry:
    path: str  # "/"-joined sanitized names, relative to the folder
    name: str
    depth: int
    is_folder: bool
    file_key: str = ""
    size: int = 0
    status: str = ""  # "downloaded", or why the file is not
    local_path: str = ""  # host path in the media cache
    media_type: str = ""


@dataclass
class _Budget:
    """Bytes (0 = unlimited) and time (loop-clock ``deadline``) every folder of one message shares."""

    limit: int
    deadline: float
    used: int = 0
    reserved: int = 0  # declared sizes of the downloads in flight

    def reserve(self, size: int) -> bool:
        if self.limit and self.used + self.reserved + size > self.limit:
            return False
        self.reserved += size
        return True

    def take(self, size: int) -> bool:
        """Account a finished download; False when its real size does not fit (its listed size lied)."""
        if self.limit and self.used + self.reserved + size > self.limit:
            return False
        self.used += size
        return True


class _Unlisted(Exception):
    """The folder listing could not be read; the message says why."""


def normalize_folder_message(payload: dict[str, Any], mentions: list[Any]) -> Any:
    """A native ``folder`` message (content ``{"file_key", "file_name"}``): listed and downloaded apart.

    It is dispatched as text, like a forward: Feishu sends a folder without a caption, so the note sent
    behind it joins its turn in the text batch (``feishu_forward_order``)."""
    from .adapter import FeishuNormalizedMessage, FeishuPostMediaRef
    from .feishu_post import folder_placeholder

    file_key = str(payload.get("file_key") or "").strip()
    name = str(payload.get("file_name") or "").strip()
    return FeishuNormalizedMessage(
        raw_type="folder", text_content="",
        folder_refs=[FeishuPostMediaRef(file_key=file_key, file_name=name, resource_type="folder")] if file_key else [],
        relation_kind="folder", metadata={"placeholder_text": folder_placeholder(name)}, mentions=mentions,
    )


async def fetch_folders(
    adapter: Any, message_id: str, refs: list[Any], *,
    max_media: Optional[int] = None, max_files: Optional[int] = None,
) -> FolderContent:
    """The folders of message ``message_id``: a text block each, plus up to ``max_media`` (default
    ``MAX_MEDIA``) attached files. Listings count toward the cap (listing first, then files). At most
    ``max_files`` (default ``MAX_FILES``) files of each folder are downloaded, all of them by
    ``DOWNLOAD_BUDGET_SECONDS`` after the first listing; the tree names the rest."""
    from gateway.platforms.base import get_inbound_media_max_bytes

    from .feishu_forward_order import DOWNLOAD_BUDGET_SECONDS

    deadline = asyncio.get_running_loop().time() + DOWNLOAD_BUDGET_SECONDS
    max_media = MAX_MEDIA if max_media is None else max_media
    max_files = MAX_FILES if max_files is None else max_files
    budget = _Budget(max(0, await asyncio.to_thread(get_inbound_media_max_bytes)), deadline=deadline)
    content = FolderContent()
    blocks: list[str] = []
    for ref in refs:
        block, listing_path, files = await _fetch_folder(adapter, message_id, ref, budget, max_files)
        blocks.append(block)
        slot = max(max_media - len(content.media_urls), 0)
        if listing_path and slot:  # JSON as text; the gateway's note tells the agent to read it
            content.media_urls.append(listing_path)
            content.media_types.append("text/plain")
            slot -= 1
        for entry in files:
            if slot and entry.local_path and not entry.media_type.startswith("audio/"):
                content.media_urls.append(entry.local_path)
                content.media_types.append(entry.media_type)
                slot -= 1
    content.text = "\n\n".join(blocks)
    return content


async def _fetch_folder(
    adapter: Any, message_id: str, ref: Any, budget: _Budget, max_files: int,
) -> tuple[str, str, list[_Entry]]:
    """``(text block, listing path or "", listed files)`` for one folder."""
    name = _safe_name(ref.file_name, "folder")
    if not adapter._client or not message_id:
        return f'Folder "{name}" could not be read: Feishu is not connected.', "", []
    try:
        items, partial = await _list(adapter, message_id, ref.file_key)
    except _Unlisted as exc:
        return f'Folder "{name}" could not be read ({exc}); none of its files were downloaded.', "", []
    except Exception:
        logger.warning("[Feishu] Failed to list folder %s of message %s", ref.file_key, message_id, exc_info=True)
        return f'Folder "{name}" could not be read; none of its files were downloaded.', "", []
    walk = _Walk()
    if partial:
        walk.notices.append("Feishu listed only part of the folder.")
    walk.add(items, "", 1)
    if not walk.entries:
        return "\n".join([f'Folder "{name}" {"has no readable entries" if walk.notices else "is empty"}.', *walk.notices]), "", []
    files = [entry for entry in walk.entries if not entry.is_folder]
    for entry in files[max_files:]:
        entry.status = "not downloaded: file limit reached"
    await _download_all(adapter, message_id, files[:max_files], budget)
    downloaded = sum(1 for entry in files if entry.local_path)
    logger.info("[Feishu] Folder of message %s: %d entries listed, %d/%d files downloaded",
                message_id, len(walk.entries), downloaded, len(files))
    listing_path = await _write_listing(name, walk)
    return _render(name, walk, listing_path), listing_path, files


async def _list(adapter: Any, message_id: str, folder_key: str) -> tuple[list[Any], bool]:
    """The listed tree's top-level items, and whether Feishu says there is more than it returned."""
    content = await adapter._tenant_get_raw(
        f"/open-apis/im/v1/files/{quote(folder_key, safe='')}/folder",
        queries=[("srctype", "message"), ("srcid", message_id), ("recursive", "true")],
    )
    if not content:
        raise _Unlisted("Feishu returned no listing")
    if len(content) > _MAX_LISTING_BYTES:
        raise _Unlisted("its listing is too large")
    try:
        payload = json.loads(content)
    except ValueError:
        raise _Unlisted("Feishu returned an unreadable listing") from None
    code = payload.get("code") if isinstance(payload, dict) else None
    if code != 0:
        raise _Unlisted(f"Feishu error {code}")
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    items = data.get("items")
    if not isinstance(items, list):
        raise _Unlisted("Feishu returned an unreadable listing")
    return items, bool(data.get("has_more") or data.get("page_token"))


class _Walk:
    """The listed tree, depth-first in Feishu's order, within ``MAX_ENTRIES`` and ``MAX_DEPTH``."""

    def __init__(self) -> None:
        self.entries: list[_Entry] = []
        self.notices: list[str] = []
        self._seen: set[str] = set()  # a key listed twice (or a folder inside itself) is listed once

    def add(self, items: list[Any], parent: str, depth: int) -> None:
        for item in items:
            if len(self.entries) >= MAX_ENTRIES:
                self._notice(f"The folder has more than {MAX_ENTRIES} entries; the rest are not listed.")
                return
            key = item.get("file_key") if isinstance(item, dict) else None
            key = key.strip() if isinstance(key, str) else ""
            if not key:
                self._notice("Some entries of the folder could not be read and are not listed.")
            elif key not in self._seen:
                self._seen.add(key)
                self._add_entry(item, key, parent, depth)

    def _add_entry(self, item: dict[str, Any], key: str, parent: str, depth: int) -> None:
        from .adapter import _to_boolean

        name = _safe_name(item.get("name"), "unnamed")
        path = f"{parent}/{name}" if parent else name
        is_folder = _to_boolean(item.get("is_folder"))
        self.entries.append(_Entry(
            path=path, name=name, depth=depth, is_folder=is_folder, file_key=key, size=_size(item.get("size")),
        ))
        children = item.get("children", [])
        if not is_folder:
            return
        if not isinstance(children, list) or _size(item.get("children_count")) > len(children):
            self._notice("Some entries of the folder could not be read and are not listed.")
            children = children if isinstance(children, list) else []
        if not children:
            return
        if depth >= MAX_DEPTH:
            self._notice(f"Entries nested more than {MAX_DEPTH} levels deep are not listed.")
            return
        self.add(children, path, depth + 1)

    def _notice(self, text: str) -> None:
        if text not in self.notices:
            self.notices.append(text)


def _safe_name(raw: Any, fallback: str) -> str:
    """A listed name as one path segment: separators and control characters replaced, never ``.``/``..``."""
    name = _UNSAFE_NAME_CHARS.sub("_", str(raw or "")).strip()
    return fallback if name in {"", ".", ".."} else name


def _size(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


async def _download_all(adapter: Any, message_id: str, files: list[_Entry], budget: _Budget) -> None:
    """Download ``files`` (a few at a time) until the budget's deadline; the rest are named as cut off."""
    semaphore = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)
    tasks = [asyncio.ensure_future(_download(adapter, message_id, entry, budget, semaphore)) for entry in files]
    if tasks:
        _done, pending = await asyncio.wait(tasks, timeout=max(budget.deadline - asyncio.get_running_loop().time(), 0))
        for task in pending:
            task.cancel()
    for entry in files:
        if not entry.status:
            entry.status = "not downloaded: time limit reached"


async def _download(adapter: Any, message_id: str, entry: _Entry, budget: _Budget, semaphore: asyncio.Semaphore) -> None:
    async with semaphore:
        if not budget.reserve(entry.size):
            entry.status = "not downloaded: over the inbound media size limit for the folder"
            return
        try:
            fetched = await _fetch_bytes(adapter, message_id, entry)
        finally:
            budget.reserved -= entry.size
        if fetched is None:
            entry.status = "download failed"
            return
        data, content_type = fetched
        if not budget.take(len(data)):
            entry.status = "not downloaded: over the inbound media size limit for the folder"
            return
        await _cache(entry, data, content_type)


async def _fetch_bytes(adapter: Any, message_id: str, entry: _Entry) -> tuple[bytes, str] | None:
    """The file's bytes (an empty file is ``b""``) and Content-Type; None when Feishu did not return the file."""
    try:
        response = await adapter._fetch_message_resource(message_id=message_id, file_key=entry.file_key, resource_type="file")
    except Exception:
        logger.warning("[Feishu] Failed to download folder file %s/%s", message_id, entry.file_key, exc_info=True)
        return None
    if not response or not response.success() or getattr(response, "file", None) is None:
        logger.debug(
            "[Feishu] Folder file download failed for %s/%s: %s %s", message_id, entry.file_key,
            getattr(response, "code", "unknown"), getattr(response, "msg", "request failed"),
        )
        return None
    return adapter._read_binary_response(response), adapter._get_response_header(response, "Content-Type")


async def _cache(entry: _Entry, data: bytes, content_type: str) -> None:
    from gateway.platforms.base import cache_document_from_bytes_async, cache_media_bytes_async

    try:
        cached = await cache_media_bytes_async(data, filename=entry.name, mime_type=content_type)
        if cached is not None:
            entry.local_path, entry.media_type = cached.path, cached.media_type
        else:  # named or served as an image but not one: still the user's file
            entry.local_path = await cache_document_from_bytes_async(data, entry.name)
            entry.media_type = "application/octet-stream"
    except (OSError, ValueError):
        logger.warning("[Feishu] Failed to cache folder file %s", entry.file_key, exc_info=True)
        entry.status = "download failed"
        return
    entry.status = "downloaded"


async def _write_listing(name: str, walk: _Walk) -> str:
    """The JSON listing file (every entry, its status and agent-visible path); "" when it can't be written."""
    from gateway.platforms.base import cache_document_from_bytes_async
    from tools.credential_files import to_agent_visible_cache_path

    entries = [
        {"path": entry.path, "type": "folder"} if entry.is_folder else {
            "path": entry.path, "type": "file", "size": entry.size, "status": entry.status,
            **({"local_path": to_agent_visible_cache_path(entry.local_path)} if entry.local_path else {}),
        }
        for entry in walk.entries
    ]
    body = json.dumps({"folder": name, "entries": entries, "notices": walk.notices}, ensure_ascii=False, indent=1)
    try:
        return await cache_document_from_bytes_async(body.encode("utf-8"), f"{name}.folder.json")
    except (OSError, ValueError):
        logger.warning("[Feishu] Failed to write a folder listing", exc_info=True)
        return ""


def _render(name: str, walk: _Walk, listing_path: str) -> str:
    from tools.credential_files import to_agent_visible_cache_path

    files = sum(1 for entry in walk.entries if not entry.is_folder)
    folders = len(walk.entries) - files
    lines = [f'Contents of folder "{name}" ({_count(files, "file")}, {_count(folders, "subfolder")}):']
    for entry in walk.entries[:MAX_TREE_LINES]:
        indent = "  " * (entry.depth - 1)
        if entry.is_folder:
            lines.append(f"{indent}- {entry.name}/")
        elif entry.local_path:
            lines.append(f"{indent}- {entry.name} → {to_agent_visible_cache_path(entry.local_path)}")
        else:
            lines.append(f"{indent}- {entry.name} ({entry.status})")
    if len(walk.entries) > MAX_TREE_LINES:
        lines.append(f"- … {len(walk.entries) - MAX_TREE_LINES} more entries, in the full listing")
    lines += walk.notices
    if listing_path:
        lines.append(f"Full listing (JSON): {to_agent_visible_cache_path(listing_path)}")
    return "\n".join(lines)


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" + ("" if number == 1 else "s")
