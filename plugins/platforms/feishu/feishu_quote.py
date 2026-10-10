"""The quoted (replied-to) Feishu message: ``reply_to_text``, ``reply_to_is_own_message`` and its media.

Every fetched message names the message it replies to in ``parent_id``, so a reply to a reply can
be walked up toward the root. The gateway wraps ``reply_to_text`` as ``[Replying to: "…"]``; when
the quoted message is itself a reply, the earlier messages of that chain (oldest first, one bounded
line each, up to ``quote_chain_max_depth`` messages in all) precede the directly quoted message,
which keeps its full text. The ancestor walk and the name lookup after it are each time-boxed: a
slow API yields a shorter (or less named) chain, never a late reply. Fetched messages are
LRU-cached by id, and a fetch the budget gave up on still completes into the caches for the next
reply in that chain.

Messages are fetched with card bodies as raw ``json_card``, so a quoted Card JSON 2.0 card (the
bot's own replies) reads as its content and a quoted merge-forward as its transcript (text only).
Images and files of the directly quoted message, never of its ancestors, are downloaded and
attached to the event (capped), as other adapters attach quoted images. The download shares the
walk's time budget and is cached per quoted message (while its files exist), so a slow download
is left out of that reply but reused, not repeated, by the next reply quoting the same message.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Optional

logger = logging.getLogger(__name__)

CHAIN_BUDGET_SECONDS = 2.5  # quoted message + ancestor walk
NAMES_BUDGET_SECONDS = 0.5  # one batch lookup naming the chain's senders
MAX_ANCESTOR_CHARS = 500
MAX_ANCESTORS_CHARS = 4000
MAX_QUOTED_MEDIA = 5
MEDIA_CACHE_SIZE = 64  # quoted messages whose downloaded attachments are kept for later replies
_TYPE_LABELS = {"interactive": "card", "media": "video", "merge_forward": "merged forward", "post": "rich text"}
# agent/context_compressor.py ends the gateway's reply pointer at the first `"]` + blank line.
_POINTER_END = '"]\n\n'


@dataclass(frozen=True)
class QuoteNode:
    message_id: str
    msg_type: str
    text: str
    sender: Any
    parent_id: str
    image_keys: tuple[str, ...] = ()
    media_refs: tuple[Any, ...] = ()


@dataclass
class QuotedReply:
    text: Optional[str] = None
    is_own: bool = False
    media_urls: list[str] = field(default_factory=list)
    media_types: list[str] = field(default_factory=list)


async def fetch_quoted_reply(adapter: Any, message_id: Optional[str]) -> QuotedReply:
    """What the event needs from the message it quotes; empty when nothing is quoted or it can't be read."""
    if not message_id or not adapter._client:
        return QuotedReply()
    deadline = asyncio.get_running_loop().time() + CHAIN_BUDGET_SECONDS
    quoted = await fetch_node(adapter, message_id)
    if quoted is None:
        return QuotedReply()
    (ancestors, complete), (media_urls, media_types) = await asyncio.gather(
        _walk_ancestors(adapter, quoted, deadline), _attachments(adapter, quoted, deadline),
    )
    text = await _render(adapter, quoted, ancestors, complete)
    return QuotedReply(
        text=text.replace(_POINTER_END, '"]\n \n') or None, is_own=_is_own(adapter, quoted.sender),
        media_urls=media_urls, media_types=media_types,
    )


async def fetch_node(adapter: Any, message_id: str) -> Optional[QuoteNode]:
    """``message_id`` as a chain node (LRU-cached on the adapter); None when it can't be fetched."""
    from .adapter import _FEISHU_MESSAGE_TEXT_CACHE_SIZE
    from .feishu_forward import fetch_message_items

    cache = adapter._quote_node_cache
    if message_id in cache:
        cache.move_to_end(message_id)
        return cache[message_id]
    try:
        items = await fetch_message_items(adapter, message_id)
        if not items:
            return None
        node = await _build_node(adapter, message_id, items)
    except Exception:
        logger.warning("[Feishu] Failed to fetch quoted message %s", message_id, exc_info=True)
        return None
    cache[message_id] = node
    while len(cache) > _FEISHU_MESSAGE_TEXT_CACHE_SIZE:
        cache.popitem(last=False)
    return node


async def _build_node(adapter: Any, message_id: str, items: list[Any]) -> QuoteNode:
    from .feishu_forward import display_text, render_merge_forward

    item = items[0]
    msg_type = str(getattr(item, "msg_type", "") or "")
    transcript = await render_merge_forward(adapter, message_id, items) if msg_type == "merge_forward" else None
    node = QuoteNode(
        message_id=message_id, msg_type=msg_type, text=transcript.text if transcript else "",
        sender=getattr(item, "sender", None), parent_id=str(getattr(item, "parent_id", "") or ""),
    )
    if transcript is not None:
        return node
    raw_content = getattr(getattr(item, "body", None), "content", "") or ""
    normalized = adapter._normalize(msg_type, raw_content, getattr(item, "mentions", None))
    return dataclasses.replace(
        node, text=display_text(normalized), image_keys=tuple(normalized.image_keys),
        media_refs=tuple(normalized.media_refs),
    )


async def _walk_ancestors(adapter: Any, quoted: QuoteNode, deadline: float) -> tuple[list[QuoteNode], bool]:
    """``quoted``'s ancestors, nearest first; False when depth, time or a failed fetch cut the chain short."""
    if not quoted.parent_id or adapter._quote_chain_max_depth <= 1:
        return [], True  # not itself a reply, or single hop by configuration
    loop = asyncio.get_running_loop()
    ancestors: list[QuoteNode] = []
    seen = {quoted.message_id}
    parent_id = quoted.parent_id
    while parent_id and parent_id not in seen:  # a malformed self-reference can't loop
        remaining = deadline - loop.time()
        if len(ancestors) >= adapter._quote_chain_max_depth - 1 or remaining <= 0:
            return ancestors, False
        node = await _within(adapter, fetch_node(adapter, parent_id), remaining)
        if node is None:
            return ancestors, False
        ancestors.append(node)
        seen.add(parent_id)
        parent_id = node.parent_id
    return ancestors, True


async def _within(adapter: Any, work: Awaitable[Any], timeout: float) -> Any:
    """``work``'s result, or None after ``timeout``; abandoned work still finishes (and fills the caches)."""
    tasks: set[asyncio.Future] = vars(adapter).setdefault("_quote_background_fetches", set())
    task = asyncio.ensure_future(work)
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout)
    except TimeoutError:
        return None


async def _attachments(adapter: Any, quoted: QuoteNode, deadline: float) -> tuple[list[str], list[str]]:
    """The quoted message's attachments as downloaded by the deadline; none when the download is still running."""
    # Quoted audio stays a text placeholder: attached, the gateway would transcribe it as the user's own voice.
    image_keys = list(quoted.image_keys[:MAX_QUOTED_MEDIA])
    files = [ref for ref in quoted.media_refs if ref.resource_type != "audio"]
    media_refs = files[:MAX_QUOTED_MEDIA - len(image_keys)]
    if not image_keys and not media_refs:
        return [], []
    cache: OrderedDict[str, asyncio.Future] = vars(adapter).setdefault("_quote_media_cache", OrderedDict())
    download = cache.get(quoted.message_id)
    if download is None or (download.done() and not _still_cached(download)):
        download = asyncio.ensure_future(_download(adapter, quoted, image_keys, media_refs))
    cache[quoted.message_id] = download
    cache.move_to_end(quoted.message_id)
    while len(cache) > MEDIA_CACHE_SIZE:
        cache.popitem(last=False)
    result = download.result() if download.done() else await _within(
        adapter, download, deadline - asyncio.get_running_loop().time(),
    )
    urls, types = result or ([], [])
    return list(urls), list(types)


def _still_cached(download: asyncio.Future) -> bool:
    """A finished download is reused while it produced files and they are still in the media cache."""
    if download.cancelled() or download.exception() is not None:
        return False
    urls = download.result()[0]
    return bool(urls) and all(os.path.exists(url) for url in urls)


async def _download(adapter: Any, quoted: QuoteNode, image_keys: list[str], media_refs: list[Any]) -> tuple[list[str], list[str]]:
    from .adapter import FeishuNormalizedMessage

    urls, types = await adapter._download_feishu_message_resources(
        message_id=quoted.message_id,
        normalized=FeishuNormalizedMessage(
            raw_type=quoted.msg_type, text_content="", image_keys=image_keys, media_refs=media_refs,
        ),
    )
    kept = [(url, media_type) for url, media_type in zip(urls, types) if not media_type.startswith("audio/")]
    return [url for url, _ in kept], [media_type for _, media_type in kept]


async def _render(adapter: Any, quoted: QuoteNode, ancestors: list[QuoteNode], complete: bool) -> str:
    if not ancestors and complete:
        return quoted.text  # nothing above it: reads exactly as the quoted message
    await _prefetch_names(adapter, [quoted, *ancestors])
    anonymous: dict[str, str] = {}
    kept: list[str] = []
    room = MAX_ANCESTORS_CHARS
    for node in ancestors:  # nearest first, so the oldest are the ones left out
        line = f"- {_label(adapter, node, anonymous)} ({_type(node)}): {_one_line(node.text)}"
        if len(line) > room:
            break
        room -= len(line)
        kept.append(line)
    lines = ["Earlier messages in this reply chain, oldest first:"]
    if not complete or len(kept) < len(ancestors):
        lines.append("- … earlier messages not shown")
    lines += reversed(kept)
    lines += [f"Directly quoted message ({_type(quoted)}, from {_label(adapter, quoted, anonymous)}):", quoted.text]
    return "\n".join(lines)


async def _prefetch_names(adapter: Any, nodes: list[QuoteNode]) -> None:
    from .feishu_names import prefetch_user_names

    open_ids = [
        str(sender.id) for sender in (node.sender for node in nodes)
        if getattr(sender, "id_type", "") == "open_id" and getattr(sender, "id", None)
    ]
    if open_ids:  # a lookup cut short leaves neutral labels; names it finds later are cached
        await _within(adapter, prefetch_user_names(adapter, open_ids), NAMES_BUDGET_SECONDS)


def _one_line(text: str) -> str:
    line = " ".join(text.split())
    return line if len(line) <= MAX_ANCESTOR_CHARS else line[:MAX_ANCESTOR_CHARS].rstrip() + "…"


def _is_own(adapter: Any, sender: Any) -> bool:
    """GET reports a bot message's sender as ``app`` with ``id`` = its app_id; peer bots differ on the id."""
    sender_id = str(getattr(sender, "id", "") or "")
    return getattr(sender, "sender_type", "") == "app" and bool(sender_id) and sender_id == adapter._app_id


def _label(adapter: Any, node: QuoteNode, anonymous: dict[str, str]) -> str:
    from .feishu_forward import sender_label

    return "You" if _is_own(adapter, node.sender) else sender_label(adapter, node.sender, anonymous)


def _type(node: QuoteNode) -> str:
    return _TYPE_LABELS.get(node.msg_type, node.msg_type or "message")
