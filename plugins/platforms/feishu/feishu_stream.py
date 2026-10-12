"""A Feishu reply streams into one CardKit card (upstream's stream-is-the-message draft contract).

The stream consumer calls ``send_draft`` with the whole text so far for each frame; the first frame of a
turn creates a streaming card and sends it as a reply, later frames replace the card's text. The turn's
final ``send()`` does not post a new message: it seals the open card with the final text (seal-
interception, like the relay adapter for Slack), and the sealed card IS the reply. ``abandon_open_draft``
seals an orphaned card in place when the turn dies (/stop, /new). Mid-turn sends that are not the final
carry ``metadata["_interim_send"]`` and never seal.

A card is keyed by the turn: the inbound message id the consumer passes as ``reply_to_message_id`` (the
same id the turn's final send replies to). Only a send for that exact turn seals the card. A sealed key is
tombstoned so a late frame for it never opens a second card.

Long replies and long-running turns: Feishu closes a card's streaming window after about ten minutes and
one markdown element holds about 30k characters. Before either limit the live card is sealed with what it
shows plus a "continued below" footer and the rest of the reply streams into a fresh card; the turn's
final seals whichever card is live, with only the text that card has not shown.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

from gateway.platforms.base import SendResult

logger = logging.getLogger(__name__)

# Cards older than this roll over before Feishu's ~600 s streaming window closes under them.
MAX_CARD_AGE_SECONDS = 540.0
# Roll over well before a markdown element's ~30k-character limit (headroom for the render pass).
MAX_CARD_CHARS = 24_000
_STATE_CAP = 512


@dataclass
class LiveCard:
    card_id: str
    message_id: str
    created_at: float
    sequence: int = 1
    shown: str = ""  # raw text the card currently shows (this card's part of the reply)
    plain_mentions: bool = False  # Feishu refused a mention in this card: render all as text
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence


@dataclass
class TurnStream:
    """One turn's reply: the live card plus the text earlier (rolled-over) cards already sealed."""

    key: str
    draft_id: int
    chat_id: str
    reply_to: Optional[str]
    metadata: Optional[dict[str, Any]]
    card: Optional[LiveCard] = None
    sealed_prefix: str = ""  # reply text shown by earlier cards of this turn
    first_message_id: Optional[str] = None


def stream_key(chat_id: str, metadata: Optional[dict[str, Any]]) -> str:
    md = metadata or {}
    turn = md.get("reply_to_message_id") or md.get("message_id")
    if turn:
        return f"{chat_id}:turn:{turn}"
    return f"{chat_id}:{md.get('thread_id') or ''}"


def merge_final(shown: str, final: str) -> str:
    """The text a sealed card shows: what the turn already streamed, completed by the final.

    The consumer's final is the turn's last assistant message, not the whole stream: a preamble streamed
    before a tool call ("Let me check the logs.") is not in it, and segments are joined with no separator.
    Replacing the card with the final would erase that preamble, so the final replaces only the tail of the
    card it restates (the longest suffix of ``shown`` the final starts with), and the rest of the card
    stays; a final the card does not show at all is appended."""
    shown, final = shown.rstrip(), final.strip()
    if not shown or not final:
        return shown or final
    if final in shown:
        return shown
    if final.startswith(shown):
        return final
    # The longest tail of the card that the final restates (the last segment, possibly extended).
    for start in range(len(shown)):
        tail = shown[start:].lstrip()
        if len(tail) >= 2 and final.startswith(tail):
            head = shown[:start].rstrip()
            return f"{head}\n\n{final}" if head else final
    return f"{shown}\n\n{final}"


def _remainder(full: str, prefix: str) -> str:
    """``full`` minus the head already shown by sealed cards. A card starts at the seam, so whitespace
    there is dropped (a card would not show it anyway); a ``full`` that does not extend the head is kept
    whole rather than guessed at."""
    if not prefix:
        return full
    for head in (prefix, prefix.rstrip()):
        if full.startswith(head):
            return full[len(head):].lstrip()
    return full


class CardStreams:
    """Per-adapter registry of open turn streams and sealed (tombstoned) keys."""

    def __init__(self) -> None:
        self.open: OrderedDict[str, TurnStream] = OrderedDict()
        self.sealed: OrderedDict[str, int] = OrderedDict()

    def _evict(self, mapping: OrderedDict) -> None:
        while len(mapping) > _STATE_CAP:
            mapping.popitem(last=False)

    def tombstone(self, key: str, draft_id: int) -> None:
        self.sealed[key] = draft_id
        self.sealed.move_to_end(key)
        self._evict(self.sealed)

    def match(self, chat_id: str, metadata: Optional[dict[str, Any]]) -> Optional[str]:
        """The open stream a send belongs to: the exact turn key only. A notice that names no turn never
        lands on an anchored turn's card (sealing someone's reply with an ack is not recoverable, a separate
        message is); an anchorless turn's own final, which carries no turn id either, still seals it."""
        key = stream_key(chat_id, metadata)
        return key if key in self.open else None


def _registry(adapter: Any) -> CardStreams:
    streams = getattr(adapter, "_card_streams", None)
    if streams is None:
        streams = adapter._card_streams = CardStreams()
    return streams


def _continued_footer() -> str:
    from agent.i18n import t

    return t("platform.feishu.stream.continued")


async def _write(adapter: Any, card: LiveCard, text: str) -> Any:
    """Stream ``text`` into ``card``; a refused mention switches the card to plain-text mentions once."""
    from . import feishu_cardkit as ck
    from .feishu_card_render import has_card_mentions

    result = await ck.stream_text(adapter, card.card_id, text, card.next_sequence(), plain_mentions=card.plain_mentions)
    if not result.ok and result.code == ck.INVALID_USER_RESOURCE and not card.plain_mentions and has_card_mentions(text):
        card.plain_mentions = True
        result = await ck.stream_text(adapter, card.card_id, text, card.next_sequence(), plain_mentions=True)
    return result


async def _seal(adapter: Any, card: LiveCard, text: str, *, footer: str = "") -> bool:
    """Replace ``card`` with its final body (streaming off). True only once Feishu accepted it."""
    from . import feishu_cardkit as ck
    from .feishu_card_render import has_card_mentions, strip_card_mentions

    def _body(plain: bool) -> dict:
        return ck.final_card_body(strip_card_mentions(text) if plain else text, footer=footer)

    result = await ck.replace_card(adapter, card.card_id, _body(card.plain_mentions), card.next_sequence())
    if not result.ok and result.code == ck.INVALID_USER_RESOURCE and has_card_mentions(text):
        card.plain_mentions = True
        result = await ck.replace_card(adapter, card.card_id, _body(True), card.next_sequence())
    if result.ok:
        card.shown = text
        return True
    # A full update failed: turn streaming off at least, so the card stops showing the loading icon.
    await ck.close_streaming(adapter, card.card_id, card.next_sequence())
    return False


async def _open_card(adapter: Any, stream: TurnStream, text: str) -> Optional[LiveCard]:
    """Create a card, send it as this turn's reply, and stream ``text`` into it."""
    from . import feishu_cardkit as ck

    card_id = await ck.create_card(adapter)
    if not card_id:
        return None
    card = LiveCard(card_id=card_id, message_id="", created_at=time.monotonic())
    try:
        response = await adapter._feishu_send_with_retry(
            chat_id=stream.chat_id, msg_type="interactive", payload=ck.card_message_content(card_id),
            reply_to=stream.reply_to, metadata=stream.metadata,
        )
    except (Exception, asyncio.CancelledError):
        # The message may still land after a cancelled await: seal the card so it never spins forever.
        await asyncio.shield(_seal(adapter, card, text))
        raise
    result = adapter._finalize_send_result(response, "streaming card send failed")
    if not result.success or not result.message_id:
        await _seal(adapter, card, text)
        return None
    card.message_id = result.message_id
    stream.first_message_id = stream.first_message_id or card.message_id
    if text:
        await _write(adapter, card, text)
        card.shown = text
    return card


async def _roll_over(adapter: Any, stream: TurnStream, *, footer: Optional[str] = None) -> bool:
    """Seal the live card with what it shows and a "continued" footer; the next frame opens a new card."""
    card = stream.card
    if card is None:
        return True
    if stream.card is not card:
        return True  # another task rolled it over while this one waited for the lock
    if not await _seal(adapter, card, card.shown, footer=_continued_footer() if footer is None else footer):
        return False
    stream.sealed_prefix += card.shown
    stream.card = None
    return True


def _needs_rollover(card: LiveCard, part: str) -> bool:
    return time.monotonic() - card.created_at >= MAX_CARD_AGE_SECONDS or len(part) > MAX_CARD_CHARS


async def send_draft(adapter: Any, chat_id: str, draft_id: int, content: str,
                     metadata: Optional[dict[str, Any]]) -> SendResult:
    """One frame of the turn's reply (the whole text so far)."""
    from . import feishu_cardkit as ck

    streams = _registry(adapter)
    key = stream_key(chat_id, metadata)
    if streams.sealed.get(key) == draft_id:
        return SendResult(success=True)  # a frame that arrived after the seal: already shown
    stream = streams.open.get(key)
    if stream is None or stream.draft_id != draft_id:
        md = dict(metadata or {})
        stream = TurnStream(key=key, draft_id=draft_id, chat_id=chat_id,
                            reply_to=md.get("reply_to_message_id"), metadata=md or None)
        streams.open[key] = stream
        streams._evict(streams.open)
    try:
        if stream.card is not None and _needs_rollover(stream.card, _remainder(content, stream.sealed_prefix)):
            async with stream.card.lock:
                await _roll_over(adapter, stream)
        part = _remainder(content, stream.sealed_prefix)
        if stream.card is None:
            card = await _open_card(adapter, stream, part)
            if card is None:
                if not stream.sealed_prefix:
                    streams.open.pop(key, None)  # nothing on screen: the consumer falls back to send/edit
                return SendResult(success=False, error="CardKit streaming card unavailable")
            stream.card = card
            return SendResult(success=True)
        async with stream.card.lock:
            if part == stream.card.shown:
                return SendResult(success=True)
            result = await _write(adapter, stream.card, part)
            if result.ok or result.code == ck.RATE_LIMITED:
                if result.ok:
                    stream.card.shown = part
                return SendResult(success=True)
            if not result.expired:
                return SendResult(success=False, error=f"CardKit stream failed: [{result.code}] {result.msg}")
            # The streaming window closed early: seal what is shown, continue in a fresh card.
            await _roll_over(adapter, stream)
        stream.card = await _open_card(adapter, stream, _remainder(content, stream.sealed_prefix))
        return SendResult(success=stream.card is not None)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("[Feishu] CardKit draft frame failed: %s", exc, exc_info=True)
        return SendResult(success=False, error=str(exc))


async def seal_with_final(adapter: Any, chat_id: str, content: str,
                          metadata: Optional[dict[str, Any]]) -> Optional[SendResult]:
    """Seal the turn's open card with the final text; None when no card is open (a plain send follows)."""
    streams = _registry(adapter)
    key = streams.match(chat_id, metadata)
    if key is None:
        return None
    stream = streams.open.pop(key)
    streams.tombstone(key, stream.draft_id)
    card = stream.card
    shown = card.shown if card else ""
    # The live card's final text: the reply minus what sealed cards show, completing what this card shows
    # (a final that only restates the turn's last segment must not erase an earlier one).
    part = merge_final(shown, _remainder(content, stream.sealed_prefix))
    if card is not None and len(part) > MAX_CARD_CHARS:
        # The final outgrew the live card: keep what it shows, continue the rest in a fresh card.
        async with card.lock:
            if not await _roll_over(adapter, stream):
                return None
        card = None
        part = _remainder(part, shown)
    if card is None:
        if not part.strip():
            return SendResult(success=True, message_id=stream.first_message_id)
        card = await _open_card(adapter, stream, part)
        if card is None:
            return None
    async with card.lock:
        if await _seal(adapter, card, part):
            return SendResult(success=True, message_id=card.message_id)
    logger.warning("[Feishu] Sealing the reply card failed; delivering the final as a plain message")
    return None


async def abandon(adapter: Any, chat_id: str, content: str, metadata: Optional[dict[str, Any]]) -> SendResult:
    """Seal an orphaned card in place with the text already on screen (/stop, /new, a superseded turn)."""
    streams = _registry(adapter)
    key = streams.match(chat_id, metadata)
    if key is None:
        return SendResult(success=True)
    stream = streams.open.pop(key)
    streams.tombstone(key, stream.draft_id)
    card = stream.card
    if card is None:
        return SendResult(success=True)
    async with card.lock:
        shown = card.shown or _remainder(content, stream.sealed_prefix)
        ok = await _seal(adapter, card, shown)
    return SendResult(success=ok, message_id=card.message_id, error=None if ok else "CardKit abandon seal failed")


async def seal_for_prompt(adapter: Any, chat_id: str) -> None:
    """Before an approval / clarify card: seal every card streaming in ``chat_id`` with what it shows.

    A prompt blocks its turn's agent, so whatever streams in the chat now belongs above the prompt; the
    turn keeps its stream, and the next frame after the answer opens a fresh card below the prompt (else
    post-answer text would keep growing the card above it). A prompt carries no turn id in a plain chat,
    so the chat is the unit.
    """
    streams = _registry(adapter)
    for stream in [s for s in streams.open.values() if s.chat_id == chat_id and s.card is not None]:
        async with stream.card.lock:
            await _roll_over(adapter, stream, footer="")


async def seal_quiet_cards(adapter: Any) -> None:
    """Seal a card about to outlive Feishu's streaming window while no frame arrives (a long tool call):
    a frame-driven rollover only happens on the next frame, and an expired card keeps its loading icon."""
    streams = _registry(adapter)
    for stream in [s for s in streams.open.values() if s.card is not None]:
        card = stream.card
        if time.monotonic() - card.created_at >= MAX_CARD_AGE_SECONDS and not card.lock.locked():
            async with card.lock:
                await _roll_over(adapter, stream)


QUIET_CHECK_SECONDS = 30.0


async def watch_quiet_cards(adapter: Any) -> None:
    """Background check (the adapter's connection lifetime) for cards outliving their streaming window."""
    while True:
        await asyncio.sleep(QUIET_CHECK_SECONDS)
        try:
            await seal_quiet_cards(adapter)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("[Feishu] Quiet-card check failed", exc_info=True)
