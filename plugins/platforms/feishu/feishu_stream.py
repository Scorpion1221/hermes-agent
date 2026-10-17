"""A Feishu reply streams into one CardKit card (upstream's stream-is-the-message draft contract).

The stream consumer calls ``send_draft`` with the whole text so far for each frame; the first frame of a
turn creates a streaming card and sends it as a reply, later frames replace the card's text. The turn's
final ``send()`` does not post a new message: it seals the open card with the final text (seal-
interception, like the relay adapter for Slack), and the sealed card IS the reply. A seal runs to the end
even when its sender is cancelled, and the gateway's resend of that same final is answered by it.
``abandon_open_draft`` seals an orphaned card in place when the turn dies (/stop, /new). Mid-turn sends
that are not the final carry ``metadata["_interim_send"]`` and never seal.

A card is keyed by the consumer's per-turn draft id. Only a turn-final send (marked ``notify``, as the
consumer's final and the gateway's final lane are) replying to the same message seals it; status bubbles,
progress, busy acks and command replies carry no ``notify`` and never seal. A sealed key is tombstoned so a
late frame for it never opens a second card.

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
from gateway.stream_consumer_fences import ensure_closed_code_fences

logger = logging.getLogger(__name__)

# Cards older than this roll over before Feishu's ~600 s streaming window closes under them.
# Reaction on the turn's final reply card.
COMPLETE = "DONE"
MAX_CARD_AGE_SECONDS = 540.0
# Roll over well before a markdown element's ~30k-character limit (headroom for the render pass).
MAX_CARD_CHARS = 24_000
# How long the gateway's resend of a final waits for the orphaned seal of that final (see ``_seal_final``).
ORPHAN_WAIT_SECONDS = 30.0
_STATE_CAP = 512


def _now() -> float:
    """Card ages and the elapsed footer (a seam tests drive without slowing asyncio's own clock)."""
    return time.monotonic()


@dataclass
class LiveCard:
    card_id: str
    message_id: str
    created_at: float
    sequence: int = 1
    shown: str = ""  # raw text the card currently shows (this card's part of the reply)
    plain_mentions: bool = False  # Feishu refused a mention in this card: render all as text

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
    started_at: Optional[float] = None  # monotonic time the turn's first card opened (the elapsed footer)
    # Serializes every card operation of the turn (frames, seal, rollover, prompt and quiet seals), so no
    # operation ever sees a card another one swapped out mid-await.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def stream_key(chat_id: str, draft_id: int) -> str:
    """One turn's stream: the consumer allocates a fresh draft id per turn (two turns that reply to the
    same topic root still get two keys)."""
    return f"{chat_id}:draft:{draft_id}"


def _squash(text: str) -> str:
    return "".join(text.split())


def merge_final(shown: str, final: str) -> str:
    """The text a sealed card shows: what the turn already streamed, completed by the final.

    The consumer's final is the turn's last assistant message, not the whole stream: a preamble streamed
    before a tool call ("Let me check the logs.") is not in it, segments are joined with no separator, and
    rolled-over cards are joined to the live one at a seam. Whitespace is ignored when comparing. The final
    replaces only the tail of what is shown that it restates (the longest such suffix); the rest stays, and
    a final the card does not show at all is appended."""
    shown, final = shown.rstrip(), final.strip()
    if not shown or not final:
        return shown or final
    squashed_shown, squashed_final = _squash(shown), _squash(final)
    if squashed_final in squashed_shown:
        return shown
    if squashed_final.startswith(squashed_shown):
        return final
    for start in range(len(shown)):
        tail = _squash(shown[start:])
        if len(tail) >= 2 and squashed_final.startswith(tail):
            head = shown[:start].rstrip()
            return f"{head}\n\n{final}" if head else final
    return f"{shown}\n\n{final}"


def _remainder(full: str, prefix: str) -> str:
    """``full`` minus the head already shown by sealed cards. A card starts at the seam, so whitespace
    there is dropped (a card would not show it anyway); ``full`` that does not extend the head (compared
    ignoring whitespace) is kept whole rather than guessed at."""
    if not prefix:
        return full
    target = _squash(prefix)
    seen = 0
    for index, char in enumerate(full):
        if not char.isspace():
            if seen == len(target):
                return full[index:]
            if char != target[seen]:
                return full
            seen += 1
    return "" if seen == len(target) else full


class CardStreams:
    """Per-adapter registry of open turn streams and sealed (tombstoned) keys."""

    def __init__(self) -> None:
        self.open: OrderedDict[str, TurnStream] = OrderedDict()
        self.sealed: OrderedDict[str, int] = OrderedDict()
        # (chat, reply anchor) -> (seal task, final text) of a final seal whose sender was cancelled.
        self.orphaned: OrderedDict[str, tuple[asyncio.Future, str]] = OrderedDict()

    def _evict(self, mapping: OrderedDict) -> None:
        while len(mapping) > _STATE_CAP:
            mapping.popitem(last=False)

    def tombstone(self, key: str, draft_id: int) -> None:
        self.sealed[key] = draft_id
        self.sealed.move_to_end(key)
        self._evict(self.sealed)

    def for_turn(self, chat_id: str, metadata: Optional[dict[str, Any]]) -> Optional[str]:
        """The newest open stream in ``chat_id`` replying to the same message as ``metadata``."""
        md = metadata or {}
        anchor = md.get("reply_to_message_id") or md.get("message_id")
        candidates = [k for k, s in self.open.items() if s.chat_id == chat_id and s.reply_to == anchor]
        return candidates[-1] if candidates else None

    def match(self, chat_id: str, metadata: Optional[dict[str, Any]]) -> Optional[str]:
        """The open stream a turn-final send belongs to.

        Only a send marked ``notify`` (the gateway's turn-final marker: the consumer's final, the normal
        final lane) seals a card; status bubbles, progress, busy acks and command replies never carry it.
        Among the chat's open streams the one replying to the same message wins; with several (two turns
        in one topic reply to the same root) the newest does, never an older turn's card."""
        if (metadata or {}).get("notify") is not True:
            return None
        return self.for_turn(chat_id, metadata)

    @staticmethod
    def _anchor_key(chat_id: str, metadata: Optional[dict[str, Any]]) -> str:
        md = metadata or {}
        return f"{chat_id}:{md.get('reply_to_message_id') or md.get('message_id')}"

    def orphan(self, chat_id: str, metadata: Optional[dict[str, Any]], seal: asyncio.Future, content: str) -> None:
        key = self._anchor_key(chat_id, metadata)
        self.orphaned[key] = (seal, content)
        self.orphaned.move_to_end(key)
        self._evict(self.orphaned)

    def take_orphan(self, chat_id: str, content: str, metadata: Optional[dict[str, Any]]) -> Optional[asyncio.Future]:
        """The orphaned final seal a gateway-lane final for the same message is the resend of. The first such
        final consumes the orphan whatever its text; the seal is returned only when that final is the text it
        sealed (compared ignoring whitespace and an unclosed fence the seal closed)."""
        if (metadata or {}).get("notify") is not True:
            return None
        seal, sealed = self.orphaned.pop(self._anchor_key(chat_id, metadata), (None, ""))
        if seal is None or _squash(ensure_closed_code_fences(content)) != _squash(ensure_closed_code_fences(sealed)):
            return None
        return seal


def _registry(adapter: Any) -> CardStreams:
    streams = getattr(adapter, "_card_streams", None)
    if streams is None:
        streams = adapter._card_streams = CardStreams()
    return streams


def _continued_footer() -> str:
    from agent.i18n import t

    return t("platform.feishu.stream.continued")


def _elapsed_footer(stream: TurnStream, *, stopped: bool = False) -> str:
    """"Done · 12.3s" / "Stopped · 1m 5s", timed from the turn's first card (a continuation card
    reports the whole reply)."""
    from agent.i18n import t

    if stream.started_at is None:
        return ""
    seconds = max(0.0, _now() - stream.started_at)
    elapsed = f"{int(seconds // 60)}m {int(seconds % 60)}s" if seconds >= 60 else f"{seconds:.1f}s"
    return t("platform.feishu.stream.stopped" if stopped else "platform.feishu.stream.done", elapsed=elapsed)


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
    card = LiveCard(card_id=card_id, message_id="", created_at=_now())
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
    if stream.started_at is None:
        stream.started_at = card.created_at
    if text:
        await _write(adapter, card, text)
        card.shown = text
    return card


async def _roll_over(adapter: Any, stream: TurnStream, *, footer: Optional[str] = None) -> bool:
    """Seal the live card with what it shows and a "continued" footer; the next frame opens a new card.
    The caller holds ``stream.lock``."""
    card = stream.card
    if card is None:
        return True
    if not await _seal(adapter, card, card.shown, footer=_continued_footer() if footer is None else footer):
        return False
    stream.sealed_prefix += card.shown
    stream.card = None
    return True


def _needs_rollover(card: LiveCard, part: str) -> bool:
    return _now() - card.created_at >= MAX_CARD_AGE_SECONDS or len(part) > MAX_CARD_CHARS


# Failures a later frame (or the seal, which carries the whole text) repairs: not a reason to give up
# streaming for the rest of the turn. A refused card or a closed window are handled separately.
def _transient(code: Optional[int]) -> bool:
    return code is None or code in {230020, 300317} or code >= 500 or 1_000_000 <= code


async def send_draft(adapter: Any, chat_id: str, draft_id: int, content: str,
                     metadata: Optional[dict[str, Any]]) -> SendResult:
    """One frame of the turn's reply (the whole text so far)."""
    streams = _registry(adapter)
    key = stream_key(chat_id, draft_id)
    if key in streams.sealed:
        return SendResult(success=True)  # a frame that arrived after the seal: already shown
    stream = streams.open.get(key)
    if stream is None:
        md = dict(metadata or {})
        stream = TurnStream(key=key, draft_id=draft_id, chat_id=chat_id,
                            reply_to=md.get("reply_to_message_id"), metadata=md or None)
        streams.open[key] = stream
        streams._evict(streams.open)
    try:
        async with stream.lock:
            if key in streams.sealed:
                return SendResult(success=True)  # sealed while this frame waited
            return await _frame(adapter, streams, stream, content)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("[Feishu] CardKit draft frame failed: %s", exc, exc_info=True)
        return SendResult(success=True) if stream.card is not None else SendResult(success=False, error=str(exc))


async def _frame(adapter: Any, streams: CardStreams, stream: TurnStream, content: str) -> SendResult:
    if stream.card is not None and _needs_rollover(stream.card, _remainder(content, stream.sealed_prefix)):
        await _roll_over(adapter, stream)
    part = _remainder(content, stream.sealed_prefix)
    if stream.card is None:
        stream.card = await _open_card(adapter, stream, part)
        if stream.card is not None:
            return SendResult(success=True)
        if stream.sealed_prefix:
            # Earlier cards of this turn are on screen: the seal opens the last card, never a resend of
            # the whole reply (the consumer's fallback would repeat what sealed cards show).
            return SendResult(success=True)
        streams.open.pop(stream.key, None)  # nothing on screen: the consumer falls back to send/edit
        return SendResult(success=False, error="CardKit streaming card unavailable")
    if part == stream.card.shown:
        return SendResult(success=True)
    result = await _write(adapter, stream.card, part)
    if result.ok:
        stream.card.shown = part
        return SendResult(success=True)
    if result.expired:
        # The streaming window closed early: seal what is shown, continue in a fresh card.
        await _roll_over(adapter, stream)
        stream.card = await _open_card(adapter, stream, _remainder(content, stream.sealed_prefix))
        return SendResult(success=True)
    if _transient(result.code):
        return SendResult(success=True)  # the next frame, or the seal, carries the whole text again
    return SendResult(success=False, error=f"CardKit stream failed: [{result.code}] {result.msg}")


async def seal_with_final(adapter: Any, chat_id: str, content: str, metadata: Optional[dict[str, Any]],
                          *, stream_final: bool = False) -> Optional[SendResult]:
    """Seal the turn's open card with the final text; None when no card is open (a plain send follows).
    ``stream_final``: the stream consumer's own final (its metadata names the user's message)."""
    streams = _registry(adapter)
    # The gateway's resend of a final whose seal was cancelled, checked before the open streams: in a topic a
    # later turn's card replies to the same root and must not be sealed with this turn's final. Only the
    # gateway's lane: a consumer's own final (``stream_final``) is never a resend.
    if not stream_final and (resent := await _resent_final(streams, chat_id, content, metadata)) is not None:
        return resent
    key = streams.match(chat_id, metadata)
    if key is None:
        md = dict(metadata or {})
        if not stream_final or md.get("notify") is not True or not adapter.supports_draft_streaming():
            return None
        # The turn's reply arrived in one piece and never opened a card: give it the card a streamed reply
        # ends as (footer, DONE). Command replies and notices are not the consumer's final: plain cards.
        stream = TurnStream(key="", draft_id=-1, chat_id=chat_id, reply_to=md.get("reply_to_message_id"),
                            metadata=md, started_at=_now())
        return await _seal_final(adapter, streams, stream, content, metadata)
    stream = streams.open.pop(key)
    streams.tombstone(key, stream.draft_id)
    return await _seal_final(adapter, streams, stream, content, metadata)


async def _seal_final(adapter: Any, streams: CardStreams, stream: TurnStream, content: str,
                      metadata: Optional[dict[str, Any]]) -> Optional[SendResult]:
    """Seal the turn's cards with its final, to the end even when the sender stops waiting.

    Feishu applies a card update it received whether or not its answer is still awaited, and the gateway
    cancels a stream consumer that takes over 5 s to finish. A cancelled sender learns nothing, so the gateway
    sends the same final again: the seal is kept for that resend (``_resent_final``), which then answers with
    the card that already shows the final instead of opening a second one."""
    async def _locked() -> Optional[SendResult]:
        async with stream.lock:
            return await _seal_turn(adapter, stream, content)

    seal = asyncio.ensure_future(_locked())
    try:
        return await asyncio.shield(seal)
    except asyncio.CancelledError:
        seal.add_done_callback(_log_orphan_failure)
        streams.orphan(stream.chat_id, metadata, seal, content)
        raise


def _log_orphan_failure(seal: asyncio.Future) -> None:
    if not seal.cancelled() and seal.exception() is not None:
        logger.warning("[Feishu] A final card seal failed after its sender stopped waiting",
                       exc_info=seal.exception())


async def _resent_final(streams: CardStreams, chat_id: str, content: str,
                        metadata: Optional[dict[str, Any]]) -> Optional[SendResult]:
    """The orphaned seal's result when ``content`` is the final it sealed (the gateway resending it); None
    otherwise, and when the seal failed or is still unanswered after ``ORPHAN_WAIT_SECONDS`` (the resend is
    then delivered as usual: a possible duplicate, never a lost answer)."""
    seal = streams.take_orphan(chat_id, content, metadata)
    if seal is None:
        return None
    try:
        result = await asyncio.wait_for(asyncio.shield(seal), ORPHAN_WAIT_SECONDS)
    except Exception:  # health: allow BLE001 -- a timeout, or the seal's failure (logged by its callback)
        return None
    return result if result is not None and result.success else None


async def _seal_turn(adapter: Any, stream: TurnStream, content: str) -> Optional[SendResult]:
    card = stream.card
    shown = card.shown if card else ""
    # Everything the turn shows (sealed cards + the live one), completed by the final; the live card ends
    # with what sealed cards do not already show. A final that only restates the last segment keeps the
    # earlier ones, and one that straddles a rollover is not repeated.
    whole = merge_final(stream.sealed_prefix + shown, content)
    part = _remainder(whole, stream.sealed_prefix)
    if card is not None and len(part) > MAX_CARD_CHARS:
        # The final outgrew the live card: keep what it shows, continue the rest in a fresh card.
        if not await _roll_over(adapter, stream):
            return None
        card = None
        part = _remainder(whole, stream.sealed_prefix)
    if card is None:
        if not part.strip():
            return SendResult(success=True, message_id=stream.first_message_id)
        card = await _open_card(adapter, stream, part)
        if card is None:
            if stream.sealed_prefix:
                # Earlier cards show the head: post only what they do not, never the whole reply again.
                return await _plain_tail(adapter, stream, part)
            return None
    if await _seal(adapter, card, part, footer=_elapsed_footer(stream)):
        await _mark_complete(adapter, card.message_id)
        return SendResult(success=True, message_id=card.message_id)
    logger.warning("[Feishu] Sealing the reply card failed; delivering the final as a plain message")
    return None


async def _mark_complete(adapter: Any, message_id: str) -> None:
    """The reply card is final: a DONE reaction says so, since a long stream gives no other signal it
    ended. Only the turn's last card gets it, never a card sealed for a prompt or a rollover."""
    if not adapter._reactions_enabled():
        return
    try:
        await adapter._add_reaction(message_id, COMPLETE)
    except Exception:  # the card is already sealed: a missing reaction must never undo that
        logger.debug("[Feishu] DONE reaction on %s failed", message_id, exc_info=True)


async def _plain_tail(adapter: Any, stream: TurnStream, part: str) -> SendResult:
    from .feishu_outbound import send_message

    return await send_message(adapter, stream.chat_id, part, reply_to=stream.reply_to, metadata=stream.metadata)


async def abandon(adapter: Any, chat_id: str, content: str, metadata: Optional[dict[str, Any]]) -> SendResult:
    """Seal an orphaned card in place with the text already on screen (/stop, /new, a superseded turn)."""
    streams = _registry(adapter)
    key = streams.for_turn(chat_id, metadata)
    if key is None:
        return SendResult(success=True)
    stream = streams.open.pop(key)
    streams.tombstone(key, stream.draft_id)
    async with stream.lock:
        card = stream.card
        if card is None:
            return SendResult(success=True)
        shown = card.shown or _remainder(content, stream.sealed_prefix)
        ok = await _seal(adapter, card, shown, footer=_elapsed_footer(stream, stopped=True))
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
        async with stream.lock:
            await _roll_over(adapter, stream, footer="")


async def seal_quiet_cards(adapter: Any) -> None:
    """Seal a card about to outlive Feishu's streaming window while no frame arrives (a long tool call):
    a frame-driven rollover only happens on the next frame, and an expired card keeps its loading icon."""
    streams = _registry(adapter)
    for stream in [s for s in streams.open.values() if s.card is not None]:
        if stream.lock.locked():
            continue  # a frame or seal is working on it; that path rolls an old card over itself
        async with stream.lock:
            if stream.card is not None and _now() - stream.card.created_at >= MAX_CARD_AGE_SECONDS:
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
