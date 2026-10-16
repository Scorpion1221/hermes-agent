"""Clicks on this bot's Feishu cards (other than its own approval / update prompts).

A click answers the card, once. Only an admin, the user the card answers or, in a DM, the user there may
click; anyone else gets an error toast and the card is left as it was. A card answers the user whose turn was
running in its chat (and topic) when it was sent, however it was sent (the adapter or the agent's own
``lark-cli``): the processing hooks record each turn (:func:`note_turn`). A card no recorded turn explains
(sent before a restart) answers the sender of the message it replies to, through the bot's own replies but
never a topic's root; a card that answers nobody is for admins. An allowed click gets a toast at once and
reaches the agent as ``card:<tag> <json>`` (what the user did), from the clicker, quoting the card, in the
card's chat and topic; then the card is redrawn with its controls disabled and who handled it. A later click
on the same card only gets a toast.

The callback names the card (``open_message_id``) but not its topic, so the card is fetched for its
``thread_id``; when that fails the click still reaches the agent, outside any topic. Its ``message_id``
names the click, ``<card>:card:<token>``, for the same reason a reaction's does (see :mod:`feishu_reaction`).

Feishu wants the callback answered within about 3 s. Over the websocket the callback runs on the SDK's own
thread, so deciding who may click can wait a bounded time for the lookups on the adapter loop; one that
overruns asks the user to click again and warms the cache for that click. The webhook transport calls back
on the adapter loop and ignores the response, so there the whole decision runs afterwards.
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import logging
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

from agent.i18n import t
from gateway.platforms.event import MessageType

from .feishu_card_operator import _running_on
from .feishu_reaction import _event_chat_type

logger = logging.getLogger(__name__)

_CARD_ID_MARK = ":card:"
_CACHE_SIZE = 1024
_OWNER_HOPS = 4  # how far up the bot's own replies a card no turn explains looks for its asker
_RETRY_AFTER_FAILED_LOOKUP_SECONDS = 30.0
_CLOCK_SLACK_SECONDS = 5.0  # Feishu's create_time against this host's clock
_LOOKUP_BUDGET_SECONDS = 2.0  # leaves headroom inside Feishu's ~3 s card-callback deadline
# What the user did: a button's value, a submitted form's values, a select / picker choice, a typed input.
_ACTION_FIELDS = ("name", "value", "form_value", "option", "options", "input_value", "checked")
# Card JSON 2.0 controls that call back and can be disabled (``overflow`` cannot, so it is removed).
_CONTROLS = frozenset({
    "button", "checker", "date_picker", "input", "interactive_container", "multi_select_person",
    "multi_select_static", "picker_datetime", "picker_time", "select_img", "select_person", "select_static",
})


@dataclass(frozen=True)
class _Click:
    chat_id: str
    card_id: str
    open_id: str
    user_id: str
    union_id: str
    operator_ids: frozenset[str]
    token: str


@dataclass(frozen=True)
class _Audience:
    """Who a card answers: everyone in a DM, else ``owner`` (the ids of one user; empty: nobody in particular)."""
    dm: bool
    owner: frozenset[str] = frozenset()


@dataclass
class _Turn:
    chat_id: str
    thread_id: str
    user_ids: frozenset[str]
    started: float
    ended: Optional[float] = None


def on_card_click(adapter: Any, data: Any, loop: Any) -> Any:
    """The synchronous callback answer for a click (a toast); the rest of the work is scheduled on ``loop``."""
    click = _parse(data)
    if click is None:
        logger.debug("[Feishu] Card action missing chat_id or operator open_id, dropping")
        return adapter._card_response()
    if click.token and adapter._is_card_action_duplicate(click.token):
        logger.debug("[Feishu] Dropping duplicate card action token: %s", click.token)
        return adapter._card_response()
    if _handled(adapter, click.card_id):
        return adapter._card_response(toast=("info", t("platform.feishu.card_action.already_handled")))
    waits = not _running_on(loop)
    allowed = _may_click_now(adapter, click, loop) if waits else _may_click_cached(adapter, click)
    if allowed is False:
        logger.warning("[Feishu] Unauthorized card click by %s on card %s", click.open_id, click.card_id or "-")
        return adapter._card_response(toast=("error", t("platform.feishu.card_action.denied")))
    if allowed is None and waits:
        return adapter._card_response(toast=("warning", t("platform.feishu.card_action.retry")))
    if allowed:
        _mark_handled(adapter, click.card_id)
    if not adapter._submit_on_loop(loop, _answer(adapter, data, click, decided=bool(allowed))):
        _handled_cards(adapter).pop(click.card_id, None)
        return adapter._card_response()
    return adapter._card_response(toast=("success", t("platform.feishu.card_action.sent")) if allowed else None)


async def _answer(adapter: Any, data: Any, click: _Click, *, decided: bool) -> None:
    if not decided:  # the webhook transport: nobody waited for the decision
        audience = await _audience(adapter, click)
        if _handled(adapter, click.card_id) or audience is None or not _may_click(adapter, click, audience):
            logger.warning("[Feishu] Card click by %s on card %s not answered", click.open_id, click.card_id or "-")
            return
        _mark_handled(adapter, click.card_id)
    await adapter._handle_card_action_event(data)
    await _mark_card_answered(adapter, click)


def _parse(data: Any) -> Optional[_Click]:
    event = getattr(data, "event", None)
    context = getattr(event, "context", None)
    operator = getattr(event, "operator", None)
    chat_id = str(getattr(context, "open_chat_id", "") or "")
    open_id = str(getattr(operator, "open_id", "") or "")
    if not chat_id or not open_id:
        return None
    user_id = str(getattr(operator, "user_id", "") or "")
    union_id = str(getattr(operator, "union_id", "") or "")
    return _Click(chat_id=chat_id, card_id=str(getattr(context, "open_message_id", "") or ""), open_id=open_id,
                  user_id=user_id, union_id=union_id, operator_ids=frozenset(i for i in (open_id, user_id, union_id) if i),
                  token=str(getattr(event, "token", "") or ""))


def callback_response(card_data: Optional[dict[str, Any]], toast: Optional[tuple[str, str]]) -> Any:
    """``FeishuAdapter._card_response``: the SDK response, carrying an inline card and / or a toast."""
    from . import adapter as facade  # the lazily bound SDK names live on the facade

    if facade.P2CardActionTriggerResponse is None:
        return None
    response = facade.P2CardActionTriggerResponse()
    if card_data is not None and facade.CallBackCard is not None:
        response.card = facade.CallBackCard()
        response.card.type, response.card.data = "raw", card_data
    if toast is not None and facade.CallBackToast is not None:
        response.toast = facade.CallBackToast()
        response.toast.type, response.toast.content = toast
    return response


# --- Who may click ---

def _may_click(adapter: Any, click: _Click, audience: _Audience) -> bool:
    if click.operator_ids & adapter._admins or audience.dm:
        return True
    if audience.owner:
        return bool(audience.owner & click.operator_ids)
    # A card that answers nobody in particular: admins only, or whoever may prompt the bot when none are set.
    return not adapter._admins and adapter._is_interactive_operator_authorized(click.open_id)


def _may_click_cached(adapter: Any, click: _Click) -> Optional[bool]:
    if click.operator_ids & adapter._admins:
        return True
    audience = _audiences(adapter).get((click.chat_id, click.card_id))
    return _may_click(adapter, click, audience) if isinstance(audience, _Audience) else None


def _may_click_now(adapter: Any, click: _Click, loop: Any) -> Optional[bool]:
    """``_may_click`` from the SDK thread, waiting a bounded time for the lookups; None when they overran."""
    known = _may_click_cached(adapter, click)
    if known is not None:
        return known
    from agent.async_utils import safe_schedule_threadsafe

    future = safe_schedule_threadsafe(
        _audience(adapter, click), loop, logger=logger,
        log_message="[Feishu] Could not schedule the card audience lookup", log_level=logging.WARNING,
    )
    if future is None:
        return None
    try:
        audience = future.result(timeout=_LOOKUP_BUDGET_SECONDS)
    except TimeoutError:
        logger.info("[Feishu] Card audience lookup for %s took over %.1fs", click.card_id, _LOOKUP_BUDGET_SECONDS)
        return None
    except concurrent.futures.CancelledError:
        return None
    return None if audience is None else _may_click(adapter, click, audience)


async def _audience(adapter: Any, click: _Click) -> Optional[_Audience]:
    """Who the clicked card answers (cached per card); None when the lookups failed (retried after a pause)."""
    import time

    key = (click.chat_id, click.card_id)
    cache = _audiences(adapter)
    cached = cache.get(key)
    if isinstance(cached, _Audience) or (cached is not None and time.monotonic() < cached):
        return cached if isinstance(cached, _Audience) else None
    audience = await _find_audience(adapter, click)
    cache[key] = audience if audience is not None else time.monotonic() + _RETRY_AFTER_FAILED_LOOKUP_SECONDS
    _trim(cache)
    return audience


async def _find_audience(adapter: Any, click: _Click) -> Optional[_Audience]:
    chat_info = await adapter.get_chat_info(click.chat_id)
    if chat_info.get("chat_mode") == "p2p":  # an unreadable chat is judged by its card
        return _Audience(dm=True)
    if not click.card_id:
        return _Audience(dm=False)
    card = await _card_message(adapter, click.card_id)
    if card is None:
        return None
    owner = _turn_owner(adapter, click.chat_id, card)
    if owner is None:
        owner = await _asker(adapter, card)
    return None if owner is None else _Audience(dm=False, owner=owner)


def _turn_owner(adapter: Any, chat_id: str, card: Any) -> Optional[frozenset[str]]:
    """The user whose turn was running in the card's chat and topic when it was sent; None when no turn, or
    turns of several users, explain it."""
    import time

    try:
        created = int(str(getattr(card, "create_time", "") or "")) / 1000
    except ValueError:
        return None
    thread_id = str(getattr(card, "thread_id", "") or "")
    now = time.time()
    owners = {turn.user_ids for turn in _turns(adapter).values()
              if turn.chat_id == chat_id and turn.thread_id == thread_id
              and turn.started - _CLOCK_SLACK_SECONDS <= created <= (turn.ended or now) + _CLOCK_SLACK_SECONDS}
    return owners.pop() if len(owners) == 1 else None


async def _asker(adapter: Any, card: Any) -> Optional[frozenset[str]]:
    """The sender of the message ``card`` replies to, skipping the bot's own messages (a card answering a
    click or a reaction replies to the bot's earlier card). In a topic every message replies to the topic's
    root, which says nothing about who asked: nobody in particular. None when a lookup failed."""
    from .feishu_quote import fetch_node

    topic_root = str(getattr(card, "root_id", "") or "") if getattr(card, "thread_id", None) else ""
    parent_id = str(getattr(card, "parent_id", "") or "")
    for _ in range(_OWNER_HOPS):
        if not parent_id or parent_id == topic_root:
            break
        node = await fetch_node(adapter, parent_id)
        if node is None:
            return None
        if str(getattr(node.sender, "sender_type", "") or "") == "user":
            return frozenset({str(getattr(node.sender, "id", "") or "")} - {""})
        parent_id = node.parent_id
    return frozenset()


def note_turn(adapter: Any, event: Any, *, running: bool) -> None:
    """A turn for ``event`` started (``running``) or ended: the cards it sends answer its user."""
    import time

    turns = _turns(adapter)
    key = str(getattr(event, "message_id", "") or id(event))
    if not running:
        if key in turns:
            turns[key].ended = time.time()
        return
    source = getattr(event, "source", None)
    ids = frozenset(i for i in (getattr(source, "user_id", None), getattr(source, "user_id_alt", None),
                                *(getattr(source, "auth_aliases", None) or ())) if i)
    if ids:
        turns[key] = _Turn(chat_id=str(getattr(source, "chat_id", "") or ""),
                           thread_id=str(getattr(source, "thread_id", "") or ""), user_ids=ids, started=time.time())
        _trim(turns)


# --- The answered card ---

async def _mark_card_answered(adapter: Any, click: _Click) -> None:
    """Redraw the card with its controls disabled and who handled it (best effort: the click still routes)."""
    from .feishu_outbound import _patch_request

    if not click.card_id or not adapter._client:
        return
    try:
        card = await _card_definition(adapter, click.card_id)
        if card is None:
            return
        name = await adapter._resolve_sender_name_from_api(click.user_id or click.open_id) or click.open_id
        content = json.dumps(answered_card(card, t("platform.feishu.card_action.handled_by", user=name)),
                             ensure_ascii=False)
        response = await adapter._run_blocking(adapter._client.im.v1.message.patch,
                                               _patch_request(click.card_id, content))
        if not adapter._response_succeeded(response):
            logger.warning("[Feishu] Could not mark card %s answered: [%s] %s", click.card_id,
                           getattr(response, "code", None), getattr(response, "msg", None))
    except Exception:
        logger.warning("[Feishu] Could not mark card %s answered", click.card_id, exc_info=True)


async def _card_definition(adapter: Any, card_id: str) -> Optional[dict[str, Any]]:
    """The card as its sender wrote it (``user_card_content``), the form a PATCH takes back."""
    from . import adapter as facade

    request = facade._sdk_build(facade.GetMessageRequest, message_id=card_id,
                                card_msg_content_type="user_card_content")
    response = await adapter._run_blocking(adapter._client.im.v1.message.get, request)
    items = getattr(getattr(response, "data", None), "items", None) if adapter._response_succeeded(response) else None
    content = getattr(getattr(items[0], "body", None), "content", "") if items else ""
    card = json.loads(content) if content else None
    return card if isinstance(card, dict) else None


def answered_card(card: dict[str, Any], note: str) -> dict[str, Any]:
    """``card`` with every control that calls back disabled and ``note`` below it, so it takes no second answer."""
    card = copy.deepcopy(card)
    if str(card.get("schema")) == "2.0":
        body = card.setdefault("body", {})
        _disable(body)
        body.setdefault("elements", []).append({"tag": "markdown", "content": f"<font color='grey'>{note}</font>"})
    else:  # Card JSON 1.0 has no disabled state: its button rows go
        elements = [e for e in card.get("elements") or [] if not (isinstance(e, dict) and e.get("tag") == "action")]
        card["elements"] = [*elements, {"tag": "markdown", "content": note}]
    return card


def _disable(node: Any) -> None:
    if isinstance(node, list):
        node[:] = [n for n in node if not (isinstance(n, dict) and n.get("tag") == "overflow")]
        for child in node:
            _disable(child)
    elif isinstance(node, dict):
        # A link button never calls back: it stays usable.
        if node.get("tag") in _CONTROLS and not (node.get("tag") == "button" and (node.get("url") or node.get("multi_url"))):
            node["disabled"] = True
        for child in node.values():
            _disable(child)


# --- Per-adapter state ---

def _handled_cards(adapter: Any) -> OrderedDict:
    return vars(adapter).setdefault("_feishu_handled_cards", OrderedDict())


def _handled(adapter: Any, card_id: str) -> bool:
    return bool(card_id) and card_id in _handled_cards(adapter)


def _mark_handled(adapter: Any, card_id: str) -> None:
    if card_id:
        cards = _handled_cards(adapter)
        cards[card_id] = True
        _trim(cards)


def _audiences(adapter: Any) -> OrderedDict:
    """(chat, card) → its ``_Audience``, or the monotonic time a failed lookup may be retried."""
    return vars(adapter).setdefault("_feishu_card_audiences", OrderedDict())


def _turns(adapter: Any) -> OrderedDict:
    """inbound message id → the ``_Turn`` it started, newest last."""
    return vars(adapter).setdefault("_feishu_turns", OrderedDict())


def _trim(cache: OrderedDict) -> None:
    while len(cache) > _CACHE_SIZE:
        cache.popitem(last=False)


# --- The click, routed to the agent ---

async def handle_card_action_event(adapter: Any, data: Any) -> None:
    click = _parse(data)
    if click is None:
        return
    action = getattr(getattr(data, "event", None), "action", None)
    tag = str(getattr(action, "tag", "") or "button")
    card = await _card_message(adapter, click.card_id)
    quote = await adapter._fetch_quoted_reply(click.card_id, media=False)  # cached by the fetch above
    chat_info = await adapter.get_chat_info(click.chat_id)  # cached; reused by _dispatch_synthetic_event
    logger.info("[Feishu] Routing card action %r from %s on card %s in %s", tag, click.open_id, click.card_id,
                click.chat_id)
    await adapter._dispatch_synthetic_event(
        text=_click_text(tag, action), message_type=MessageType.TEXT, chat_id=click.chat_id,
        sender_id=SimpleNamespace(open_id=click.open_id, user_id=click.user_id or None, union_id=click.union_id or None),
        event_chat_type=_event_chat_type(chat_info), raw_message=data,
        # No card id (a click outside a chat card) maps back to no reply target: the answer is a new message.
        message_id=f"{click.card_id}{_CARD_ID_MARK}{click.token or uuid.uuid4().hex}",
        thread_id=str(getattr(card, "thread_id", "") or "") or None,
        reply_to_message_id=click.card_id or None, reply_to_text=quote.text, reply_to_is_own_message=quote.is_own,
    )


async def _card_message(adapter: Any, card_id: str) -> Any:
    """The clicked card as ``GET im/v1/messages`` reports it (its topic), cached for the quote; None when unreadable."""
    from .feishu_forward import fetch_message_items
    from .feishu_quote import cache_node

    if not card_id or not adapter._client:
        return None
    try:
        items = await fetch_message_items(adapter, card_id)
    except Exception:
        logger.debug("[Feishu] Failed to fetch clicked card %s", card_id, exc_info=True)
        return None
    if not items:
        return None
    try:
        await cache_node(adapter, card_id, items)
    except Exception:  # the quote then fetches it again
        logger.debug("[Feishu] Failed to read clicked card %s as a quote", card_id, exc_info=True)
    return items[0]


def _click_text(tag: str, action: Any) -> str:
    fields = {name: value for name in _ACTION_FIELDS
              if (value := getattr(action, name, None)) not in (None, "", [], {})}
    try:
        payload = json.dumps(fields, ensure_ascii=False) if fields else ""
    except (TypeError, ValueError):
        payload = ""
    return f"card:{tag} {payload}".rstrip()
