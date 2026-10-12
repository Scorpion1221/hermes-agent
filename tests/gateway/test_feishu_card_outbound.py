"""Feishu replies go out as Card JSON 2.0 cards (``render_mode: card``, the default) and are edited as cards.

The fake client stands in for the lark SDK at ``im.v1.message``: it records each create/patch/update request
and answers with whatever the test queued. Everything above it is the real adapter.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter


def _ok(message_id: str = "om_sent") -> SimpleNamespace:
    return SimpleNamespace(success=lambda: True, code=0, msg="success", data=SimpleNamespace(message_id=message_id))


def _refused(code: int, msg: str) -> SimpleNamespace:
    return SimpleNamespace(success=lambda: False, code=code, msg=msg, data=None)


class _MessageAPI:
    def __init__(self, answers: dict[str, list] | None = None):
        self.calls: list[tuple[str, object]] = []
        self._answers = {name: list(queue) for name, queue in (answers or {}).items()}

    def _answer(self, name: str, request: object) -> object:
        self.calls.append((name, request))
        queue = self._answers.get(name)
        return queue.pop(0) if queue else _ok()

    def create(self, request):
        return self._answer("create", request)

    def patch(self, request):
        return self._answer("patch", request)

    def update(self, request):
        return self._answer("update", request)


def _adapter(api: _MessageAPI, **extra) -> FeishuAdapter:
    adapter = FeishuAdapter(PlatformConfig(extra=extra))
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=api)))

    async def _direct(func, *args):
        return func(*args)

    adapter._run_blocking = _direct
    return adapter


def _sent(api: _MessageAPI) -> list[tuple[str, str]]:
    return [(r.request_body.msg_type, r.request_body.content) for name, r in api.calls if name == "create"]


def _card_markdown(payload: str) -> str:
    card = json.loads(payload)
    assert card["schema"] == "2.0"
    (element,) = card["body"]["elements"]
    assert element["tag"] == "markdown"
    return element["content"]


def test_a_reply_goes_out_as_a_card_with_its_markdown_rendered_for_cards():
    api = _MessageAPI()
    reply = "# Plan\n**Status:**done\n| a | b |\n|---|---|\n| 1 | 2 |\n```\n# not a heading\n```"

    result = asyncio.run(_adapter(api).send("oc_chat", reply))

    assert result.success and result.message_id == "om_sent"
    ((msg_type, payload),) = _sent(api)
    assert msg_type == "interactive"
    markdown = _card_markdown(payload)
    assert markdown.startswith("### Plan\n**Status:** done\n| a | b |")
    assert "```\n# not a heading\n```" in markdown


def test_plain_prose_is_a_card_too_so_every_reply_looks_the_same():
    api = _MessageAPI()

    asyncio.run(_adapter(api).send("oc_chat", "Just a sentence."))

    ((msg_type, payload),) = _sent(api)
    assert msg_type == "interactive" and _card_markdown(payload) == "Just a sentence."


def test_every_chunk_of_a_long_reply_is_a_card():
    api = _MessageAPI()
    adapter = _adapter(api)
    adapter.truncate_message = lambda text, limit: ["Plain intro.", "## Then\nmore"]

    asyncio.run(adapter.send("oc_chat", "Plain intro.\n## Then\nmore"))

    assert [msg_type for msg_type, _ in _sent(api)] == ["interactive", "interactive"]


def test_render_mode_post_keeps_the_post_and_text_choice():
    api = _MessageAPI()
    adapter = _adapter(api, render_mode="post")

    asyncio.run(adapter.send("oc_chat", "Just a sentence."))
    asyncio.run(adapter.send("oc_chat", "## Heading\n**bold**"))

    assert [msg_type for msg_type, _ in _sent(api)] == ["text", "post"]


def test_a_card_feishu_refuses_for_its_content_is_resent_as_plain_text():
    api = _MessageAPI({"create": [_refused(230099, "Failed to create card content, invalid card schema")]})

    result = asyncio.run(_adapter(api).send("oc_chat", "## Title\n**bold** words"))

    assert result.success
    assert [msg_type for msg_type, _ in _sent(api)] == ["interactive", "text"]
    assert json.loads(_sent(api)[1][1]) == {"text": "Title\nbold words"}


def test_an_unrelated_send_failure_is_not_turned_into_a_text_resend():
    api = _MessageAPI({"create": [_refused(99991663, "tenant access token invalid")]})

    result = asyncio.run(_adapter(api).send("oc_chat", "hello"))

    assert not result.success and "99991663" in (result.error or "")
    assert [msg_type for msg_type, _ in _sent(api)] == ["interactive"]


def test_a_placeholder_mention_never_reaches_feishu_and_a_refused_real_one_is_resent_as_text():
    api = _MessageAPI({"create": [_refused(100290, "invalid user resource")]})
    reply = '<at user_id="ou_?">Ghost</at> and <at user_id="ou_0123456789abcdef">Alice</at> please look'

    result = asyncio.run(_adapter(api).send("oc_chat", reply))

    assert result.success
    first, retry = (_card_markdown(payload) for _, payload in _sent(api))
    assert first == '@Ghost and <at user_id="ou_0123456789abcdef">Alice</at> please look'
    assert retry == "@Ghost and @Alice please look"


def test_editing_a_card_reply_patches_the_card():
    api = _MessageAPI()

    result = asyncio.run(_adapter(api).edit_message("oc_chat", "om_card", "## Done\n**Result:**ok"))

    assert result.success and result.message_id == "om_card"
    ((name, request),) = api.calls
    assert name == "patch" and request.message_id == "om_card"
    assert _card_markdown(request.request_body.content) == "#### Done\n**Result:** ok"


def test_editing_a_message_that_is_not_a_card_falls_back_to_put():
    api = _MessageAPI({"patch": [_refused(230001, "Your request contains an invalid request parameter")]})

    result = asyncio.run(_adapter(api).edit_message("oc_chat", "om_text", "plain update"))

    assert result.success
    assert [name for name, _ in api.calls] == ["patch", "update"]
    put = api.calls[1][1].request_body
    assert put.msg_type == "text" and json.loads(put.content) == {"text": "plain update"}


def test_an_edit_whose_mention_feishu_refuses_is_patched_again_with_mentions_as_text():
    api = _MessageAPI({"patch": [_refused(100290, "invalid user resource")]})

    result = asyncio.run(
        _adapter(api).edit_message("oc_chat", "om_card", '<at user_id="ou_0123456789abcdef">Alice</at> done'),
    )

    assert result.success
    assert [name for name, _ in api.calls] == ["patch", "patch"]
    assert _card_markdown(api.calls[1][1].request_body.content) == "@Alice done"


def test_render_mode_post_edits_with_put_only():
    api = _MessageAPI()

    asyncio.run(_adapter(api, render_mode="post").edit_message("oc_chat", "om_post", "## Heading"))

    assert [name for name, _ in api.calls] == ["update"]
    assert api.calls[0][1].request_body.msg_type == "post"


def test_a_streamed_preview_card_is_edited_in_place_through_to_its_final_text():
    api = _MessageAPI()
    adapter = _adapter(api)

    sent = asyncio.run(adapter.send("oc_chat", "Thinking…"))
    asyncio.run(adapter.edit_message("oc_chat", sent.message_id, "Partial **answer:**so far"))
    final = asyncio.run(adapter.edit_message("oc_chat", sent.message_id, "Full **answer:**42", finalize=True))

    assert final.success
    assert [name for name, _ in api.calls] == ["create", "patch", "patch"]
    assert _card_markdown(api.calls[-1][1].request_body.content) == "Full **answer:** 42"


def test_a_rate_limited_card_edit_is_reported_as_such_and_not_retried_as_put():
    api = _MessageAPI({"patch": [_refused(230020, "This operation triggers the frequency limit")]})

    result = asyncio.run(_adapter(api).edit_message("oc_chat", "om_card", "update"))

    assert not result.success and "230020" in (result.error or "")
    assert [name for name, _ in api.calls] == ["patch"]


def test_when_put_fails_too_the_patch_error_is_what_the_caller_sees():
    api = _MessageAPI({
        "patch": [_refused(230099, "Failed to create card content")],
        "update": [_refused(230001, "invalid request parameter")],
    })

    result = asyncio.run(_adapter(api).edit_message("oc_chat", "om_card", "update"))

    assert not result.success and "230099" in (result.error or "")


def test_a_card_still_refused_for_a_mention_after_the_retry_goes_out_as_plain_text():
    refused = _refused(100290, "invalid user resource")
    api = _MessageAPI({"create": [refused, refused]})

    result = asyncio.run(_adapter(api).send("oc_chat", '<at user_id="ou_0123456789abcdef">Alice</at> look'))

    assert result.success
    assert [msg_type for msg_type, _ in _sent(api)] == ["interactive", "interactive", "text"]
    assert json.loads(_sent(api)[2][1]) == {"text": "@Alice look"}


def test_an_unknown_render_mode_falls_back_to_cards():
    api = _MessageAPI()

    asyncio.run(_adapter(api, render_mode="cards").send("oc_chat", "hello"))

    assert [msg_type for msg_type, _ in _sent(api)] == ["interactive"]
