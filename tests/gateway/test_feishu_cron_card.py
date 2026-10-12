"""A cron delivery to Feishu renders as a notification card; everything else stays an ordinary reply.

The wrapped text comes from the real cron ``_deliver_result`` (captured at its standalone send), so the test
breaks if cron's wrapper and the card's parser drift apart. The Feishu side is the real adapter with a fake
``im.v1.message`` client.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform, PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter
from plugins.platforms.feishu.feishu_cron_card import parse_cron_delivery


class _MessageAPI:
    def __init__(self):
        self.created: list = []

    def create(self, request):
        self.created.append(request)
        return SimpleNamespace(success=lambda: True, code=0, msg="ok", data=SimpleNamespace(message_id="om_cron"))


def _adapter(**extra) -> tuple[FeishuAdapter, _MessageAPI]:
    api = _MessageAPI()
    adapter = FeishuAdapter(PlatformConfig(extra=extra))
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=api)))

    async def _direct(func, *args):
        return func(*args)

    adapter._run_blocking = _direct
    return adapter, api


def _cron_text(job: dict, output: str, *, wrap: bool = True) -> str:
    """What cron hands the Feishu standalone sender for ``output``."""
    from cron.scheduler_delivery import _deliver_result

    pconfig = MagicMock()
    pconfig.enabled = True
    gateway_cfg = MagicMock()
    gateway_cfg.platforms = {Platform.FEISHU: pconfig}
    send = AsyncMock(return_value={"success": True})
    with patch("gateway.config.load_gateway_config", return_value=gateway_cfg), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": wrap}}), \
         patch("tools.send_message_tool._send_to_platform", new=send):
        _deliver_result({**job, "deliver": "origin", "origin": {"platform": "feishu", "chat_id": "oc_home"}}, output)
    send.assert_called_once()
    return send.call_args[0][3]


def _cards(api: _MessageAPI) -> list[dict]:
    assert all(r.request_body.msg_type == "interactive" for r in api.created)
    return [json.loads(r.request_body.content) for r in api.created]


JOB = {"id": "a25ca1ae14f5", "name": "GitHub Releases 早检"}


def test_a_wrapped_cron_delivery_becomes_a_notification_card():
    adapter, api = _adapter()
    text = _cron_text(JOB, "# Summary\n**Status:**no new releases")

    assert asyncio.run(adapter.send("oc_home", text, metadata={"job_id": JOB["id"]})).success

    (card,) = _cards(api)
    assert card["header"]["title"]["content"] == "⏰ GitHub Releases 早检"
    body, rule, details, hint = card["body"]["elements"]
    assert body["content"] == "### Summary\n**Status:** no new releases"
    assert rule == {"tag": "hr"}
    assert "a25ca1ae14f5" in details["content"]
    assert "Cronjob Response" not in json.dumps(card, ensure_ascii=False)
    assert hint["content"]


def test_the_standalone_path_without_metadata_gets_the_same_card():
    adapter, api = _adapter()

    asyncio.run(adapter.send("oc_home", _cron_text(JOB, "All good.")))

    (card,) = _cards(api)
    assert card["header"]["title"]["content"] == "⏰ GitHub Releases 早检"
    assert card["body"]["elements"][0]["content"] == "All good."


def test_a_long_cron_output_spans_cards_with_the_header_first_and_the_footer_last():
    adapter, api = _adapter()
    adapter.truncate_message = lambda text, limit: ["part one", "part two", "part three"]

    asyncio.run(adapter.send("oc_home", _cron_text(JOB, "part one part two part three")))

    first, middle, last = _cards(api)
    assert "header" in first and "header" not in middle and "header" not in last
    assert [len(c["body"]["elements"]) for c in (first, middle, last)] == [1, 1, 4]


def test_unwrapped_cron_output_and_ordinary_replies_are_not_cron_cards():
    adapter, api = _adapter()

    asyncio.run(adapter.send("oc_home", _cron_text(JOB, "Just the output.", wrap=False)))
    asyncio.run(adapter.send("oc_home", "Cronjob Response: looks like it\nbut is a normal reply"))

    for card in _cards(api):
        assert "header" not in card and len(card["body"]["elements"]) == 1


def test_render_mode_post_sends_the_cron_text_as_cron_wrote_it():
    adapter, api = _adapter(render_mode="post")
    text = _cron_text(JOB, "All good.")

    asyncio.run(adapter.send("oc_home", text))

    (request,) = api.created
    assert request.request_body.msg_type in {"post", "text"}
    assert "Cronjob Response: GitHub Releases 早检" in request.request_body.content


def test_the_parser_reads_exactly_cron_s_wrapper():
    delivery = parse_cron_delivery(_cron_text(JOB, "line 1\n\nline 2"))

    assert delivery is not None
    assert (delivery.name, delivery.job_id, delivery.body) == ("GitHub Releases 早检", "a25ca1ae14f5", "line 1\n\nline 2")
    assert parse_cron_delivery(_cron_text({**JOB, "name": "other"}, "x").replace("stop reminder other", "stop x")) is None


def _refused(code: int, msg: str) -> SimpleNamespace:
    return SimpleNamespace(success=lambda: False, code=code, msg=msg, data=None)


class _ScriptedAPI(_MessageAPI):
    def __init__(self, answers):
        super().__init__()
        self._answers = list(answers)

    def create(self, request):
        self.created.append(request)
        return self._answers.pop(0) if self._answers else SimpleNamespace(
            success=lambda: True, code=0, msg="ok", data=SimpleNamespace(message_id="om_cron"))


def _scripted_adapter(answers) -> tuple[FeishuAdapter, _ScriptedAPI]:
    adapter, _ = _adapter()
    api = _ScriptedAPI(answers)
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=api)))
    return adapter, api


def test_a_cron_card_resent_without_mentions_is_still_a_cron_card():
    adapter, api = _scripted_adapter([_refused(100290, "invalid user resource")])
    text = _cron_text(JOB, '<at user_id="ou_0123456789abcdef">Alice</at> please check')

    asyncio.run(adapter.send("oc_home", text))

    retry = json.loads(api.created[1].request_body.content)
    assert retry["header"]["title"]["content"] == "⏰ GitHub Releases 早检"
    assert retry["body"]["elements"][0]["content"] == "@Alice please check"


def test_a_cron_card_that_falls_back_to_text_still_says_which_job_sent_it():
    adapter, api = _scripted_adapter([_refused(230099, "Failed to create card content, invalid card schema")])

    asyncio.run(adapter.send("oc_home", _cron_text(JOB, "All good.")))

    assert api.created[1].request_body.msg_type == "text"
    assert json.loads(api.created[1].request_body.content)["text"] == "⏰ GitHub Releases 早检 (a25ca1ae14f5)\n\nAll good."


def test_an_unknown_render_mode_still_renders_cron_cards_like_the_default():
    adapter, api = _adapter(render_mode="cards")

    asyncio.run(adapter.send("oc_home", _cron_text(JOB, "All good.")))

    (card,) = _cards(api)
    assert card["header"]["title"]["content"] == "⏰ GitHub Releases 早检"
