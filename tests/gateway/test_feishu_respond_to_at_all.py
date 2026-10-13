"""``platforms.feishu.extra.respond_to_at_all``: whether an @all in a group counts as mentioning the bot.

Default on (Feishu's @everyone reaches every member, the bot included). A deployment that sits in big
announcement groups turns it off so an @all broadcast no longer starts a turn; a real @bot still does.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter

_OPEN_GROUP = {"FEISHU_GROUP_POLICY": "open"}


def _adapter(**extra) -> FeishuAdapter:
    adapter = FeishuAdapter(PlatformConfig(extra=extra))
    adapter._bot_open_id, adapter._bot_user_id, adapter._bot_name = "ou_bot", "", "Hermes"
    return adapter


def _admitted(adapter: FeishuAdapter, content: str, mentions=()) -> bool:
    sender = SimpleNamespace(sender_type="user", sender_id=SimpleNamespace(open_id="ou_alice", user_id=None))
    message = SimpleNamespace(content=content, mentions=list(mentions), chat_type="group", chat_id="oc_group",
                              message_type="text")
    return adapter._admit(sender, message) is None


def _at_bot():
    return SimpleNamespace(key="@_user_1", id=SimpleNamespace(open_id="ou_bot", user_id=None), name="Hermes")


@patch.dict(os.environ, _OPEN_GROUP, clear=True)
@pytest.mark.parametrize("extra,admitted", [({}, True), ({"respond_to_at_all": True}, True),
                                            ({"respond_to_at_all": False}, False)])
def test_an_at_all_broadcast_starts_a_turn_only_while_the_setting_is_on(extra, admitted):
    assert _admitted(_adapter(**extra), '{"text":"@_all 全员注意：周五停电"}') is admitted


@patch.dict(os.environ, _OPEN_GROUP, clear=True)
def test_with_at_all_off_a_message_that_also_mentions_the_bot_is_still_answered():
    adapter = _adapter(respond_to_at_all=False)
    assert _admitted(adapter, '{"text":"@_all @_user_1 帮忙确认一下"}', mentions=[_at_bot()])


@patch.dict(os.environ, _OPEN_GROUP, clear=True)
def test_with_at_all_off_an_unmentioned_group_message_is_still_ignored():
    assert not _admitted(_adapter(respond_to_at_all=False), '{"text":"随便聊聊"}')


@patch.dict(os.environ, _OPEN_GROUP, clear=True)
def test_a_group_that_needs_no_mention_answers_an_at_all_either_way():
    adapter = _adapter(respond_to_at_all=False, group_rules={"oc_group": {"policy": "open", "require_mention": False}})
    assert _admitted(adapter, '{"text":"@_all 全员注意"}')
