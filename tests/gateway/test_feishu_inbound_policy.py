from __future__ import annotations

from types import SimpleNamespace

from gateway.platforms.feishu_inbound.policy import (
    allow_feishu_group_message,
    feishu_message_mentions_bot,
    feishu_post_mentions_bot,
    should_accept_feishu_group_message,
)


class _Rule:
    def __init__(self, *, policy: str, allowlist=None, blacklist=None):
        self.policy = policy
        self.allowlist = set(allowlist or [])
        self.blacklist = set(blacklist or [])


def test_allow_group_message_respects_admin_allowlist_blacklist_and_disabled():
    sender = SimpleNamespace(open_id='ou_alice', user_id=None)
    assert allow_feishu_group_message(sender_id=sender, admins={'ou_alice'}) is True
    assert allow_feishu_group_message(sender_id=sender, group_rules={'chat': _Rule(policy='allowlist', allowlist=['ou_alice'])}, chat_id='chat') is True
    assert allow_feishu_group_message(sender_id=sender, group_rules={'chat': _Rule(policy='blacklist', blacklist=['ou_alice'])}, chat_id='chat') is False
    assert allow_feishu_group_message(sender_id=sender, group_rules={'chat': _Rule(policy='disabled')}, chat_id='chat') is False


def test_mention_helpers_match_bot_identity():
    mentions = [SimpleNamespace(name='Hermes', id=SimpleNamespace(open_id='ou_bot', user_id='u_bot'))]
    assert feishu_message_mentions_bot(mentions, bot_open_id='ou_bot') is True
    assert feishu_message_mentions_bot(mentions, bot_name='Hermes') is True
    assert feishu_message_mentions_bot(mentions, bot_open_id='ou_other') is False
    assert feishu_post_mentions_bot(['ou_bot'], bot_open_id='ou_bot') is True
    assert feishu_post_mentions_bot(['u_bot'], bot_user_id='u_bot') is True
    assert feishu_post_mentions_bot([], bot_open_id='ou_bot') is False


def test_should_accept_group_message_requires_policy_and_bot_or_all_mention():
    message = SimpleNamespace(content='{"text":"@_all hi"}', mentions=[])
    sender = SimpleNamespace(open_id='ou_alice', user_id=None)
    assert should_accept_feishu_group_message(
        message=message,
        sender_id=sender,
        group_policy='open',
        normalize_message=lambda **_: SimpleNamespace(mentioned_ids=[]),
    ) is True

    bot_mentioned = SimpleNamespace(content='', mentions=[SimpleNamespace(name='Hermes', id=SimpleNamespace(open_id='ou_bot', user_id=None))])
    assert should_accept_feishu_group_message(
        message=bot_mentioned,
        sender_id=sender,
        group_rules={'chat': _Rule(policy='allowlist', allowlist=['ou_alice'])},
        chat_id='chat',
        bot_open_id='ou_bot',
        normalize_message=lambda **_: SimpleNamespace(mentioned_ids=[]),
    ) is True

    other_mentioned = SimpleNamespace(content='', mentions=[SimpleNamespace(name='Other', id=SimpleNamespace(open_id='ou_other', user_id=None))])
    assert should_accept_feishu_group_message(
        message=other_mentioned,
        sender_id=sender,
        group_policy='open',
        bot_open_id='ou_bot',
        normalize_message=lambda **_: SimpleNamespace(mentioned_ids=[]),
    ) is False
