"""The session prompt names this conversation's exact send target.

Without it the only raw chat id in the prompt is the home channel's, so a model asked to send "here"
(``hermes send --to``, a cron ``deliver``) copies that one and the message lands in another chat.
"""

from __future__ import annotations

import re

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.session import SessionSource, build_session_context, build_session_context_prompt

_TARGET = re.compile(r"\*\*Reply Target:\*\* `([^`]+)`")


def _prompt(*, platform=Platform.FEISHU, chat_id="oc_group", thread_id=None, redact_pii=False):
    home = HomeChannel(platform=platform, chat_id="oc_owner_dm", name="Owner DM")
    config = GatewayConfig(platforms={platform: PlatformConfig(enabled=True, token="t", home_channel=home)})
    source = SessionSource(platform=platform, chat_id=chat_id, chat_name="Ops", chat_type="group",
                           user_name="Human", thread_id=thread_id)
    return build_session_context_prompt(build_session_context(source, config), redact_pii=redact_pii)


def test_the_reply_target_is_this_conversation_and_comes_before_the_home_channel():
    prompt = _prompt()

    assert _TARGET.findall(prompt) == ["feishu:oc_group"]
    assert prompt.index("**Reply Target:**") < prompt.index("oc_owner_dm")


def test_a_topic_reply_target_names_the_topic():
    assert _TARGET.findall(_prompt(thread_id="omt_topic")) == ["feishu:oc_group:omt_topic"]


def test_the_reply_target_parses_back_to_this_conversation():
    """The line is a target the send engine accepts verbatim, not a description of one."""
    from tools.send_message_targets import resolve_send_target

    [target] = _TARGET.findall(_prompt(thread_id="omt_topic"))
    platform, _, ref = target.partition(":")

    assert (platform, *resolve_send_target(platform, ref)) == ("feishu", "oc_group", "omt_topic", None)


def test_pii_redaction_hides_the_reply_target():
    prompt = _prompt(platform=Platform.TELEGRAM, chat_id="-1001234567890", redact_pii=True)

    assert not _TARGET.findall(prompt)
    assert "-1001234567890" not in prompt
