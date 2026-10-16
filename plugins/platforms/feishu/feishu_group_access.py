"""Per-group access on Feishu: ``platforms.feishu.extra.group_rules`` and the intake gate for group traffic.

The adapter gates every group message with ``_allow_group_message`` before dispatch. The gateway's
user allowlist runs again afterwards and asks ``authorizes_group_sender``, which is narrower: only
an explicit rule for that chat counts, never the default policy.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FeishuGroupRule:
    """Per-group policy rule for controlling which users may interact with the bot."""

    policy: str  # "open" | "allowlist" | "blacklist" | "admin_only" | "disabled"
    allowlist: set[str] = field(default_factory=set)
    blacklist: set[str] = field(default_factory=set)
    require_mention: Optional[bool] = None  # None = inherit global
    policy_explicit: bool = True  # False when the config entry omitted ``policy`` (intake then reads it as open)


class FeishuGroupAccessMixin:
    def _require_mention_for(self, chat_id: str) -> bool:
        rule = self._group_rules.get(chat_id) if chat_id else None
        if rule and rule.require_mention is not None:
            return rule.require_mention
        return self._require_mention

    def _allow_group_message(self, sender_id: Any, chat_id: str = "", *, is_bot: bool = False) -> bool:
        """Per-group policy gate for non-DM traffic."""
        sender_ids = {getattr(sender_id, "open_id", None), getattr(sender_id, "user_id", None)} - {None}
        if sender_ids and self._admins and (sender_ids & self._admins):
            return True
        rule = self._group_rules.get(chat_id) if chat_id else None
        if rule:
            policy, allowlist, blacklist = rule.policy, rule.allowlist, rule.blacklist
        else:
            policy, allowlist, blacklist = self._default_group_policy or self._group_policy, self._allowed_group_users, set()
        # Channel locks apply to everyone; allowlist/blacklist only gate humans (bots were
        # already cleared upstream by FEISHU_ALLOW_BOTS).
        if policy in ("disabled", "admin_only"):
            return False
        if policy == "open" or is_bot:
            return True
        if policy == "allowlist":
            return bool(sender_ids and (sender_ids & allowlist))
        if policy == "blacklist":
            return bool(sender_ids and not (sender_ids & blacklist))
        return bool(sender_ids and (sender_ids & self._allowed_group_users))

    def authorizes_group_sender(self, chat_id: Optional[str], sender_ids: Iterable[str]) -> bool:
        """Whether this chat's own ``group_rules`` entry lets the sender use the bot: ``open``, or an
        ``allowlist`` naming one of ``sender_ids``. The gateway's user allowlist defers to it, or
        FEISHU_ALLOWED_USERS would silently ignore every member the rule admitted.

        Only an explicit rule for this chat counts, because the operator chose who may talk there by
        naming the chat or its members. The default policy (``default_group_policy`` /
        ``FEISHU_GROUP_POLICY``), an entry that leaves ``policy`` unset (intake reads it as ``open``),
        and ``blacklist`` all admit people nobody listed, so none of them count (SECURITY.md §2.6).
        ``admin_only`` admits ``admins``, who are not part of this grant. The rule
        is read on every check, so removing it revokes access for follow-ups that are already queued.
        """
        rule = self._group_rules.get(str(chat_id or ""))
        if rule is None or not rule.policy_explicit:
            return False
        if rule.policy == "open":
            return True
        return rule.policy == "allowlist" and bool({str(i) for i in sender_ids if i} & rule.allowlist)
