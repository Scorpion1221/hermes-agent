from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Optional


def _sender_ids(sender_id: Any) -> set[str]:
    open_id = getattr(sender_id, 'open_id', None)
    user_id = getattr(sender_id, 'user_id', None)
    return {str(value).strip() for value in (open_id, user_id) if value}


def _rule_policy(rule: Any, default: str) -> str:
    if rule is None:
        return default
    return str(getattr(rule, 'policy', '') or '').strip().lower() or default


def _rule_allowlist(rule: Any) -> set[str]:
    if rule is None:
        return set()
    raw = getattr(rule, 'allowlist', set()) or set()
    return {str(value).strip() for value in raw if str(value).strip()}


def _rule_blacklist(rule: Any) -> set[str]:
    if rule is None:
        return set()
    raw = getattr(rule, 'blacklist', set()) or set()
    return {str(value).strip() for value in raw if str(value).strip()}


def allow_feishu_group_message(
    *,
    sender_id: Any,
    chat_id: str = '',
    admins: Iterable[str] = (),
    group_rules: Mapping[str, Any] | None = None,
    default_group_policy: str = '',
    group_policy: str = '',
    allowed_group_users: Iterable[str] = (),
    is_bot: bool = False,
) -> bool:
    sender_ids = _sender_ids(sender_id)
    admin_ids = {str(value).strip() for value in admins if str(value).strip()}
    if sender_ids and admin_ids and (sender_ids & admin_ids):
        return True

    group_rules = group_rules or {}
    rule = group_rules.get(chat_id) if chat_id else None
    policy = _rule_policy(rule, default_group_policy or group_policy)
    allowlist = _rule_allowlist(rule) if rule is not None else {str(v).strip() for v in allowed_group_users if str(v).strip()}
    blacklist = _rule_blacklist(rule)

    if policy == 'disabled':
        return False
    if policy == 'open':
        return True
    if policy == 'admin_only':
        return False
    # Bots cleared upstream by FEISHU_ALLOW_BOTS bypass allowlist/blacklist gates.
    if is_bot:
        return True
    if policy == 'allowlist':
        return bool(sender_ids and (sender_ids & allowlist))
    if policy == 'blacklist':
        return bool(sender_ids and not (sender_ids & blacklist))

    fallback_allowlist = {str(v).strip() for v in allowed_group_users if str(v).strip()}
    return bool(sender_ids and (sender_ids & fallback_allowlist))


def feishu_message_mentions_bot(
    mentions: list[Any],
    *,
    bot_open_id: str = '',
    bot_user_id: str = '',
    bot_name: str = '',
) -> bool:
    for mention in mentions:
        mention_id = getattr(mention, 'id', None)
        mention_open_id = getattr(mention_id, 'open_id', None)
        mention_user_id = getattr(mention_id, 'user_id', None)
        mention_name = (getattr(mention, 'name', None) or '').strip()
        if bot_open_id and mention_open_id == bot_open_id:
            return True
        if bot_user_id and mention_user_id == bot_user_id:
            return True
        if bot_name and mention_name == bot_name:
            return True
    return False


def feishu_post_mentions_bot(mentioned_ids: list[str], *, bot_open_id: str = '', bot_user_id: str = '') -> bool:
    normalized_ids = {str(value).strip() for value in mentioned_ids if str(value).strip()}
    if not normalized_ids:
        return False
    return (bot_open_id and bot_open_id in normalized_ids) or (bot_user_id and bot_user_id in normalized_ids)


def should_accept_feishu_group_message(
    *,
    message: Any,
    sender_id: Any,
    chat_id: str = '',
    admins: Iterable[str] = (),
    group_rules: Mapping[str, Any] | None = None,
    default_group_policy: str = '',
    group_policy: str = '',
    allowed_group_users: Iterable[str] = (),
    bot_open_id: str = '',
    bot_user_id: str = '',
    bot_name: str = '',
    normalize_message: Callable[..., Any],
) -> bool:
    if not allow_feishu_group_message(
        sender_id=sender_id,
        chat_id=chat_id,
        admins=admins,
        group_rules=group_rules,
        default_group_policy=default_group_policy,
        group_policy=group_policy,
        allowed_group_users=allowed_group_users,
    ):
        return False
    raw_content = getattr(message, 'content', '') or ''
    if '@_all' in raw_content:
        return True
    mentions = getattr(message, 'mentions', None) or []
    if mentions:
        return feishu_message_mentions_bot(
            mentions,
            bot_open_id=bot_open_id,
            bot_user_id=bot_user_id,
            bot_name=bot_name,
        )
    normalized = normalize_message(
        message_type=getattr(message, 'message_type', '') or '',
        raw_content=raw_content,
    )
    mentioned_ids = list(getattr(normalized, 'mentioned_ids', []) or [])
    return feishu_post_mentions_bot(
        mentioned_ids,
        bot_open_id=bot_open_id,
        bot_user_id=bot_user_id,
    )


__all__ = [
    'allow_feishu_group_message',
    'feishu_message_mentions_bot',
    'feishu_post_mentions_bot',
    'should_accept_feishu_group_message',
]
