"""Feishu messages fetched for context: merge-forward transcripts, quoted-message text, card JSON.

The client is faked at the SDK boundary (``im.v1.message.get``, ``im.v1.message_resource.get``,
``contact.v3.user.get``) and the raw tenant GET (``client.request``), with payloads shaped like
the live API: ``GET im/v1/messages/{forward}`` returns the forward followed by every descendant
flattened (``upper_message_id`` = direct container), a forwarded resource downloads only through
the outermost forward's id, and ``contact/v3/users/batch`` rejects the whole call when any id is
unknown. All names, ids and texts are synthetic.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import time
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

ROOT = "om_fwd_root"
NESTED = "om_fwd_nested"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class _SdkRequest(SimpleNamespace):
    """What a bound SDK request builder produces (``_sdk_build`` falls back to a bare SimpleNamespace)."""


class _FakeSdkBuilder:
    def __init__(self):
        self._fields = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def _set(value):
            self._fields[name] = value
            return self

        return _set

    def build(self):
        return _SdkRequest(**self._fields)


class _FakeSdkRequestType:
    @staticmethod
    def builder():
        return _FakeSdkBuilder()


@pytest.fixture(autouse=True)
def _fake_lark_requests(monkeypatch):
    """Stand in for the lazily imported contact SDK request and the raw tenant GET builder."""
    import plugins.platforms.feishu.adapter as feishu_mod

    for parent in ("lark_oapi", "lark_oapi.api", "lark_oapi.api.contact"):
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    contact_v3 = types.ModuleType("lark_oapi.api.contact.v3")
    contact_v3.GetUserRequest = _FakeSdkRequestType
    monkeypatch.setitem(sys.modules, "lark_oapi.api.contact.v3", contact_v3)
    monkeypatch.setattr(
        feishu_mod, "_tenant_get_request", lambda uri, *, queries=None: SimpleNamespace(uri=uri, queries=queries),
    )


class FakeFeishu:
    """Feishu as seen through the SDK client the adapter holds."""

    def __init__(self, messages=None, users=None, *, get_ok=True):
        self.messages = messages or {}
        self.users = users or {}
        self.get_ok = get_ok
        self.get_delay = 0.0  # seconds a message GET takes (on the adapter's SDK executor thread)
        self.message_gets, self.resource_gets, self.batch_gets, self.user_gets = [], [], [], []
        self.im = SimpleNamespace(v1=SimpleNamespace(
            message=SimpleNamespace(get=self._get_message),
            message_resource=SimpleNamespace(get=self._get_resource),
        ))
        self.contact = SimpleNamespace(v3=SimpleNamespace(user=SimpleNamespace(get=self._get_user)))

    def _get_message(self, request):
        self.message_gets.append(request)
        time.sleep(self.get_delay)
        if not self.get_ok:
            return SimpleNamespace(success=lambda: False, code=230002, msg="no permission")
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(items=self.messages[request.message_id]))

    def _get_resource(self, request):
        self.resource_gets.append(request)
        if request.message_id != ROOT:
            return SimpleNamespace(success=lambda: False, code=234003, msg="File not in msg.")
        if request.type == "image":
            return SimpleNamespace(
                success=lambda: True, file=io.BytesIO(PNG), file_name=f"{request.file_key}.png",
                raw=SimpleNamespace(headers={"Content-Type": "image/png"}),
            )
        return SimpleNamespace(
            success=lambda: True, file=io.BytesIO(b"col_a,col_b\n1,2\n"), file_name=None,
            raw=SimpleNamespace(headers={"Content-Type": "application/pdf"}),
        )

    def request(self, request):  # raw tenant GET: contact/v3/users/batch
        self.batch_gets.append(request)
        ids = [value for key, value in request.queries if key == "user_ids"]
        if any(oid not in self.users for oid in ids):
            body = {"code": 99992351, "msg": "not a valid {open_id}"}
        else:
            body = {"code": 0, "data": {"items": [{"open_id": oid, "name": self.users[oid]} for oid in ids]}}
        return SimpleNamespace(raw=SimpleNamespace(content=json.dumps(body).encode()))

    def _get_user(self, request):
        self.user_gets.append(request.user_id)
        name = self.users.get(request.user_id)
        if name is None:
            return SimpleNamespace(success=lambda: False)
        user = SimpleNamespace(name=name, display_name=None, nickname=None, en_name=None)
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(user=user))


def item(message_id, msg_type, content, *, upper=ROOT, sender="ou_alice", at=0, sender_type="user"):
    return SimpleNamespace(
        message_id=message_id, msg_type=msg_type, upper_message_id=upper, create_time=str(at), mentions=None,
        body=SimpleNamespace(content=content if isinstance(content, str) else json.dumps(content)),
        sender=SimpleNamespace(
            id=sender, id_type="app_id" if sender_type == "app" else "open_id", sender_type=sender_type,
        ),
    )


def forward_root(message_id=ROOT, upper=None, at=0):
    return item(message_id, "merge_forward", "Merged and Forwarded Message", upper=upper, at=at)


def json_card(elements, header=None):
    card = {"schema": "2.0", "body": {"tag": "body", "property": {"elements": elements}}}
    if header:
        card["header"] = header
    return {"json_card": json.dumps(card), "card_schema": 2}


def node(tag, **prop):
    return {"tag": tag, "id": f"_{tag}", "property": prop}


def text(content, *styles):
    return node("plain_text", content=content, **({"textStyle": {"attributes": list(styles)}} if styles else {}))


def markdown(*children):
    return node("markdown", elements=list(children))


def make_adapter(client):
    from gateway.config import PlatformConfig
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = client
    adapter._app_id = "cli_self"
    adapter._bot_name = "Hermes"
    return adapter


def inbound_forward(adapter, message_id=ROOT):
    message = SimpleNamespace(
        message_id=message_id, message_type="merge_forward", content="Merged and Forwarded Message", mentions=None,
    )
    return asyncio.run(adapter._extract_message_content(message))


def dispatched_event(adapter, message):
    """Run one inbound message through ``_process_inbound_message``; return the event it dispatches."""
    adapter.get_chat_info = AsyncMock(return_value={"name": "DM"})
    adapter._resolve_sender_profile = AsyncMock(return_value={"user_id": "u1", "user_name": "Alice", "user_id_alt": None})
    adapter.build_source = Mock(return_value=SimpleNamespace(thread_id=None))
    adapter._dispatch_inbound_event = AsyncMock()
    asyncio.run(adapter._process_inbound_message(
        data={}, message=message, sender_id=SimpleNamespace(open_id="ou_alice", user_id=None, union_id=None),
        chat_type="p2p", message_id=message.message_id,
    ))
    return adapter._dispatch_inbound_event.await_args.args[0]


def inbound_message(message_id, message_type, content, *, parent_id=None):
    return SimpleNamespace(
        message_id=message_id, message_type=message_type, content=content, mentions=None, chat_id="oc_dm",
        parent_id=parent_id, upper_message_id=None, root_id=parent_id, thread_id=None,
    )


# --- inbound merge-forward ---

def test_inbound_merge_forward_renders_a_named_nested_chronological_transcript():
    nested_card = json_card([markdown(text("Build "), text("green", "bold"))])
    items = [
        forward_root(),
        item("om_c3", "text", {"text": "see the thread below"}, sender="ou_bob", at=30),
        item("om_c1", "text", {"text": "kickoff at ten"}, at=10),
        forward_root(NESTED, upper=ROOT, at=20),
        item("om_n2", "interactive", nested_card, upper=NESTED, sender="cli_self", sender_type="app", at=6),
        item("om_n1", "post", {"title": "Plan", "content": [[{"tag": "text", "text": "step one"}]]},
             upper=NESTED, sender="ou_bob", at=5),
    ]
    client = FakeFeishu({ROOT: items}, users={"ou_alice": "Alice", "ou_bob": "Bob"})

    text_out, message_type, media_urls, _types, _inlined, _mentions = inbound_forward(make_adapter(client))

    assert text_out == "\n".join([
        "[Merged forward message]",
        "- Alice: kickoff at ten",
        "- Alice: [Merged forward message]",
        "  - Bob: Plan",
        "    step one",
        "  - Hermes: Build **green**",
        "- Bob: see the thread below",
    ])
    assert message_type.value == "text"
    assert media_urls == []
    (request,) = client.message_gets
    assert (request.message_id, request.card_msg_content_type) == (ROOT, "raw_card_content")


def test_inbound_merge_forward_downloads_child_media_through_the_outer_forward():
    post_with_image = {"title": "", "content": [[{"tag": "text", "text": "chart"}, {"tag": "img", "image_key": "img_post"}]]}
    items = [
        forward_root(),
        item("om_img", "image", {"image_key": "img_top"}, at=1),
        forward_root(NESTED, upper=ROOT, at=2),
        item("om_post", "post", post_with_image, upper=NESTED, at=2),
        item("om_file", "file", {"file_key": "file_report", "file_name": "report.pdf"}, upper=NESTED, at=3),
    ]
    client = FakeFeishu({ROOT: items}, users={"ou_alice": "Alice"})

    event = dispatched_event(
        make_adapter(client), inbound_message(ROOT, "merge_forward", "Merged and Forwarded Message"),
    )

    assert [(r.message_id, r.file_key) for r in client.resource_gets] == [
        (ROOT, "img_top"), (ROOT, "img_post"), (ROOT, "file_report"),
    ]
    assert event.message_type.value == "text"
    assert event.media_types == ["image/png", "image/png", "application/pdf"]
    assert all(os.path.isfile(path) for path in event.media_urls)
    assert "  - Alice: [Attachment: report.pdf]" in event.text.splitlines()


def test_forward_child_media_is_capped_while_every_child_stays_in_the_transcript():
    from plugins.platforms.feishu.feishu_forward import MAX_FORWARD_MEDIA

    count = MAX_FORWARD_MEDIA + 5
    items = [forward_root()] + [item(f"om_i{i}", "image", {"image_key": f"img_{i}"}, at=i) for i in range(count)]
    client = FakeFeishu({ROOT: items}, users={"ou_alice": "Alice"})

    text_out, _type, media_urls, _types, _inlined, _mentions = inbound_forward(make_adapter(client))

    assert len(client.resource_gets) == len(media_urls) == MAX_FORWARD_MEDIA
    assert text_out.count("- Alice: [Image]") == count


def test_long_forward_is_cut_at_the_entry_budget_and_says_how_much_is_missing():
    from plugins.platforms.feishu.feishu_forward import MAX_FORWARD_ENTRIES

    count = MAX_FORWARD_ENTRIES + 30
    items = [forward_root()] + [item(f"om_t{i}", "text", {"text": f"line {i}"}, at=i) for i in range(count)]
    client = FakeFeishu({ROOT: items}, users={"ou_alice": "Alice"})

    lines = inbound_forward(make_adapter(client))[0].splitlines()

    assert lines[MAX_FORWARD_ENTRIES] == f"- Alice: line {MAX_FORWARD_ENTRIES - 1}"
    assert lines[-1] == "- … 30 more forwarded message(s) not shown"


def test_unresolvable_forward_senders_get_neutral_labels_never_raw_ids():
    items = [
        forward_root(),
        item("om_a", "text", {"text": "from a colleague"}, sender="ou_alice", at=1),
        item("om_x", "text", {"text": "from outside"}, sender="ou_external", at=2),
        item("om_s", "text", {"text": "from this bot"}, sender="cli_self", sender_type="app", at=3),
        item("om_p", "text", {"text": "from another bot"}, sender="cli_peer", sender_type="app", at=4),
        item("om_y", "text", {"text": "outside again"}, sender="ou_external", at=5),
    ]
    client = FakeFeishu({ROOT: items}, users={"ou_alice": "Alice"})

    text_out = inbound_forward(make_adapter(client))[0]

    # One unknown id makes Feishu refuse the whole batch; each sender then gets its own lookup.
    assert len(client.batch_gets) == 1 and sorted(client.user_gets) == ["ou_alice", "ou_external"]
    assert text_out.splitlines()[1:] == [
        "- Alice: from a colleague",
        "- Participant 1: from outside",
        "- Hermes: from this bot",
        "- Bot: from another bot",
        "- Participant 1: outside again",
    ]
    assert "ou_" not in text_out and "cli_" not in text_out


def test_a_sender_no_lookup_can_name_is_not_looked_up_again_on_the_next_forward():
    items = [
        forward_root(),
        item("om_a", "text", {"text": "from a colleague"}, sender="ou_alice", at=1),
        item("om_x", "text", {"text": "from someone who left"}, sender="ou_departed", at=2),
    ]
    client = FakeFeishu({ROOT: items}, users={"ou_alice": "Alice"})
    adapter = make_adapter(client)

    first = inbound_forward(adapter)[0]
    second = inbound_forward(adapter)[0]

    assert first == second == "[Merged forward message]\n- Alice: from a colleague\n- Participant 1: from someone who left"
    assert len(client.batch_gets) == 1 and sorted(client.user_gets) == ["ou_alice", "ou_departed"]


def test_forward_sender_names_come_from_one_batch_call_and_are_cached():
    items = [
        forward_root(),
        item("om_a", "text", {"text": "one"}, sender="ou_alice", at=1),
        item("om_b", "text", {"text": "two"}, sender="ou_bob", at=2),
        item("om_c", "text", {"text": "three"}, sender="ou_alice", at=3),
    ]
    client = FakeFeishu({ROOT: items}, users={"ou_alice": "Alice", "ou_bob": "Bob"})
    adapter = make_adapter(client)

    first = inbound_forward(adapter)[0]
    second = inbound_forward(adapter)[0]

    assert first == second == "[Merged forward message]\n- Alice: one\n- Bob: two\n- Alice: three"
    (batch,) = client.batch_gets
    assert batch.uri == "/open-apis/contact/v3/users/batch"
    assert batch.queries == [("user_id_type", "open_id"), ("user_ids", "ou_alice"), ("user_ids", "ou_bob")]
    assert client.user_gets == []


def test_prefetched_names_keep_the_sender_cache_bounded():
    from plugins.platforms.feishu.feishu_names import NAME_CACHE_MAX, prefetch_user_names

    ids = [f"ou_user{i:05d}" for i in range(NAME_CACHE_MAX + 60)]
    client = FakeFeishu(users={oid: f"User {oid[-5:]}" for oid in ids})
    adapter = make_adapter(client)

    asyncio.run(prefetch_user_names(adapter, ids))

    assert len(adapter._sender_name_cache) == NAME_CACHE_MAX
    assert adapter._get_cached_sender_name(ids[-1]) == f"User {ids[-1][-5:]}"
    assert max(sum(key == "user_ids" for key, _ in r.queries) for r in client.batch_gets) <= 50


def test_context_references_in_a_forward_stay_literal(tmp_path):
    """Forwarded messages are third-party text: the gateway's @-reference expander must not read a
    local file or fetch a URL named in one, as it already refuses to for quoted replies."""
    from agent.context_references import preprocess_context_references_async

    (tmp_path / "plan.md").write_text("SECRET PLAN", encoding="utf-8")
    items = [
        forward_root(),
        item("om_a", "text", {"text": "fyi @file:plan.md"}, sender="ou_external", at=1),
        item("om_b", "text", {"text": "see @url:http://10.0.0.1/admin and @diff"}, sender="ou_external", at=2),
        item("om_c", "text", {"text": "@Alice: ok, mail me at alice@example.com"}, sender="ou_external", at=3),
    ]
    text_out = inbound_forward(make_adapter(FakeFeishu({ROOT: items})))[0]
    fetched = []

    result = asyncio.run(preprocess_context_references_async(
        text_out, cwd=tmp_path, allowed_root=tmp_path, context_length=100_000, url_fetcher=fetched.append,
    ))

    assert (result.references, result.expanded, fetched) == ([], False, [])
    assert "SECRET PLAN" not in result.message
    assert text_out.splitlines()[1:] == [
        "- Participant 1: fyi ＠file:plan.md",
        "- Participant 1: see ＠url:http://10.0.0.1/admin and ＠diff",
        "- Participant 1: @Alice: ok, mail me at alice@example.com",
    ]


@pytest.mark.parametrize("failure", ["refused", "raises"])
def test_merge_forward_fetch_failure_keeps_the_placeholder(failure):
    client = FakeFeishu(get_ok=False)
    if failure == "raises":
        client.im.v1.message.get = Mock(side_effect=RuntimeError("network down"))

    text_out, message_type, media_urls, _types, _inlined, _mentions = inbound_forward(make_adapter(client))

    assert (text_out, message_type.value, media_urls) == ("[Merged forward message]", "text", [])


def inbound_event(message_id, message_type, content):
    message = inbound_message(message_id, message_type, content)
    message.chat_type = "p2p"
    sender = SimpleNamespace(
        sender_type="user", sender_id=SimpleNamespace(open_id="ou_alice", user_id=None, union_id=None),
    )
    return SimpleNamespace(event=SimpleNamespace(message=message, sender=sender))


def delivered_turns(adapter, *events):
    """Deliver ``events`` concurrently, in arrival order, through the real inbound path; return each
    agent turn's text. The batch timer is parked and the pending batches flushed once all are in."""
    adapter.get_chat_info = AsyncMock(return_value={"name": "DM"})
    adapter.handle_message = AsyncMock()
    adapter._text_batch_delay_seconds = adapter._text_batch_split_delay_seconds = 600

    async def deliver():
        await asyncio.gather(*(adapter._handle_message_event_data(event) for event in events))
        for task in adapter._pending_text_batch_tasks.values():
            task.cancel()
        for key in list(adapter._pending_text_batches):
            await adapter._flush_text_batch_now(key)

    asyncio.run(deliver())
    return [call.args[0].text for call in adapter.handle_message.await_args_list]


def test_a_note_sent_behind_a_slow_forward_joins_it_as_one_turn_forward_first():
    items = [forward_root(), item("om_a", "text", {"text": "the numbers"}, at=1)]
    client = FakeFeishu({ROOT: items}, users={"ou_alice": "Alice"})
    client.get_delay = 0.5

    turns = delivered_turns(
        make_adapter(client),
        inbound_event(ROOT, "merge_forward", "Merged and Forwarded Message"),
        inbound_event("om_note", "text", json.dumps({"text": "please summarize"})),
    )

    assert turns == ["[Merged forward message]\n- Alice: the numbers\nplease summarize"]


def test_a_long_forward_transcript_does_not_push_its_note_into_a_turn_of_its_own():
    from plugins.platforms.feishu.adapter import _DEFAULT_TEXT_BATCH_MAX_CHARS
    from plugins.platforms.feishu.feishu_forward import MAX_ENTRY_CHARS

    count = _DEFAULT_TEXT_BATCH_MAX_CHARS // MAX_ENTRY_CHARS + 1
    items = [forward_root()] + [item(f"om_{i}", "text", {"text": "x" * MAX_ENTRY_CHARS}, at=i) for i in range(count)]
    client = FakeFeishu({ROOT: items}, users={"ou_alice": "Alice"})

    turns = delivered_turns(
        make_adapter(client),
        inbound_event(ROOT, "merge_forward", "Merged and Forwarded Message"),
        inbound_event("om_note", "text", json.dumps({"text": "please summarize"})),
    )

    assert len(turns) == 1 and len(turns[0]) > _DEFAULT_TEXT_BATCH_MAX_CHARS
    assert turns[0].startswith("[Merged forward message]\n") and turns[0].endswith("\nplease summarize")


def test_text_batches_without_a_forward_still_flush_at_the_char_cap():
    from plugins.platforms.feishu.adapter import _DEFAULT_TEXT_BATCH_MAX_CHARS

    long_text = "y" * (_DEFAULT_TEXT_BATCH_MAX_CHARS - 10)
    turns = delivered_turns(
        make_adapter(FakeFeishu()),
        inbound_event("om_1", "text", json.dumps({"text": long_text})),
        inbound_event("om_2", "text", json.dumps({"text": "and one more thing"})),
    )

    assert turns == [long_text, "and one more thing"]


# --- quoted messages (reply_to_text) ---

def test_quoted_card_v2_reply_reads_as_its_content():
    card = json_card([markdown(text("Answer: "), text("42", "bold"))])
    reply = item("om_card", "interactive", card, upper=None, sender="cli_self", sender_type="app")
    client = FakeFeishu({"om_card": [reply]})

    event = dispatched_event(
        make_adapter(client), inbound_message("om_q", "text", json.dumps({"text": "why?"}), parent_id="om_card"),
    )

    assert event.reply_to_text == "Answer: **42**"
    assert client.message_gets[0].card_msg_content_type == "raw_card_content"


def test_quoted_merge_forward_reads_as_its_transcript_without_downloading_media():
    items = [
        forward_root(),
        item("om_a", "text", {"text": "the numbers"}, at=1),
        item("om_i", "image", {"image_key": "img_chart"}, at=2),
    ]
    client = FakeFeishu({ROOT: items}, users={"ou_alice": "Alice"})

    quoted = asyncio.run(make_adapter(client)._fetch_quoted_reply(ROOT)).text

    assert quoted == "[Merged forward message]\n- Alice: the numbers\n- Alice: [Image]"
    assert client.resource_gets == []


def test_quoted_fetch_through_the_loaded_plugin_uses_its_bound_sdk_request(monkeypatch):
    """The plugin loader imports the adapter under its own package name; the fetch helpers must use
    that module's SDK bindings (``_load_lark_oapi`` binds them there), not a second copy's."""
    from gateway.config import PlatformConfig
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import discover_plugins

    discover_plugins()
    adapter_cls = platform_registry.get("feishu").adapter_factory
    monkeypatch.setattr(sys.modules[adapter_cls.__module__], "GetMessageRequest", _FakeSdkRequestType)
    client = FakeFeishu({"om_parent": [item("om_parent", "text", {"text": "the plan"}, upper=None)]})
    adapter = adapter_cls(PlatformConfig())
    adapter._client = client

    assert asyncio.run(adapter._fetch_quoted_reply("om_parent")).text == "the plan"
    (request,) = client.message_gets
    assert type(request) is _SdkRequest
    assert (request.message_id, request.card_msg_content_type) == ("om_parent", "raw_card_content")


def test_quoted_image_reads_as_an_image_placeholder():
    client = FakeFeishu({"om_img": [item("om_img", "image", {"image_key": "img_1"}, upper=None)]})

    assert asyncio.run(make_adapter(client)._fetch_quoted_reply("om_img")).text == "[Image]"


def test_quoted_message_lookup_failure_returns_none_and_retries_next_time():
    client = FakeFeishu(get_ok=False)
    adapter = make_adapter(client)

    assert asyncio.run(adapter._fetch_quoted_reply("om_gone")).text is None
    assert asyncio.run(adapter._fetch_quoted_reply("om_gone")).text is None
    assert len(client.message_gets) == 2


# --- card JSON rendering ---

def test_card_v2_json_card_renders_as_markdown():
    from plugins.platforms.feishu.adapter import normalize_feishu_message

    cell = lambda *children: {"data": markdown(*children)}  # noqa: E731
    header = node("card_header", title=text("Daily report"), subtitle=text("ops"))
    elements = [
        markdown(
            text("Status: "), text("ok", "bold"), node("br"), text("see "),
            node("link", content="dashboard", url={"url": "https://example.com/d"}),
            node("heading", level=2, elements=[text("Numbers")]),
            node("table", columns=[{"name": "0", "displayName": "Metric"}, {"name": "1", "displayName": "Value"}],
                 rows=[{"0": cell(text("p95")), "1": cell(node("code_span", content="120ms"))}]),
            node("list", items=[
                {"level": 0, "type": "ol", "elements": [text("first")]},
                {"level": 1, "type": "ul", "elements": [text("nested")]},
                {"level": 0, "type": "ol", "elements": [text("second")]},
            ]),
            node("blockquote", elements=[text("quoted")]),
            node("code_block", language="python", contents=[
                {"contents": [{"content": "print(", "contentType": "text"}, {"content": "1)", "contentType": "text"}]},
            ]),
            node("hr"),
        ),
    ]

    normalized = normalize_feishu_message(
        message_type="interactive", raw_content=json.dumps(json_card(elements, header=header)),
    )

    assert normalized.text_content == "\n\n".join([
        "Daily report", "ops", "Status: **ok**\nsee [dashboard](https://example.com/d)", "## Numbers",
        "| Metric | Value |\n| --- | --- |\n| p95 | `120ms` |", "1. first\n  - nested\n2. second",
        "> quoted", "```python\nprint(1)\n```", "---",
    ])


def test_schema_1_json_card_renders_title_body_and_actions():
    from plugins.platforms.feishu.adapter import normalize_feishu_message

    button = lambda label: node("button", text=text(label), type="default")  # noqa: E731
    card = {
        "header": node("card_header", title=text("Approval required"), template="orange"),
        "body": {"elements": [
            {**node("markdown_v1", elements=[text("Reason:", "bold"), text(" writes a config file")]),
             "fallback": markdown(text("downgraded copy"))},
            node("action", actions=[button("Allow"), button("Deny")]),
        ]},
        "config": {"convertVersion": 1},
    }

    normalized = normalize_feishu_message(
        message_type="interactive", raw_content=json.dumps({"json_card": json.dumps(card), "card_schema": 1}),
    )

    assert normalized.text_content == "Approval required\n\n**Reason:** writes a config file\n\nActions: Allow, Deny"


def test_card_tags_and_mentions_render_inline_without_breaking_the_paragraph():
    from plugins.platforms.feishu.feishu_card_text import render_json_card

    tag = lambda label: node("text_tag", color="green", text=text(label))  # noqa: E731
    header = node("card_header", title=text("Release"), textTagList=[tag("beta"), tag("P1")])
    elements = [markdown(
        text("Deploy "), tag("PASSED"), text(" on staging, owner "), node("at", userID="ou_owner0001"),
        text(" please verify"),
    )]

    rendered = render_json_card(json_card(elements, header=header))

    assert rendered == "Release [beta] [P1]\n\nDeploy [PASSED] on staging, owner @user please verify"


def test_card_text_is_bounded():
    from plugins.platforms.feishu.feishu_card_text import MAX_CARD_TEXT_CHARS, render_json_card

    rendered = render_json_card(json_card([markdown(text("x" * (MAX_CARD_TEXT_CHARS * 2)))]))

    assert len(rendered) == MAX_CARD_TEXT_CHARS + 1 and rendered.endswith("…")


def test_unreadable_json_card_falls_back_to_the_legacy_card_walker():
    from plugins.platforms.feishu.adapter import normalize_feishu_message

    normalized = normalize_feishu_message(
        message_type="interactive", raw_content=json.dumps({"json_card": "{not json", "title": "Legacy title"}),
    )

    assert normalized.text_content == "Legacy title"
