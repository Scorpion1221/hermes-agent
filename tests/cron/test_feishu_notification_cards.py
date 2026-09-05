"""Real scheduler -> router/standalone -> Feishu rendering, with network-only fakes."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cron.scheduler import _deliver_result
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.platforms.base import SendResult
from gateway.platforms.feishu_inbound.cardkit import build_cron_notification_card
from plugins.platforms.feishu.adapter import FeishuAdapter, _standalone_send


@pytest.fixture
def delivery(monkeypatch):
    pconfig = PlatformConfig(enabled=True, extra={"app_id": "test-app", "app_secret": "test-secret"})
    config = GatewayConfig(platforms={Platform.FEISHU: pconfig})
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr("cron.scheduler.load_config", lambda: {"cron": {"wrap_response": True}})
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    entry = PlatformEntry(
        name="feishu", label="Feishu", adapter_factory=FeishuAdapter,
        check_fn=lambda: True, standalone_sender_fn=_standalone_send,
        max_message_length=8000,
    )
    monkeypatch.setattr(platform_registry, "get", lambda name, **kwargs: entry if name == "feishu" else None)
    monkeypatch.setattr("plugins.platforms.feishu.adapter._load_lark_oapi", lambda: True)
    monkeypatch.setattr(FeishuAdapter, "_build_lark_client", lambda *args, **kwargs: object())
    response = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="delivered-card"))
    wire = AsyncMock(return_value=response)
    monkeypatch.setattr(FeishuAdapter, "_feishu_send_with_retry", wire)
    adapter = FeishuAdapter(pconfig)
    adapter._client = object()
    job = {"id": "job-test", "name": "配额检查", "deliver": "origin", "origin": {"platform": "feishu", "chat_id": "oc_test"}}
    return adapter, job, wire


@pytest.mark.asyncio
@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("status", ["completed", "error"])
async def test_cron_delivery_renders_the_same_structured_card_on_both_paths(delivery, live, status):
    adapter, job, wire = delivery
    kwargs = {"adapters": {Platform.FEISHU: adapter}, "loop": asyncio.get_running_loop()} if live else {}
    error = await asyncio.to_thread(
        _deliver_result, job, "**结果：**需要关注\n\n完整结果 END", run_status=status, elapsed_seconds=73, **kwargs,
    )
    assert error is None
    wire.assert_awaited_once()
    send = wire.call_args.kwargs
    assert send["msg_type"] == "interactive"
    card = json.loads(send["payload"])
    assert job["name"] in card["header"]["title"]["content"]
    assert card["header"]["template"] == ("red" if status == "error" else "blue")
    assert card["body"]["elements"][0]["content"] == "**结果：** 需要关注\n\n完整结果 END"
    text = json.dumps(card, ensure_ascii=False)
    assert "Cronjob Response" not in text
    assert "To stop or manage" not in text
    assert "job-test" in text and "1分 13秒" in text


@pytest.mark.asyncio
async def test_explicit_unwrapped_output_keeps_its_existing_plain_card(delivery, monkeypatch):
    _, job, wire = delivery
    monkeypatch.setattr("cron.scheduler.load_config", lambda: {"cron": {"wrap_response": False}})
    assert await asyncio.to_thread(_deliver_result, job, "EXACT CONTENT") is None
    card = json.loads(wire.call_args.kwargs["payload"])
    assert "header" not in card
    assert card["body"]["elements"][0]["content"] == "EXACT CONTENT"


@pytest.mark.asyncio
async def test_long_output_and_media_keep_notification_metadata(delivery, monkeypatch, tmp_path):
    _, job, wire = delivery
    media = tmp_path / "report.txt"
    media.write_text("attached report")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path,))
    upload = AsyncMock(return_value=SendResult(success=True, message_id="file"))
    monkeypatch.setattr(FeishuAdapter, "send_document", upload)
    content = "结果行\n" * 2500 + "UNIQUE_FINAL_TAIL"
    assert await asyncio.to_thread(_deliver_result, job, content + f"\nMEDIA:{media}") is None
    cards = [json.loads(call.kwargs["payload"]) for call in wire.call_args_list]
    assert len(cards) > 1
    assert all(job["name"] in card["header"]["title"]["content"] for card in cards)
    text = "\n".join(card["body"]["elements"][0]["content"] for card in cards)
    assert text.count("UNIQUE_FINAL_TAIL") == 1
    assert text.count("结果行") == 2500
    assert "MEDIA:" not in text
    upload.assert_awaited_once()


@pytest.mark.asyncio
async def test_rejected_card_falls_back_without_losing_report(delivery):
    _, job, wire = delivery
    response = wire.return_value
    wire.side_effect = [RuntimeError("invalid card schema"), response]
    assert await asyncio.to_thread(_deliver_result, job, "REPORT END") is None
    assert wire.call_args.kwargs["msg_type"] == "text"
    assert "REPORT END" in wire.call_args.kwargs["payload"]


def test_model_claims_cannot_promote_notification_to_success():
    card = build_cron_notification_card("✅ Business success!", {"name": "check", "status": "completed"})
    assert card["header"]["template"] == "blue"
    assert "成功" not in card["header"]["subtitle"]["content"]
