"""A forced plugin re-discovery must not re-import a platform package a live adapter is running.

``hermes plugins install/enable`` nudges the running gateway (``reload-plugins``), which calls
``discover_and_load(force=True)``. That unloads every plugin and imports each one again. For a
platform whose adapter is already connected, re-importing its package replaces the module that
adapter's code resolves names from (``from . import adapter``, ``sys.modules[__package__ +
".adapter"]``) with a fresh copy whose lazily bound SDK names are unset again: the connected
adapter never binds a second time. On 2026-10-21 that turned every Feishu card redraw and quote
fetch into ``'SimpleNamespace' object has no attribute 'token_types'`` until a restart.

The re-discovery must reuse the package the process already runs, through the real discovery and
deferred-registry path.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest


_PLATFORM = "reloadplat"


def _adapter_modules():
    return [m for n, m in sys.modules.items() if n.startswith("hermes_plugins.") and n.endswith(f"{_PLATFORM}_platform.adapter")]


def _write_platform(root: Path) -> None:
    """A bundled-style platform whose SDK name is bound lazily, as feishu's ``_load_lark_oapi`` does."""
    pkg = root / _PLATFORM
    pkg.mkdir(parents=True)
    (pkg / "plugin.yaml").write_text(f"name: {_PLATFORM}-platform\nkind: platform\nversion: 1.0.0\n")
    (pkg / "__init__.py").write_text("from .adapter import register\n")
    (pkg / "adapter.py").write_text(textwrap.dedent(f'''
        SDK = None  # bound on connect, never at import

        def bind_sdk():
            globals()["SDK"] = "bound"

        def sdk_seen_by_live_code():
            from . import adapter as facade  # how the feishu helpers resolve the SDK names
            return facade.SDK

        def register(ctx):
            ctx.register_platform(name="{_PLATFORM}", label="Reload probe", adapter_factory=lambda cfg: None,
                                  check_fn=lambda: True)
    '''))


@pytest.fixture
def bundled_platform(tmp_path):
    from gateway.platform_registry import platform_registry

    root = tmp_path / "bundled_platforms"
    _write_platform(root)
    before = set(sys.modules)
    yield root
    platform_registry.unregister(_PLATFORM)
    for name in set(sys.modules) - before:
        if name.startswith("hermes_plugins."):
            sys.modules.pop(name, None)


def _manifest(root: Path):
    from hermes_cli.plugins import PluginManifest
    return PluginManifest(name=f"{_PLATFORM}-platform", kind="platform", source="bundled",
                          path=str(root / _PLATFORM), key=f"{_PLATFORM}-platform")


def test_force_rediscovery_keeps_the_running_platform_module(bundled_platform, monkeypatch):
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import PluginManager

    root = bundled_platform
    mgr = PluginManager()
    # Discovery of this one bundled platform, through the same deferred registration discover_and_load uses.
    monkeypatch.setattr(mgr, "_discover_and_load_inner", lambda: mgr._register_deferred_platform(_manifest(root)))

    mgr.discover_and_load()
    assert platform_registry.get(_PLATFORM) is not None  # first use materializes the adapter
    [running] = _adapter_modules()
    running.bind_sdk()  # the adapter connected: SDK names bound once, for the life of the process
    assert running.sdk_seen_by_live_code() == "bound"

    mgr.discover_and_load(force=True)  # `hermes plugins install --enable` -> reload-plugins
    assert platform_registry.get(_PLATFORM) is not None  # still registered after the sweep

    assert _adapter_modules() == [running], "force re-discovery re-imported a platform package a live adapter uses"
    assert running.sdk_seen_by_live_code() == "bound"


def test_force_rediscovery_still_reimports_an_unmaterialized_platform(bundled_platform, monkeypatch):
    """Only a package the process already imported is reused: a platform never materialized stays lazy."""
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import PluginManager

    root = bundled_platform
    mgr = PluginManager()
    monkeypatch.setattr(mgr, "_discover_and_load_inner", lambda: mgr._register_deferred_platform(_manifest(root)))
    mgr.discover_and_load()
    mgr.discover_and_load(force=True)
    assert _adapter_modules() == []  # still deferred, nothing imported
    assert platform_registry.get(_PLATFORM) is not None
