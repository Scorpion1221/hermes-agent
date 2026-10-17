"""Packaged ``.skill`` artifacts and emphasis-wrapped ``MEDIA:`` tags reach the user as files, not literal text.

Models present a file in bold (``**MEDIA:/x/pkg.skill**``) or name it in prose without a tag
(``packaged at /x/pkg.skill``). A ``.skill`` package is a deliverable like an archive; a tag for a
type outside the extension list still delivers only once the path validates on disk, wrapped in
emphasis or not, and leaves no stray ``**`` behind.
"""

from __future__ import annotations

import pytest

from gateway.platforms.base import BasePlatformAdapter


@pytest.fixture
def out(tmp_path, monkeypatch):
    root = tmp_path / "output"
    root.mkdir()
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (str(root),))
    monkeypatch.delenv("HERMES_MEDIA_DELIVERY_STRICT", raising=False)
    return root


def _file(root, name):
    path = root / name
    path.write_text("content", encoding="utf-8")
    return path


@pytest.mark.parametrize("template", ["MEDIA:{}", "**MEDIA:{}**", "*MEDIA:{}*", "_MEDIA:{}_", "`MEDIA:{}`"])
def test_a_skill_package_tag_delivers_the_file(out, template):
    package = _file(out, "solvely-config-deploy.skill")

    media, cleaned = BasePlatformAdapter.extract_media(f"打包好了\n\n{template.format(package)}")

    assert [path for path, _voice in media] and media[0][0].endswith("solvely-config-deploy.skill")
    assert cleaned == "打包好了"


def test_a_skill_package_named_in_prose_is_attached(out):
    package = _file(out, "pkg.skill")

    paths, cleaned = BasePlatformAdapter.extract_local_files(f"The package is at {package} for you")

    assert paths == [str(package)]
    assert cleaned == "The package is at  for you"


@pytest.mark.parametrize("name", ["build.py", "my_file.py", "trailing_.log", "Caddyfile"])
@pytest.mark.parametrize("wrap", ["**", "*", "_", "__"])
def test_an_emphasis_wrapped_tag_for_a_validated_file_delivers_without_leftover_marks(out, name, wrap):
    path = _file(out, name)
    text = f"Saved {wrap}MEDIA:{path}{wrap}. Done"

    media, cleaned = BasePlatformAdapter.extract_media(text)

    assert [p for p, _voice in media] == [str(path.resolve())]
    assert cleaned == "Saved . Done"
    assert BasePlatformAdapter.strip_media_directives_for_display(text) == "Saved . Done"


def test_an_underscore_inside_the_file_name_is_part_of_the_path(out):
    path = _file(out, "my_file_.py")

    media, cleaned = BasePlatformAdapter.extract_media(f"MEDIA:{path} sent")

    assert [p for p, _voice in media] == [str(path.resolve())]
    assert cleaned == "sent"


def test_an_emphasis_wrapped_tag_for_a_missing_file_stays_visible(out):
    text = f"**MEDIA:{out}/missing.py**"

    assert BasePlatformAdapter.extract_media(text) == ([], text)
    assert BasePlatformAdapter.strip_media_directives_for_display(text) == text
