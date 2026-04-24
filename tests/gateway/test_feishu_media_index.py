import importlib.util
import json
from pathlib import Path
import sys
import uuid

from gateway.platforms.feishu_inbound import media_index as idx


def _load_isolated_media_index_module(alias: str):
    spec = importlib.util.spec_from_file_location(alias, Path(idx.__file__))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def test_put_and_get_round_trip_persists_normalized_entry(tmp_path, monkeypatch):
    index_path = tmp_path / 'cache' / 'feishu_media_index.json'
    monkeypatch.setattr(idx, '_INDEX_PATH', index_path)
    media = tmp_path / 'cache' / 'img.png'
    media.parent.mkdir(parents=True)
    media.write_bytes(b'png')

    stored = idx.put_feishu_media_index_entry(
        message_id=' om1 ',
        file_key=' img1 ',
        cached_path=f' {media} ',
        content_type=' image/png ',
        resource_type=' image ',
    )
    entry = idx.get_feishu_media_index_entry('om1', 'img1')

    persisted = json.loads(index_path.read_text(encoding='utf-8'))
    assert stored.message_id == 'om1'
    assert stored.file_key == 'img1'
    assert stored.cached_path == str(media)
    assert stored.content_type == 'image/png'
    assert stored.resource_type == 'image'
    assert entry is not None
    assert entry.key == idx.make_media_index_key('feishu', 'om1', 'img1')
    assert entry.cached_path == str(media)
    assert entry.content_type == 'image/png'
    assert persisted == {
        entry.key: {
            'cached_path': str(media),
            'content_type': 'image/png',
            'file_key': 'img1',
            'message_id': 'om1',
            'platform': 'feishu',
            'resource_type': 'image',
            'updated_at': stored.updated_at,
        }
    }


def test_missing_entry_is_a_clean_miss_without_creating_persistent_state(tmp_path, monkeypatch):
    index_path = tmp_path / 'cache' / 'feishu_media_index.json'
    monkeypatch.setattr(idx, '_INDEX_PATH', index_path)

    assert idx.get_feishu_media_index_entry('missing', 'file') is None
    assert not index_path.exists()


def test_missing_file_is_pruned_from_disk(tmp_path, monkeypatch):
    index_path = tmp_path / 'cache' / 'feishu_media_index.json'
    monkeypatch.setattr(idx, '_INDEX_PATH', index_path)
    idx.put_feishu_media_index_entry(
        message_id='om2', file_key='img2', cached_path=str(tmp_path / 'missing.png'), content_type='image/png', resource_type='image'
    )
    assert idx.make_media_index_key('feishu', 'om2', 'img2') in json.loads(index_path.read_text(encoding='utf-8'))
    assert idx.get_feishu_media_index_entry('om2', 'img2') is None
    assert json.loads(index_path.read_text(encoding='utf-8')) == {}


def test_remove_entry_updates_persistent_storage_and_reports_missing_keys(tmp_path, monkeypatch):
    index_path = tmp_path / 'cache' / 'feishu_media_index.json'
    monkeypatch.setattr(idx, '_INDEX_PATH', index_path)
    media = tmp_path / 'cache' / 'img.png'
    media.parent.mkdir(parents=True)
    media.write_bytes(b'png')
    idx.put_feishu_media_index_entry(message_id='om3', file_key='img3', cached_path=str(media))

    assert idx.remove_feishu_media_index_entry('om3', 'img3') is True
    assert idx.remove_feishu_media_index_entry('om3', 'img3') is False
    assert idx.get_feishu_media_index_entry('om3', 'img3') is None
    assert json.loads(index_path.read_text(encoding='utf-8')) == {}


def test_stale_media_entry_is_pruned_after_restart_when_cached_file_disappears(tmp_path, monkeypatch):
    hermes_home = tmp_path / 'hermes-home'
    monkeypatch.setenv('HERMES_HOME', str(hermes_home))

    media = hermes_home / 'cache' / 'voice.m4a'
    media.parent.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b'audio')

    writer = _load_isolated_media_index_module(f'test_media_index_stale_writer_{uuid.uuid4().hex}')
    writer.put_feishu_media_index_entry(
        message_id='om-stale',
        file_key='aud-stale',
        cached_path=str(media),
        content_type='audio/m4a',
        resource_type='audio',
    )
    index_path = writer._INDEX_PATH
    assert json.loads(index_path.read_text(encoding='utf-8'))

    media.unlink()

    reader = _load_isolated_media_index_module(f'test_media_index_stale_reader_{uuid.uuid4().hex}')
    assert reader.get_feishu_media_index_entry('om-stale', 'aud-stale') is None
    assert json.loads(index_path.read_text(encoding='utf-8')) == {}


def test_media_index_reuses_entries_across_module_restarts(tmp_path, monkeypatch):
    hermes_home = tmp_path / 'hermes-home'
    monkeypatch.setenv('HERMES_HOME', str(hermes_home))

    media = hermes_home / 'cache' / 'voice.m4a'
    media.parent.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b'audio')

    writer = _load_isolated_media_index_module(f'test_media_index_writer_{uuid.uuid4().hex}')
    written = writer.put_feishu_media_index_entry(
        message_id='om4',
        file_key='aud4',
        cached_path=str(media),
        content_type='audio/m4a',
        resource_type='audio',
    )

    reader = _load_isolated_media_index_module(f'test_media_index_reader_{uuid.uuid4().hex}')
    entry = reader.get_feishu_media_index_entry('om4', 'aud4')

    assert writer._INDEX_PATH == hermes_home / 'cache' / 'feishu_media_index.json'
    assert reader._INDEX_PATH == writer._INDEX_PATH
    assert entry is not None
    assert entry.cached_path == str(media)
    assert entry.content_type == 'audio/m4a'
    assert entry.resource_type == 'audio'
    assert entry.updated_at == written.updated_at

    assert reader.remove_feishu_media_index_entry('om4', 'aud4') is True

    after_restart = _load_isolated_media_index_module(f'test_media_index_after_{uuid.uuid4().hex}')
    assert after_restart.get_feishu_media_index_entry('om4', 'aud4') is None
