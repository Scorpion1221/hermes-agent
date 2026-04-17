from pathlib import Path

from gateway.platforms.feishu_inbound import media_index as idx


def test_put_and_get_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(idx, '_INDEX_PATH', tmp_path / 'cache' / 'feishu_media_index.json')
    media = tmp_path / 'cache' / 'img.png'
    media.parent.mkdir(parents=True)
    media.write_bytes(b'png')

    idx.put_feishu_media_index_entry(
        message_id='om1', file_key='img1', cached_path=str(media), content_type='image/png', resource_type='image'
    )
    entry = idx.get_feishu_media_index_entry('om1', 'img1')
    assert entry is not None
    assert entry.cached_path == str(media)
    assert entry.content_type == 'image/png'


def test_missing_file_is_pruned(tmp_path, monkeypatch):
    monkeypatch.setattr(idx, '_INDEX_PATH', tmp_path / 'cache' / 'feishu_media_index.json')
    idx.put_feishu_media_index_entry(
        message_id='om2', file_key='img2', cached_path=str(tmp_path / 'missing.png'), content_type='image/png', resource_type='image'
    )
    assert idx.get_feishu_media_index_entry('om2', 'img2') is None


def test_remove_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(idx, '_INDEX_PATH', tmp_path / 'cache' / 'feishu_media_index.json')
    media = tmp_path / 'cache' / 'img.png'
    media.parent.mkdir(parents=True)
    media.write_bytes(b'png')
    idx.put_feishu_media_index_entry(message_id='om3', file_key='img3', cached_path=str(media))
    assert idx.remove_feishu_media_index_entry('om3', 'img3') is True
    assert idx.get_feishu_media_index_entry('om3', 'img3') is None
