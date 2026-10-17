"""Feishu file-name / content-type helpers (extension, media type, chat type), shared as a mixin."""

from __future__ import annotations

import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from gateway.platforms.base import SUPPORTED_DOCUMENT_TYPES

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
_AUDIO_EXTENSIONS = {".ogg", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".opus", ".webm"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}


class FeishuMediaTypesMixin:
    @staticmethod
    def _get_audio_duration_ms(file_path: str) -> int:
        """OGG/Opus duration in ms (pure Python): last granule position / 48000 Hz; 0 on non-OGG or error."""
        import struct
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            pos = last_granule = 0
            while pos < len(data) - 27:
                pos = data.find(b"OggS", pos)
                if pos == -1 or pos + 27 > len(data):
                    break
                granule = struct.unpack_from("<q", data, pos + 6)[0]
                num_segments = data[pos + 26]
                last_granule = granule if granule > 0 else last_granule
                if pos + 27 + num_segments > len(data):
                    break
                pos += num_segments + sum(data[pos + 27 : pos + 27 + num_segments])
            return int(last_granule / 48000 * 1000) if last_granule > 0 else 0
        except Exception:
            return 0

    @staticmethod
    def _read_binary_response(response: Any) -> bytes:
        file_obj = getattr(response, "file", None)
        if file_obj is None:
            return b""
        if hasattr(file_obj, "getvalue"):
            return bytes(file_obj.getvalue())
        return bytes(file_obj.read())

    @staticmethod
    def _get_response_header(response: Any, name: str) -> str:
        raw = getattr(response, "raw", None)
        headers = getattr(raw, "headers", {}) or {}
        return str(headers.get(name, headers.get(name.lower(), "")) or "").split(";", 1)[0].strip().lower()

    @staticmethod
    def _guess_extension(filename: str, content_type: str, default: str, *, allowed: set[str]) -> str:
        ext = Path(filename or "").suffix.lower()
        if ext in allowed:
            return ext
        guessed = mimetypes.guess_extension((content_type or "").split(";", 1)[0].strip().lower() or "")
        return guessed if guessed in allowed else default

    @staticmethod
    def _normalize_media_type(content_type: str, *, default: str) -> str:
        normalized = (content_type or "").split(";", 1)[0].strip().lower()
        return normalized or default

    @staticmethod
    def _guess_document_media_type(filename: str) -> str:
        ext = Path(filename or "").suffix.lower()
        return SUPPORTED_DOCUMENT_TYPES.get(ext, mimetypes.guess_type(filename or "")[0] or "application/octet-stream")

    @staticmethod
    def _display_name_from_cached_path(path: str) -> str:
        basename = os.path.basename(path)
        parts = basename.split("_", 2)
        display_name = parts[2] if len(parts) >= 3 else basename
        return re.sub(r"[^\w.\- ]", "_", display_name)

    @staticmethod
    def _guess_media_type_from_filename(filename: str) -> str:
        guessed = (mimetypes.guess_type(filename or "")[0] or "").lower()
        if guessed:
            return guessed
        ext = Path(filename or "").suffix.lower()
        if ext in _VIDEO_EXTENSIONS:
            return f"video/{ext.lstrip('.')}"
        if ext in _AUDIO_EXTENSIONS:
            return f"audio/{ext.lstrip('.')}"
        if ext in _IMAGE_EXTENSIONS:
            return FeishuMediaTypesMixin._default_image_media_type(ext)
        return ""

    @staticmethod
    def _map_chat_type(raw_chat_type: str) -> str:
        normalized = (raw_chat_type or "").strip().lower()
        if normalized == "p2p":
            return "dm"
        if any(marker in normalized for marker in ("topic", "thread", "forum")):
            return "forum"
        return "group" if normalized == "group" else "dm"

    @staticmethod
    def _resolve_source_chat_type(*, chat_info: dict[str, Any], event_chat_type: str) -> str:
        resolved = str(chat_info.get("type") or "").strip().lower()
        if resolved in {"group", "forum"}:
            return resolved
        return "dm" if event_chat_type == "p2p" else "group"

    @staticmethod
    def _default_image_media_type(ext: str) -> str:
        normalized_ext = (ext or "").lower()
        if normalized_ext in {".jpg", ".jpeg"}:
            return "image/jpeg"
        return f"image/{normalized_ext.lstrip('.') or 'jpeg'}"
