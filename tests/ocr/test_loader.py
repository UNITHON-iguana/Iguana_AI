"""Unit tests for OCR image loaders."""

import base64
from pathlib import Path
import pytest
from ocr.loader import BytesImageLoader, FileImageLoader


def test_file_image_loader(tmp_path: Path):
    loader = FileImageLoader()
    dummy_img = tmp_path / "board_sample.jpg"
    dummy_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00"  # fake JPEG header
    dummy_img.write_bytes(dummy_bytes)

    loaded = loader.load(dummy_img)
    assert loaded.file_name == "board_sample.jpg"
    assert loaded.mime_type == "image/jpeg"
    assert loaded.image_bytes == dummy_bytes
    assert loaded.size_bytes == len(dummy_bytes)


def test_file_image_loader_directory(tmp_path: Path):
    loader = FileImageLoader()
    (tmp_path / "img1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "img2.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "ignore.txt").write_text("not an image")

    loaded_list = loader.load_directory(tmp_path)
    assert len(loaded_list) == 2
    assert {img.file_name for img in loaded_list} == {"img1.png", "img2.jpg"}


def test_bytes_image_loader_raw():
    loader = BytesImageLoader()
    raw = b"raw_test_image_content"
    loaded = loader.load(raw, mime_type="image/png", file_name="custom.png")
    assert loaded.image_bytes == raw
    assert loaded.mime_type == "image/png"
    assert loaded.file_name == "custom.png"


def test_bytes_image_loader_base64():
    loader = BytesImageLoader()
    raw = b"base64_encoded_content"
    b64_str = base64.b64encode(raw).decode("utf-8")
    loaded = loader.load(b64_str, mime_type="image/jpeg")
    assert loaded.image_bytes == raw
    assert loaded.mime_type == "image/jpeg"


def test_bytes_image_loader_data_uri():
    loader = BytesImageLoader()
    raw = b"data_uri_content"
    b64_str = base64.b64encode(raw).decode("utf-8")
    data_uri = f"data:image/webp;base64,{b64_str}"
    loaded = loader.load(data_uri)
    assert loaded.image_bytes == raw
    assert loaded.mime_type == "image/webp"
