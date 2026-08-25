"""Unit tests for OCR Pipeline."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from ocr.engine import BaseOCREngine
from ocr.loader import FileImageLoader
from ocr.pipeline import OCRPipeline
from ocr.saver import MemoryOCRResultSaver
from ocr.schemas import BoardTableItem, LoadedImageData, OCRResult


class FakeEngine(BaseOCREngine):
    def process(self, image_data: LoadedImageData) -> OCRResult:
        return OCRResult(
            source_id=image_data.source_id,
            file_name=image_data.file_name,
            success=True,
            data=BoardTableItem(공사명="테스트 공사", 공종="마감"),
        )


def test_pipeline_with_memory_saver(tmp_path: Path):
    dummy_img = tmp_path / "board_test.png"
    dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n")

    loader = FileImageLoader()
    engine = FakeEngine()
    saver = MemoryOCRResultSaver()

    pipeline = OCRPipeline(loader=loader, engine=engine, saver=saver)
    result = pipeline.process_image(dummy_img, output_dir=tmp_path)

    assert result.success is True
    assert result.data.공사명 == "테스트 공사"

    saved = saver.get_results()
    assert len(saved) == 1
    assert saved[0]["data"]["공사명"] == "테스트 공사"


def test_pipeline_batch_progress(tmp_path: Path):
    (tmp_path / "img1.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "img2.jpg").write_bytes(b"\xff\xd8\xff")

    loader = FileImageLoader()
    engine = FakeEngine()
    saver = MemoryOCRResultSaver()

    progress_calls = []

    def on_progress(idx, total, res):
        progress_calls.append((idx, total, res.file_name))

    pipeline = OCRPipeline(loader=loader, engine=engine, saver=saver)
    results = pipeline.process_directory(tmp_path, on_progress=on_progress)

    assert len(results) == 2
    assert len(progress_calls) == 2
    assert progress_calls[0] == (1, 2, "img1.jpg")
    assert progress_calls[1] == (2, 2, "img2.jpg")
