"""Unit tests for OCR result savers."""

import json
from pathlib import Path
import pytest
from ocr.saver import JsonFileOCRResultSaver, MemoryOCRResultSaver
from ocr.schemas import BoardTableItem, OCRResult


def test_json_file_saver_preserves_unicode_and_nulls(tmp_path: Path):
    saver = JsonFileOCRResultSaver()
    item = BoardTableItem(
        공사명="마포 래미안 신축",
        공종="철근콘크리트공사",
        위치="지상 10층 슬래브",
        내용="철근 배근 간격 체크 (Φ13@200, D16~D25, Ø300 배관 관통)",
        일자=None,
    )
    result = OCRResult(
        source_id=str(tmp_path / "board.jpg"),
        file_name="board.jpg",
        success=True,
        data=item,
        raw_response='{"raw": "test"}',
        execution_time_sec=0.25,
        model_used="gemini-3.5-flash-lite",
    )

    summary = saver.save(result, destination=tmp_path)
    assert summary.success is True
    
    saved_file = tmp_path / "board_ocr.json"
    assert saved_file.is_file()

    content = saved_file.read_text(encoding="utf-8")
    # Verify special characters preserved without \u escape
    assert "Φ13@200" in content
    assert "Ø300" in content
    assert '"일자": null' in content

    # Verify JSON load
    loaded = json.loads(content)
    assert loaded["data"]["공종"] == "철근콘크리트공사"
    assert loaded["data"]["일자"] is None


def test_json_file_saver_batch(tmp_path: Path):
    saver = JsonFileOCRResultSaver()
    r1 = OCRResult(source_id="1", file_name="f1.jpg", success=True, data=BoardTableItem(공사명="A"))
    r2 = OCRResult(source_id="2", file_name="f2.jpg", success=False, error_message="fail")

    summaries = saver.save_batch([r1, r2], destination=tmp_path)
    assert len(summaries) == 3  # 2 individual + 1 batch summary
    assert (tmp_path / "batch_ocr_summary.json").is_file()


def test_memory_saver():
    saver = MemoryOCRResultSaver()
    r1 = OCRResult(source_id="1", file_name="f1.jpg", success=True, data=BoardTableItem(공사명="A"))
    saver.save(r1)

    results = saver.get_results()
    assert len(results) == 1
    assert results[0]["data"]["공사명"] == "A"
