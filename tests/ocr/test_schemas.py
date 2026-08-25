"""Unit tests for OCR schemas."""

import json
import pytest
from ocr.schemas import BoardTableItem, LoadedImageData, OCRResult


def test_board_table_item_all_fields_present():
    item = BoardTableItem(
        공사명="OO아파트 신축공사",
        공종="골조공사",
        위치="101동 3층",
        내용="슬래브 철근 배근(D10, D13, Φ16)",
        일자="2026-08-25",
    )
    data = item.model_dump()
    assert data["공사명"] == "OO아파트 신축공사"
    assert data["내용"] == "슬래브 철근 배근(D10, D13, Φ16)"
    assert "Φ16" in data["내용"]


def test_board_table_item_null_fields_retained():
    # If fields are not provided or unidentifiable, they must serialize as null
    item = BoardTableItem(
        공사명="OO현장",
        공종=None,
        위치=None,
        내용="타설 작업",
        일자=None,
    )
    dumped = item.model_dump()
    assert dumped["공종"] is None
    assert dumped["위치"] is None
    assert dumped["일자"] is None
    assert dumped["공사명"] == "OO현장"
    assert dumped["내용"] == "타설 작업"

    # Verify JSON serialization has null values
    json_str = json.dumps(dumped, ensure_ascii=False)
    parsed = json.loads(json_str)
    assert parsed["공종"] is None
    assert "공종" in parsed
    assert "위치" in parsed
    assert "일자" in parsed


def test_ocr_result_to_dict():
    item = BoardTableItem(
        공사명="테스트공사",
        공종=None,
        위치="지하 1층",
        내용="방수공사",
        일자="2026-08-25",
    )
    res = OCRResult(
        source_id="/path/to/img.jpg",
        file_name="img.jpg",
        success=True,
        data=item,
        raw_response='{"공사명": "테스트공사"}',
        execution_time_sec=0.1234,
        model_used="gemini-3.5-flash-lite",
    )
    d = res.to_dict()
    assert d["success"] is True
    assert d["data"]["공사명"] == "테스트공사"
    assert d["data"]["공종"] is None
    assert d["execution_time_sec"] == 0.123
    assert d["model_used"] == "gemini-3.5-flash-lite"
