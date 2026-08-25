"""AI 서버 엔드포인트 및 통합 파이프라인 테스트 모듈."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient
import numpy as np

from ai.src.ocr.schemas import BoardTableItem, OCRResult
from ai.src.server.main import app
from ai.src.server.service import pipeline_service
from ai.src.structuring.schemas import StructuredItem, StructuredRecord, StructuringResult
from ai.src.table_crop.cropper import CropResult


@pytest.fixture
def client():
    """테스트용 FastAPI 클라이언트 fixture."""
    return TestClient(app)


def test_health_check(client):
    """서버 헬스체크 엔드포인트 테스트."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "fieldnote-ai-server"
    assert data["mode"] in ["stub", "integrated"]


def test_analyze_photo_integrated_flow(client):
    """사진 분석 통합 엔드포인트 요청 및 응답 스키마 검증 (Mock 기반)."""
    original_stub = pipeline_service.is_stub
    pipeline_service.is_stub = False
    fake_crop_img = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_crop_res = CropResult(
        cropped_image=fake_crop_img,
        bbox=(10, 10, 80, 80),
        original_shape=(100, 100, 3),
        confidence=0.95,
        is_fallback=False,
        source_id="test_src",
        file_name="test.jpg",
        metadata={},
    )
    mock_ocr_res = OCRResult(
        source_id="test_src",
        file_name="test.jpg",
        success=True,
        data=BoardTableItem(
            공사명="힐스테이트 신축공사",
            공종="금속관벽체",
            위치="101동 3층",
            내용="벽체D65*2 양면 시공",
            일자="2024-06-28",
        ),
        raw_response='{"공종": "금속관벽체"}',
        execution_time_sec=1.5,
    )
    mock_struct_res = StructuringResult(
        success=True,
        records=[
            StructuredRecord(
                location="101동 3층",
                workDate="2024-06-28",
                items=[
                    StructuredItem(
                        matchedWorkTypeId=103,
                        workType="금속관벽체",
                        spec="65",
                        quantity=2.0,
                    )
                ],
            )
        ],
        execution_time_sec=1.2,
    )

    try:
        with patch.object(
            pipeline_service, "download_image", new=AsyncMock(return_value=b"fake_bytes")
        ), patch.object(
            pipeline_service.cropper, "crop", return_value=mock_crop_res
        ), patch.object(
            pipeline_service.ocr_engine, "process", return_value=mock_ocr_res
        ), patch.object(
            pipeline_service.structuring_engine, "process", return_value=mock_struct_res
        ):

            payload = {
                "image_url": "https://fieldnote-bucket.s3.ap-northeast-2.amazonaws.com/photos/sample_01.jpg",
                "task_id": "test_task_123",
                "work_types": [
                    {"id": 103, "name": "금속관벽체"},
                    {"id": 104, "name": "금속관천정"},
                ],
            }
            response = client.post("/api/v1/analyze", json=payload)
            assert response.status_code == 200

            data = response.json()
            assert data["success"] is True
            assert data["task_id"] == "test_task_123"
            assert "ocr_raw_text" in data["data"]
            assert "record" in data["data"]

            record = data["data"]["record"]
            assert record["location"] == "101동 3층"
            assert record["workDate"] == "2024-06-28"
            assert len(record["items"]) == 1

            first_item = record["items"][0]
            assert first_item["matchedWorkTypeId"] == 103
            assert first_item["spec"] == "65"
            assert first_item["quantity"] == 2.0
    finally:
        pipeline_service.is_stub = original_stub


def test_analyze_invalid_url(client):
    """잘못된 URL 요청 시 400 에러 반환 검증."""
    payload = {
        "image_url": "invalid_url_without_http",
        "task_id": "test_task_fail",
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["code"] == "INVALID_IMAGE_URL"


def test_analyze_camel_case_payload(client):
    """카멜케이스(imageUrl, workTypes 등) 페이로드 파싱 지원 검증."""
    original_stub = pipeline_service.is_stub
    original_delay = pipeline_service.stub_delay_sec
    pipeline_service.is_stub = True
    pipeline_service.stub_delay_sec = 0.01
    try:
        payload = {
            "imageUrl": "https://fieldnote-bucket.s3.ap-northeast-2.amazonaws.com/photos/sample_01.jpg",
            "taskId": "task_camel_001",
            "workTypes": [
                {"workTypeId": 201, "workTypeName": "철근배근"},
                {"id": 202, "name": "콘크리트타설"},
            ],
        }
        response = client.post("/api/v1/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["task_id"] == "task_camel_001"
        assert data["data"]["record"]["items"][0]["matchedWorkTypeId"] == 201
    finally:
        pipeline_service.is_stub = original_stub
        pipeline_service.stub_delay_sec = original_delay
