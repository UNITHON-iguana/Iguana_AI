"""통합 서버 파이프라인 단위 및 엔드투엔드 테스트 모듈."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient
import numpy as np

from ai.src.ocr.schemas import BoardTableItem, OCRResult
from ai.src.server.main import app
from ai.src.server.schemas import WorkTypeInputItem
from ai.src.server.service import PipelineService, pipeline_service
from ai.src.structuring.schemas import StructuredItem, StructuredRecord, StructuringResult
from ai.src.table_crop.cropper import CropResult


@pytest.fixture
def client():
    """테스트용 FastAPI 클라이언트 fixture."""
    return TestClient(app)


def test_health_check(client):
    """서버 헬스체크 엔드포인트 테스트."""
    original_stub = pipeline_service.is_stub
    pipeline_service.is_stub = False
    try:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["mode"] == "integrated"
    finally:
        pipeline_service.is_stub = original_stub


@pytest.mark.asyncio
async def test_pipeline_service_orchestration_mocked():
    """하위 모듈(table_crop, ocr, structuring)이 정상적으로 오케스트레이션되는지 검증."""
    # Mock 컴포넌트 생성
    mock_cropper = MagicMock()
    fake_crop_img = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cropper.crop.return_value = CropResult(
        cropped_image=fake_crop_img,
        bbox=(10, 10, 80, 80),
        original_shape=(100, 100, 3),
        confidence=0.95,
        is_fallback=False,
        source_id="test_src",
        file_name="test.jpg",
        metadata={},
    )

    mock_ocr = MagicMock()
    mock_ocr.process.return_value = OCRResult(
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

    mock_struct = MagicMock()
    mock_struct.process.return_value = StructuringResult(
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

    service = PipelineService(
        cropper=mock_cropper,
        ocr_engine=mock_ocr,
        structuring_engine=mock_struct,
        is_stub=False,
    )

    # download_image 모킹
    service.download_image = AsyncMock(return_value=b"fake_jpeg_bytes_data")

    # 서비스 호출
    from ai.src.server.schemas import AnalyzeRequest

    request = AnalyzeRequest(
        image_url="https://s3.amazonaws.com/bucket/photo.jpg",
        task_id="task_mock_01",
        work_types=[
            WorkTypeInputItem(id=103, name="금속관벽체"),
            WorkTypeInputItem(id=104, name="금속관천정"),
        ],
    )

    response = await service.analyze(request)

    assert response.success is True
    assert response.task_id == "task_mock_01"
    assert response.data is not None
    assert "2024-06-28" in response.data.ocr_raw_text
    assert response.data.record.location == "101동 3층"
    assert response.data.record.workDate == "2024-06-28"
    assert len(response.data.record.items) == 1
    assert response.data.record.items[0].matchedWorkTypeId == 103
    assert response.data.record.items[0].spec == "65"
    assert response.data.record.items[0].quantity == 2.0


def test_analyze_download_failed(client):
    """이미지 다운로드 실패 시 200 OK 내에 success: False와 에러 상세가 반환되는지 검증."""
    original_stub = pipeline_service.is_stub
    original_timeout = pipeline_service.download_timeout_sec
    pipeline_service.is_stub = False
    pipeline_service.download_timeout_sec = 0.5
    try:
        payload = {
            "image_url": "https://invalid-non-existent-domain-12345.com/photo.jpg",
            "task_id": "test_download_fail",
            "work_types": [{"id": 101, "name": "내화충전"}],
        }
        response = client.post("/api/v1/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "IMAGE_DOWNLOAD_FAILED"
    finally:
        pipeline_service.is_stub = original_stub
        pipeline_service.download_timeout_sec = original_timeout
