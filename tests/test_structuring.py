"""텍스트 정형화 및 공종 분류 모듈 단위 테스트."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.structuring.engine import BaseStructuringEngine
from src.structuring.prompt import StructuringPromptBuilder
from src.structuring.repository import (
    BaseWorkTypeRepository,
    LocalJsonWorkTypeRepository,
)
from src.structuring.schemas import (
    InputRecord,
    StructuredItem,
    StructuredRecord,
    StructuringResult,
    WorkTypeItem,
)
from src.structuring.service import BaseResponseHandler, StructuringService


class TestWorkTypeRepository:
    """WorkType 저장소 테스트."""

    def test_local_json_repository_load(self):
        repo = LocalJsonWorkTypeRepository()
        items = repo.get_work_types()
        assert len(items) >= 20

        # ID 조회 테스트
        item_101 = repo.get_by_id(101)
        assert item_101 is not None
        assert item_101.name == "금속관벽체"

        # 이름 조회 테스트
        item_duct = repo.get_by_name("보온덕트벽체")
        assert item_duct is not None
        assert item_duct.id == 201

    def test_nonexistent_id_or_name(self):
        repo = LocalJsonWorkTypeRepository()
        assert repo.get_by_id(99999) is None
        assert repo.get_by_name("존재하지않는공종") is None


class TestPromptBuilder:
    """프롬프트 빌더 테스트."""

    def test_build_system_instruction(self):
        work_types = [
            WorkTypeItem(id=101, name="금속관벽체"),
            WorkTypeItem(id=201, name="보온덕트벽체"),
        ]
        prompt = StructuringPromptBuilder.build_system_instruction(work_types)
        assert "ID 101: 금속관벽체" in prompt
        assert "ID 201: 보온덕트벽체" in prompt
        assert "시공면 계수" in prompt
        assert "Few-Shot 예시" in prompt

    def test_build_user_content(self):
        records = [
            InputRecord(text="보 50 양면", location="3동 38층", workDate="2024-06-28"),
            InputRecord(text="무 1600*500 단면"),
        ]
        user_content = StructuringPromptBuilder.build_user_content(records)
        parsed = json.loads(user_content)
        assert len(parsed) == 2
        assert parsed[0]["text"] == "보 50 양면"
        assert parsed[0]["location"] == "3동 38층"
        assert parsed[0]["workDate"] == "2024-06-28"
        assert parsed[1]["text"] == "무 1600*500 단면"
        assert "location" not in parsed[1]


class TestSchemas:
    """Pydantic 스키마 및 DTO 유효성 테스트."""

    def test_structured_record_validation(self):
        record_data = {
            "location": "지하4",
            "workDate": "2024-06-28",
            "items": [
                {"matchedWorkTypeId": 101, "spec": "65", "quantity": 2.0},
                {"matchedWorkTypeId": 301, "spec": "1600*600", "quantity": 1.0},
            ],
        }
        record = StructuredRecord.model_validate(record_data)
        assert record.location == "지하4"
        assert record.workDate == "2024-06-28"
        assert len(record.items) == 2
        assert record.items[0].matchedWorkTypeId == 101
        assert record.items[0].spec == "65"
        assert record.items[0].quantity == 2.0


class TestStructuringService:
    """StructuringService 통합 흐름 테스트 (Mock Engine 활용)."""

    def test_service_process_with_mock_engine(self):
        mock_engine = MagicMock(spec=BaseStructuringEngine)
        mock_result = StructuringResult(
            success=True,
            records=[
                StructuredRecord(
                    location="지하4",
                    workDate="2024-06-28",
                    items=[
                        StructuredItem(matchedWorkTypeId=101, spec="65", quantity=2.0)
                    ],
                )
            ],
            execution_time_sec=0.1,
            model_used="gemini-3.7-flash",
        )
        mock_engine.process.return_value = mock_result

        service = StructuringService(engine=mock_engine)
        result = service.process(
            {"text": "2-벽체D65*2 양면", "location": "지하4", "workDate": "2024-06-28"}
        )

        assert result.success is True
        assert len(result.records) == 1
        assert result.records[0].items[0].matchedWorkTypeId == 101
        assert result.records[0].items[0].spec == "65"
        assert result.records[0].items[0].quantity == 2.0
        mock_engine.process.assert_called_once()

    def test_response_handler_callback(self):
        mock_engine = MagicMock(spec=BaseStructuringEngine)
        mock_result = StructuringResult(success=True, records=[])
        mock_engine.process.return_value = mock_result

        mock_handler = MagicMock(spec=BaseResponseHandler)

        service = StructuringService(
            engine=mock_engine, response_handler=mock_handler
        )
        result = service.process("테스트 텍스트")

        assert result.success is True
        mock_handler.handle_response.assert_called_once_with(mock_result)
