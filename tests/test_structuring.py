"""텍스트 정형화 및 공종 분류 모듈 단위 테스트."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    from ai.src.structuring.engine import (
        BaseStructuringEngine,
        GeminiStructuringEngine,
    )
    from ai.src.structuring.prompt import StructuringPromptBuilder
    from ai.src.structuring.repository import (
        BaseWorkTypeRepository,
        LocalJsonWorkTypeRepository,
    )
    from ai.src.structuring.schemas import (
        InputRecord,
        LLMStructuredItem,
        LLMStructuredRecord,
        LLMStructuringBatchResponse,
        StructuredItem,
        StructuredRecord,
        StructuringResult,
        WorkTypeItem,
    )
    from ai.src.structuring.service import BaseResponseHandler, StructuringService
except ImportError:
    from src.structuring.engine import (
        BaseStructuringEngine,
        GeminiStructuringEngine,
    )
    from src.structuring.prompt import StructuringPromptBuilder
    from src.structuring.repository import (
        BaseWorkTypeRepository,
        LocalJsonWorkTypeRepository,
    )
    from src.structuring.schemas import (
        InputRecord,
        LLMStructuredItem,
        LLMStructuredRecord,
        LLMStructuringBatchResponse,
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
        # 보수적 파싱 및 환각 방지 지침 검증
        assert "잘못된 정보 출력 < 미출력" in prompt
        assert "items: []" in prompt

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

    def test_llm_structured_record_validation(self):
        record_data = {
            "location": "지하4",
            "workDate": "2024-06-28",
            "items": [
                {
                    "matchedWorkTypeId": 101,
                    "workType": "금속관벽체",
                    "spec": "65",
                    "quantity": 2.0,
                    "evidence": "보 65 양면",
                    "confidence": "HIGH",
                }
            ],
        }
        record = LLMStructuredRecord.model_validate(record_data)
        assert len(record.items) == 1
        assert record.items[0].confidence == "HIGH"
        assert record.items[0].evidence == "보 65 양면"


class TestAllOrNothingPostProcessing:
    """단일 결측 또는 낮은 신뢰도 발생 시 전면 빈 리스트([]) 처리 로직 테스트."""

    @patch.object(GeminiStructuringEngine, "__init__", return_value=None)
    def test_all_or_nothing_with_one_null_item(self, mock_init):
        engine = GeminiStructuringEngine()
        engine.model = "gemini-3.7-flash"
        engine.temperature = 0.0
        engine.thinking_level = "MEDIUM"
        engine.thinking_budget = None
        engine._api_key = "dummy"
        engine._client = MagicMock()

        # LLM 응답 시뮬레이션: 3개 중 1개의 spec이 null인 경우
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "records": [
                {
                    "location": "101동 3층",
                    "workDate": "2024-06-28",
                    "items": [
                        {
                            "matchedWorkTypeId": 101,
                            "workType": "금속관벽체",
                            "spec": "50",
                            "quantity": 1.0,
                            "confidence": "HIGH",
                        },
                        {
                            "matchedWorkTypeId": 101,
                            "workType": "금속관벽체",
                            "spec": None,  # 👈 결측
                            "quantity": 1.0,
                            "confidence": "LOW",
                        },
                        {
                            "matchedWorkTypeId": 201,
                            "workType": "보온덕트벽체",
                            "spec": "1000*500",
                            "quantity": 2.0,
                            "confidence": "HIGH",
                        },
                    ],
                }
            ]
        })
        engine._client.models.generate_content.return_value = mock_response

        work_types = [
            WorkTypeItem(id=101, name="금속관벽체"),
            WorkTypeItem(id=201, name="보온덕트벽체"),
        ]
        result = engine.process(
            [InputRecord(text="보 50 보 치수없음 보 1000*500")], work_types
        )

        assert result.success is True
        assert len(result.records) == 1
        # All-or-Nothing 규칙에 의해 1개라도 결측이 있으면 items는 빈 리스트 [] 여야 함
        assert result.records[0].items == []
        assert result.records[0].location == "101동 3층"
        assert result.records[0].workDate == "2024-06-28"

    @patch.object(GeminiStructuringEngine, "__init__", return_value=None)
    def test_all_or_nothing_all_valid_items(self, mock_init):
        engine = GeminiStructuringEngine()
        engine.model = "gemini-3.7-flash"
        engine.temperature = 0.0
        engine.thinking_level = "MEDIUM"
        engine.thinking_budget = None
        engine._api_key = "dummy"
        engine._client = MagicMock()

        # LLM 응답 시뮬레이션: 모두 정상적인 경우
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "records": [
                {
                    "location": "지하4",
                    "workDate": "2024-06-28",
                    "items": [
                        {
                            "matchedWorkTypeId": 101,
                            "workType": "금속관벽체",
                            "spec": "50",
                            "quantity": 2.0,
                            "evidence": "보 50*2",
                            "confidence": "HIGH",
                        },
                        {
                            "matchedWorkTypeId": 201,
                            "workType": "보온덕트벽체",
                            "spec": "1000*500",
                            "quantity": 1.0,
                            "evidence": "보 1000*500",
                            "confidence": "HIGH",
                        },
                    ],
                }
            ]
        })
        engine._client.models.generate_content.return_value = mock_response

        work_types = [
            WorkTypeItem(id=101, name="금속관벽체"),
            WorkTypeItem(id=201, name="보온덕트벽체"),
        ]
        result = engine.process(
            [InputRecord(text="보 50*2 보 1000*500")], work_types
        )

        assert result.success is True
        assert len(result.records) == 1
        assert len(result.records[0].items) == 2
        assert result.records[0].items[0].matchedWorkTypeId == 101
        assert result.records[0].items[0].spec == "50"
        assert result.records[0].items[1].matchedWorkTypeId == 201
        assert result.records[0].items[1].spec == "1000*500"


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
