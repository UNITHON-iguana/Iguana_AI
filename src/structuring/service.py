"""텍스트 정형화 통합 서비스 및 파이프라인 모듈.

외부 서버(FastAPI / Webhook / RPC)에서 요청이 인입될 때,
WorkType 저장소 조회 -> LLM 프롬프트 생성 및 추론 -> 응답 정형화 및 반환을
총괄하는 통합 서비스 진입점을 제공합니다.
"""

from abc import ABC, abstractmethod
import logging
from typing import List, Optional, Sequence, Union

from .engine import BaseStructuringEngine, GeminiStructuringEngine
from .repository import BaseWorkTypeRepository, LocalJsonWorkTypeRepository
from .schemas import InputRecord, StructuredRecord, StructuringResult, WorkTypeItem

logger = logging.getLogger(__name__)


class BaseResponseHandler(ABC):
    """외부 서버로 결과를 전송하기 위한 응답 핸들러/콜백 인터페이스.

    향후 웹훅, 메시지 큐, 외부 HTTP API 연동 시 이 인터페이스를 구현하여 주입할 수 있습니다.
    """

    @abstractmethod
    def handle_response(self, result: StructuringResult) -> None:
        """정형화 결과를 외부 대상으로 전달/처리합니다.

        Args:
            result (StructuringResult): 정형화 결과 DTO.
        """
        pass


class StructuringService:
    """텍스트 정형화 및 공종 분류 통합 파이프라인 서비스.

    WorkType Repository, LLM Engine, Response Handler를 조율하여
    단일/다중 현장 작업 기록을 정형화합니다.
    """

    def __init__(
        self,
        repository: Optional[BaseWorkTypeRepository] = None,
        engine: Optional[BaseStructuringEngine] = None,
        response_handler: Optional[BaseResponseHandler] = None,
    ) -> None:
        """StructuringService 초기화.

        Args:
            repository (Optional[BaseWorkTypeRepository]): WorkType 저장소 (기본: LocalJsonWorkTypeRepository).
            engine (Optional[BaseStructuringEngine]): LLM 추론 엔진 (기본: GeminiStructuringEngine).
            response_handler (Optional[BaseResponseHandler]): 외부 응답 전송용 핸들러 (선택).
        """
        self.repository = repository or LocalJsonWorkTypeRepository()
        self.engine = engine or GeminiStructuringEngine()
        self.response_handler = response_handler

    def process(self, input_record: Union[InputRecord, dict, str]) -> StructuringResult:
        """단일 작업 기록을 정형화합니다.

        Args:
            input_record (Union[InputRecord, dict, str]): 입력 작업 기록 객체, dict 또는 원문 문자열.

        Returns:
            StructuringResult: 정형화 결과 DTO.
        """
        if isinstance(input_record, str):
            record = InputRecord(text=input_record)
        elif isinstance(input_record, dict):
            record = InputRecord.model_validate(input_record)
        else:
            record = input_record

        return self.process_batch([record])

    def process_batch(
        self, input_records: Sequence[Union[InputRecord, dict, str]]
    ) -> StructuringResult:
        """여러 작업 기록을 일괄 정형화합니다.

        Args:
            input_records (Sequence[Union[InputRecord, dict, str]]): 입력 작업 기록 목록.

        Returns:
            StructuringResult: 일괄 정형화 결과 DTO.
        """
        # [입력 데이터 정규화]
        records: List[InputRecord] = []
        for r in input_records:
            if isinstance(r, str):
                records.append(InputRecord(text=r))
            elif isinstance(r, dict):
                records.append(InputRecord.model_validate(r))
            else:
                records.append(r)

        # [WorkType 메타데이터 조회]
        work_types: List[WorkTypeItem] = self.repository.get_work_types()

        # [LLM 엔진 처리 수행]
        result = self.engine.process(records, work_types)

        # [외부 응답 핸들러가 등록된 경우 알림]
        if self.response_handler:
            try:
                self.response_handler.handle_response(result)
            except Exception as e:
                logger.error(f"외부 응답 핸들러 실행 중 예외 발생: {e}")

        return result
