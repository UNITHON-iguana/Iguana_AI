"""건설 현장 텍스트 정형화 및 공종 분류 LLM 엔진 모듈.

Google GenAI SDK(Gemini 3.7 Flash)를 활용하여 비정형 작업 기록을
Pydantic 기반 Structured Output 스키마로 정밀하게 구조화합니다.
"""

from abc import ABC, abstractmethod
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from .prompt import StructuringPromptBuilder
from .schemas import (
    InputRecord,
    StructuredRecord,
    StructuringBatchResponse,
    StructuringResult,
    WorkTypeItem,
)

logger = logging.getLogger(__name__)

# --- [환경변수 로드] ---
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()


class BaseStructuringEngine(ABC):
    """텍스트 정형화 엔진의 추상 인터페이스."""

    @abstractmethod
    def process(
        self,
        input_records: Sequence[InputRecord],
        work_types: Sequence[WorkTypeItem],
    ) -> StructuringResult:
        """입력 작업 기록 목록을 분석하여 정형화 결과를 반환합니다.

        Args:
            input_records (Sequence[InputRecord]): 정형화 대상 입력 레코드 목록.
            work_types (Sequence[WorkTypeItem]): 표준 WorkType 메타데이터 목록.

        Returns:
            StructuringResult: 정형화 수행 결과 DTO.
        """
        pass


class GeminiStructuringEngine(BaseStructuringEngine):
    """Google Gemini 3.7 Flash 기반 텍스트 정형화 및 공종 분류 엔진.

    - 기본 모델: gemini-3.7-flash
    - 추론(Thinking): Medium 레벨 적용
    - 응답: StructuringBatchResponse Pydantic Structured Output 강제
    """

    DEFAULT_MODEL = "gemini-3.7-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        thinking_level: str = "MEDIUM",
        thinking_budget: Optional[int] = None,
    ) -> None:
        """GeminiStructuringEngine 초기화.

        Args:
            api_key (Optional[str]): Gemini API 키 (미제공 시 GEMINI_API_KEY 환경변수 사용).
            model (Optional[str]): 사용할 모델명 (기본: gemini-3.7-flash).
            temperature (float): 생성 온도 (결정론적 결과를 위해 기본 0.0).
            thinking_level (str): Thinking 레벨 ('LOW', 'MEDIUM', 'HIGH', 기본: 'MEDIUM').
            thinking_budget (Optional[int]): Thinking 토큰 예산 (선택).

        Raises:
            ValueError: API 키를 찾을 수 없는 경우 발생.
        """
        self._api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not self._api_key or self._api_key.strip() == "":
            raise ValueError(
                "GEMINI_API_KEY is required. Set it in .env or pass it to GeminiStructuringEngine."
            )

        self.model = model or os.environ.get("GEMINI_MODEL") or self.DEFAULT_MODEL
        self.temperature = temperature
        self.thinking_level = thinking_level
        self.thinking_budget = thinking_budget

        self._client = genai.Client(api_key=self._api_key)

    def _mask_api_key(self, text: str) -> str:
        """에러 로그 등에서 API 키 노출을 방지하기 위한 마스킹 헬퍼."""
        if self._api_key and self._api_key in text:
            return text.replace(self._api_key, "***API_KEY_MASKED***")
        return text

    def _build_config(
        self, system_instruction: str
    ) -> types.GenerateContentConfig:
        """GenerateContentConfig 객체를 생성합니다."""
        config_kwargs: Dict[str, Any] = {
            "system_instruction": system_instruction,
            "temperature": self.temperature,
            "response_mime_type": "application/json",
            "response_schema": StructuringBatchResponse,
        }

        # [Thinking 설정 적용]
        try:
            if self.thinking_level:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=self.thinking_level
                )
            elif self.thinking_budget is not None and self.thinking_budget > 0:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                )
        except Exception as e:
            logger.warning(f"ThinkingConfig 설정 중 경고 발생: {e}")

        return types.GenerateContentConfig(**config_kwargs)

    def process(
        self,
        input_records: Sequence[InputRecord],
        work_types: Sequence[WorkTypeItem],
    ) -> StructuringResult:
        """입력 레코드들을 Gemini API를 통해 일괄 정형화합니다.

        Args:
            input_records (Sequence[InputRecord]): 정형화 대상 입력 레코드 목록.
            work_types (Sequence[WorkTypeItem]): 표준 WorkType 목록.

        Returns:
            StructuringResult: 분석 및 변환 결과 DTO.
        """
        if not input_records:
            return StructuringResult(
                success=True,
                records=[],
                execution_time_sec=0.0,
                model_used=self.model,
            )

        start_time = time.time()
        system_instruction = StructuringPromptBuilder.build_system_instruction(
            work_types
        )
        user_content = StructuringPromptBuilder.build_user_content(input_records)
        config = self._build_config(system_instruction)

        try:
            logger.info(
                f"Gemini API 정형화 요청 시작 (모델: {self.model}, 레코드 수: {len(input_records)})"
            )
            response = self._client.models.generate_content(
                model=self.model,
                contents=[user_content],
                config=config,
            )
            elapsed = time.time() - start_time
            raw_text = response.text or ""

            # [응답 역직렬화 및 검증]
            parsed_data = json.loads(raw_text)
            records: List[StructuredRecord] = []

            if isinstance(parsed_data, dict) and "records" in parsed_data:
                batch_resp = StructuringBatchResponse.model_validate(parsed_data)
                records = batch_resp.records
            elif isinstance(parsed_data, list):
                for item in parsed_data:
                    records.append(StructuredRecord.model_validate(item))
            elif isinstance(parsed_data, dict):
                records.append(StructuredRecord.model_validate(parsed_data))

            logger.info(
                f"Gemini API 정형화 완료 (레코드 {len(records)}개 생성, 소요시간: {elapsed:.2f}s)"
            )
            return StructuringResult(
                success=True,
                records=records,
                raw_response=raw_text,
                execution_time_sec=elapsed,
                model_used=self.model,
            )

        except json.JSONDecodeError as e:
            elapsed = time.time() - start_time
            error_msg = f"Gemini JSON 역직렬화 실패: {str(e)}"
            logger.error(error_msg)
            return StructuringResult(
                success=False,
                records=[],
                raw_response=locals().get("raw_text"),
                error_message=self._mask_api_key(error_msg),
                execution_time_sec=elapsed,
                model_used=self.model,
            )

        except ValidationError as e:
            elapsed = time.time() - start_time
            error_msg = f"Pydantic 스키마 검증 실패: {str(e)}"
            logger.error(error_msg)
            return StructuringResult(
                success=False,
                records=[],
                raw_response=locals().get("raw_text"),
                error_message=self._mask_api_key(error_msg),
                execution_time_sec=elapsed,
                model_used=self.model,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"Gemini API 호출 중 예외 발생: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            return StructuringResult(
                success=False,
                records=[],
                raw_response=locals().get("raw_text"),
                error_message=self._mask_api_key(error_msg),
                execution_time_sec=elapsed,
                model_used=self.model,
            )
