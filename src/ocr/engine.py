"""건설 현장 공사 보드판 표(Table) 전용 OCR 엔진 모듈.

Google GenAI SDK(Gemini)를 활용하여 보드판 표 이미지로부터 무가공 원문(Raw Text)을 전사하고,
Structured JSON Schema를 통해 5대 핵심 항목(공사명, 공종, 위치, 내용, 일자)으로 정밀 추출합니다.
"""

from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from .schemas import BoardTableItem, LoadedImageData, OCRResult

# --- [환경변수 로드] ---
# [변수] ENV_PATH: 프로젝트 루트의 .env 파일 절대경로
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"

# [조건 검사] 프로젝트 루트에 .env 파일이 존재하는지 확인 (존재할 경우 해당 파일의 환경변수를 명시적으로 로드하여 API 키 등 설정 확보)
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# [상수] Gemini 모델에 주입되는 공사 보드판 전사 전용 시스템 프롬프트
# 핵심 규칙: 1. 무가공 원문 보존(특수문자/숫자 왜곡 금지), 2. 임의 추정(환각) 금지, 3. 표 항목 정확한 1:1 매핑
SYSTEM_INSTRUCTION = """당신은 건설 현장 공사 보드판 표(Table) 이미지를 읽어내는 고정밀 OCR 엔진입니다. 제공된 이미지의 표에서 각 항목에 해당하는 텍스트를
시각적으로 보이는 그대로 전사(Transcribe)하세요. [핵심 규칙]

1. 무가공 원문 보존:
   - 텍스트 내의 모든 특수문자(*, -, ., /, ~, Φ, Ø 등), 숫자, 알파벳, 한글을 왜곡이나 누락 없이 원형 그대로 추출하세요.
   - 텍스트를 임의로 요약, 수정, 해석, 표준화하거나 오타를 교정하지 마세요.
2. 임의 추정(환각) 금지:
   - 이미지에 실제로 보이지 않는 글자나 누락된 정보를 상상하여 채워 넣지 마세요.
   - 글자가 흐릿하거나 식별이 어려운 항목의 값은 null로 반환하세요.
3. 필드별 텍스트 매핑:
   - 이미지 표의 좌측 항목명(공사명, 공종, 위치, 내용, 일자)에 대응하는 우측 셀의 값을 정확히 추출하여 매핑하세요."""


class BaseOCREngine(ABC):
    """OCR 엔진의 추상 기반 클래스.

    LoadedImageData를 입력받아 OCR 분석을 수행하고 OCRResult를 생성하는 공통 인터페이스를 정의합니다.
    """

    @abstractmethod
    def process(self, image_data: LoadedImageData) -> OCRResult:
        """단일 로드된 이미지에 대해 OCR 분석을 수행합니다.

        Args:
            image_data (LoadedImageData): OCR 대상 이미지 데이터 컨테이너.

        Returns:
            OCRResult: 분석 및 전사 결과 객체.

        Raises:
            NotImplementedError: 하위 클래스에서 구현되지 않은 경우 발생.
        """
        pass

    def process_batch(self, images: Sequence[LoadedImageData]) -> List[OCRResult]:
        """여러 이미지에 대해 순차적으로 OCR 처리를 수행하고 결과 목록을 반환합니다.

        Args:
            images (Sequence[LoadedImageData]): 처리할 이미지 데이터 객체들의 시퀀스.

        Returns:
            List[OCRResult]: 각 이미지별 OCR 결과 객체들의 리스트.
        """
        # [변수] results: 순차 처리된 OCRResult 결과 객체들을 저장할 리스트
        results: List[OCRResult] = []

        for img in images:
            # [단일 이미지 처리] 각 이미지에 대해 OCR 엔진 실행 후 결과 수집
            res = self.process(img)
            results.append(res)

        return results


class GeminiOCREngine(BaseOCREngine):
    """Google Gemini API를 활용한 고정밀 공사 보드판 OCR 엔진 클래스."""

    # [상수] 기본 권장 모델 및 대체(Fallback) 모델 목록
    DEFAULT_MODEL = "gemini-3.5-flash-lite"
    FALLBACK_MODELS = ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-1.5-flash"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        system_instruction: str = SYSTEM_INSTRUCTION,
        temperature: float = 0.0,
        thinking_budget: int = 0,
    ):
        """GeminiOCREngine 인스턴스를 초기화합니다.

        Args:
            api_key (Optional[str]): Gemini API 키. 미제공 시 GEMINI_API_KEY 환경변수에서 로드.
            model (Optional[str]): 사용할 모델명. 미제공 시 GEMINI_MODEL 환경변수 또는 기본 모델 사용.
            system_instruction (str): Gemini에 전달할 시스템 프롬프트. 기본값은 보드판 전용 전사 프롬프트.
            temperature (float): 생성 온도 (결정론적이고 일관된 결과를 위해 기본값 0.0).
            thinking_budget (int): 모델의 추론(Thinking) 토큰 예산. 지연 시간 최소화를 위해 기본값 0.

        Raises:
            ValueError: API 키가 환경변수 및 매개변수 어디에도 설정되지 않은 경우 발생.
        """
        # [변수] self._api_key: 직접 전달된 API 키 또는 환경변수에서 읽어온 키
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")

        # [조건 검사] API 키가 유효하게 설정되어 있는지 확인 (API 키가 없으면 Gemini API 통신이 불가능하므로 즉시 오류 발생)
        if not self._api_key or self._api_key.strip() == "":
            raise ValueError(
                "GEMINI_API_KEY is required. Set it in your .env file or pass it to GeminiOCREngine."
            )

        # [설정 변수 할당] 모델명, 시스템 지침, 온도, 사고 예산 설정
        self.model = model or os.environ.get("GEMINI_MODEL") or self.DEFAULT_MODEL
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.thinking_budget = thinking_budget

        # [클라이언트 초기화] Google GenAI Client 생성
        self._client = genai.Client(api_key=self._api_key)

    def _mask_api_key(self, text: str) -> str:
        """예외 메시지 등에 실수로 노출될 수 있는 API 키 문자열을 마스킹 처리합니다.

        Args:
            text (str): 원본 텍스트 또는 오류 메시지.

        Returns:
            str: API 키가 마스킹된 안전한 텍스트.
        """
        # [조건 검사] API 키가 실제 존재하고 대상 텍스트 내에 포함되어 있는지 확인 (보안 규칙에 따른 민감정보 노출 방지)
        if self._api_key and self._api_key in text:
            return text.replace(self._api_key, "***API_KEY_MASKED***")
        return text

    def _build_config(self) -> types.GenerateContentConfig:
        """정형화된 스키마 출력 및 옵션이 적용된 GenerateContentConfig 객체를 구성합니다.

        Returns:
            types.GenerateContentConfig: Gemini API 호출에 사용할 설정 객체.
        """
        # [변수] config_kwargs: Gemini API 호출을 위한 설정 파라미터 딕셔너리
        config_kwargs: Dict[str, Any] = {
            "system_instruction": self.system_instruction,
            "temperature": self.temperature,
            "response_mime_type": "application/json",
            "response_schema": BoardTableItem,
        }

        # [조건 검사] thinking_budget이 설정되어 있고 양수인지 확인 (GenAI SDK API 규격상 0 이하의 값은 유효하지 않으므로 양수일 때만 추가)
        if self.thinking_budget and self.thinking_budget > 0:
            try:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                )
            except Exception:
                # [예외 무시] 지원되지 않는 SDK 버전인 경우 안전하게 스킵
                pass

        return types.GenerateContentConfig(**config_kwargs)

    def process(self, image_data: LoadedImageData) -> OCRResult:
        """Gemini API를 호출하여 이미지 내 보드판 표의 텍스트를 전사하고 결과를 반환합니다.

        Args:
            image_data (LoadedImageData): 입력 이미지 데이터 컨테이너.

        Returns:
            OCRResult: 파싱된 항목 데이터, 원본 응답, 실행 시간 및 성공 여부가 포함된 결과 객체.
        """
        # [변수] start_time: 실행 시간 측정을 위한 고정밀 시작 시각
        start_time = time.perf_counter()

        # [변수] config: GenerateContentConfig 객체 생성
        config = self._build_config()

        # [변수] image_part: 프롬프트 텍스트 없이 순수 이미지 바이너리만 포함하는 Part 객체 구성
        image_part = types.Part.from_bytes(
            data=image_data.image_bytes,
            mime_type=image_data.mime_type,
        )

        # [변수] active_model: 현재 시도 중인 주 모델명
        active_model = self.model

        # [변수] last_error: 호출 실패 시 기록할 마지막 예외 객체
        last_error: Optional[Exception] = None

        # [변수] models_to_try: 시도할 모델 후보 목록 (기본 지정 모델 우선 시도)
        models_to_try = [self.model]
        for fb in self.FALLBACK_MODELS:
            # [조건 검사] 대체 모델이 시도 목록에 이미 포함되어 있는지 확인 (중복 호출 방지)
            if fb not in models_to_try:
                models_to_try.append(fb)

        # --- [모델 호출 및 재시도 루프] ---
        for current_model in models_to_try:
            try:
                # [API 호출] Gemini 멀티모달 generate_content 실행
                response = self._client.models.generate_content(
                    model=current_model,
                    contents=[image_part],
                    config=config,
                )

                # [변수] raw_text: Gemini로부터 반환된 원본 텍스트
                raw_text = response.text or ""

                # [변수] parsed_data: 원본 응답을 파싱하여 생성된 BoardTableItem 모델 객체
                parsed_data = self._parse_json_response(raw_text)

                # [변수] elapsed: 총 소요 시간(초) 계산
                elapsed = time.perf_counter() - start_time

                # [성공 결과 반환]
                return OCRResult(
                    source_id=image_data.source_id,
                    file_name=image_data.file_name,
                    success=True,
                    data=parsed_data,
                    raw_response=raw_text,
                    error_message=None,
                    execution_time_sec=elapsed,
                    model_used=current_model,
                    metadata={"mime_type": image_data.mime_type, "size_bytes": image_data.size_bytes},
                )
            except Exception as e:
                last_error = e
                # [변수] err_str: 오류 판별을 위한 소문자 변환 에러 문자열
                err_str = str(e).lower()

                # [조건 검사] 오류가 모델 미지원 또는 404 Not Found에 해당하는지 확인 (지정된 모델을 사용할 수 없는 경우 다음 대체 모델로 재시도)
                if "404" in err_str or "not found" in err_str or "not supported" in err_str:
                    continue
                else:
                    # [조건 분기] 기타 인증 에러나 치명적 오류의 경우 즉시 반복 중단
                    break

        # --- [실패 처리] ---
        # [변수] elapsed: 실패 시점까지의 총 소요 시간 계산
        elapsed = time.perf_counter() - start_time

        # [변수] safe_error_msg: API 키 등이 마스킹된 안전한 오류 메시지 문자열
        safe_error_msg = self._mask_api_key(str(last_error))

        return OCRResult(
            source_id=image_data.source_id,
            file_name=image_data.file_name,
            success=False,
            data=None,
            raw_response=None,
            error_message=safe_error_msg,
            execution_time_sec=elapsed,
            model_used=active_model,
            metadata={"mime_type": image_data.mime_type, "size_bytes": image_data.size_bytes},
        )

    def _parse_json_response(self, raw_text: str) -> BoardTableItem:
        """Gemini 모델이 반환한 JSON 문자열을 파싱하고 검증하여 BoardTableItem 객체를 생성합니다.

        Args:
            raw_text (str): Gemini API 응답 원문 텍스트.

        Returns:
            BoardTableItem: 유효성이 검증된 공사 보드판 항목 데이터.

        Raises:
            json.JSONDecodeError: JSON 파싱에 실패한 경우 발생.
            ValidationError: Pydantic 스키마 검증에 실패한 경우 발생.
        """
        # [변수] cleaned_text: 양끝 공백이 제거된 응답 문자열
        cleaned_text = raw_text.strip()

        # [조건 검사] 응답이 마크다운 코드블록(```json ... ```)으로 래핑되어 있는지 확인 (순수 JSON 파싱을 위해 래퍼 태그 제거)
        if cleaned_text.startswith("```"):
            # [변수] lines: 개행 단위로 분리된 문자열 라인 리스트
            lines = cleaned_text.splitlines()
            # [조건 검사] 첫 줄이 코드블록 시작 태그(``` 또는 ```json)인지 확인하여 제거
            if lines[0].startswith("```"):
                lines = lines[1:]
            # [조건 검사] 마지막 줄이 코드블록 닫는 태그(```)인지 확인하여 제거
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            # [변수] cleaned_text: 코드블록이 제거된 순수 JSON 텍스트 재조합
            cleaned_text = "\n".join(lines).strip()

        # [변수] data_dict: JSON 문자열을 파싱한 딕셔너리
        data_dict = json.loads(cleaned_text)

        # [변수] normalized: 5개 필수 필드가 누락되지 않도록 기본값(None)을 보장하는 정규화 딕셔너리
        normalized = {
            "공사명": data_dict.get("공사명"),
            "공종": data_dict.get("공종"),
            "위치": data_dict.get("위치"),
            "내용": data_dict.get("내용"),
            "일자": data_dict.get("일자"),
        }

        # [Pydantic 인스턴스화] 스키마 검증 수행 후 반환
        return BoardTableItem(**normalized)
