"""건설 현장 텍스트 정형화 및 공종 분류 모듈의 데이터 스키마 및 DTO 정의.

외부 입력 모델(InputRecord), 공종 메타데이터(WorkTypeItem),
LLM 출력 정형화 항목(LLMStructuredItem, LLMStructuredRecord, LLMStructuringBatchResponse),
외부/서버 호환 항목(StructuredItem, StructuredRecord, StructuringBatchResponse),
파이프라인 최종 결과 DTO(StructuringResult)를 정의합니다.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class WorkTypeItem(BaseModel):
    """표준 공종(WorkType) 메타데이터 모델.

    Attributes:
        id (int): 고유 식별자 ID (예: 101).
        name (str): 표준 공종/품목 이름 (예: '금속관벽체').
        description (Optional[str]): 공종 상세 설명 (선택).
    """

    id: int = Field(..., description="표준 WorkType 고유 ID")
    name: str = Field(..., description="표준 공종/품목명")
    description: Optional[str] = Field(default=None, description="공종 설명")


class InputRecord(BaseModel):
    """정형화 대상 단일 작업 기록 입력 모델.

    Attributes:
        text (str): 보드판 OCR 텍스트 또는 작업 일지 원문 내용.
        location (Optional[str]): 시공 위치 원문 (예: '지하4', '3동 38층').
        workDate (Optional[str]): 작업 일자 원문 (예: '2024-06-28').
        metadata (Dict[str, Any]): 이미지 ID, 소스 식별자 등 추가 부가 정보.
    """

    text: str = Field(..., description="정형화 대상 원문 작업 내용 텍스트")
    location: Optional[str] = Field(default=None, description="시공 위치 텍스트 원문 (선택)")
    workDate: Optional[str] = Field(default=None, description="작업 일자 텍스트 원문 (선택)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")


# ============================================================================
# [LLM Structured Output 전용 스키마 - Self-Reflection 및 근거/신뢰도 포함]
# ============================================================================

class LLMStructuredItem(BaseModel):
    """LLM이 자기점검(Self-Reflection)을 거쳐 생성하는 세부 시공 품목 모델."""

    matchedWorkTypeId: Optional[int] = Field(
        default=None,
        description="원문 텍스트에 명확한 공종 근거가 있는 경우에만 ID 부여 (불확실하거나 모호하면 반드시 null)",
    )
    workType: Optional[str] = Field(
        default=None,
        description="매칭된 표준 WorkType 공종명 (예: '금속관벽체', 식별 불가 시 null)",
    )
    spec: Optional[str] = Field(
        default=None,
        description="원문에 명시된 규격 (배관 직경 숫자 문자열, 덕트 가로*세로 치수, 없으면 반드시 null)",
    )
    quantity: float = Field(
        ...,
        description="시공 수량 (기본 개수 * 시공면 계수: 단면 0.5, 양면 1.0, 입상 1.0)",
    )
    evidence: Optional[str] = Field(
        default=None,
        description="원문 텍스트에서 본 품목/규격/수량을 추출한 핵심 어구 (추측인 경우 'NONE')",
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        default="HIGH",
        description="추출 신뢰도 (원문에 직접적인 명시가 부족하거나 추정한 경우 반드시 'LOW')",
    )


class LLMStructuredRecord(BaseModel):
    """LLM이 생성하는 단일 작업 레코드 정형화 모델."""

    location: Optional[str] = Field(
        default=None,
        description="정규화된 시공 위치 (원문에 명시되지 않거나 식별 불가 시 반드시 null)",
    )
    workDate: Optional[str] = Field(
        default=None,
        description="정규화된 작업 일자 (YYYY-MM-DD, 원문에 명시되지 않거나 식별 불가 시 반드시 null)",
    )
    items: List[LLMStructuredItem] = Field(
        default_factory=list,
        description="추출된 세부 시공 품목 목록 (비공사 텍스트, 모호한 텍스트인 경우 반드시 빈 리스트 [])",
    )


class LLMStructuringBatchResponse(BaseModel):
    """Gemini Structured Output을 위한 최상위 컨테이너 모델."""

    records: List[LLMStructuredRecord] = Field(
        default_factory=list,
        description="입력 순서에 대응되는 정형화된 작업 레코드 목록",
    )


# ============================================================================
# [서버 및 외부 호환 표준 DTO / Models]
# ============================================================================

class StructuredItem(BaseModel):
    """정형화된 단일 시공 품목 모델 (외부 및 server 모듈 호환)."""

    matchedWorkTypeId: Optional[int] = Field(
        default=None,
        description="매칭된 표준 WorkType ID (101~313, 불확실 시 null)",
    )
    workType: Optional[str] = Field(
        default=None,
        description="매칭된 표준 WorkType 공종명 (예: '금속관벽체', 식별 불가 시 null)",
    )
    spec: Optional[str] = Field(
        default=None,
        description="규격 (배관: 직경 정수 문자열, 덕트: 가로*세로 단면 치수, 마감: 치수 또는 null)",
    )
    quantity: float = Field(
        ...,
        description="시공 수량 (기본 개수 * 시공면 계수: 단면 0.5, 양면 1.0, 입상 1.0)",
    )


class StructuredRecord(BaseModel):
    """단일 입력 레코드에 대한 정형화 결과 모델 (외부 및 server 모듈 호환)."""

    location: Optional[str] = Field(
        default=None,
        description="정규화된 시공 위치 (식별 불가 시 null)",
    )
    workDate: Optional[str] = Field(
        default=None,
        description="정규화된 작업 일자 (YYYY-MM-DD 형식, 식별 불가 시 null)",
    )
    items: List[StructuredItem] = Field(
        default_factory=list,
        description="분리 및 정형화된 세부 시공 품목 목록",
    )


# 하위 호환성을 위한 별칭
StructuringBatchResponse = LLMStructuringBatchResponse


@dataclass
class StructuringResult:
    """텍스트 정형화 파이프라인 수행 결과 DTO.

    Attributes:
        success (bool): 정형화 처리 성공 여부.
        records (List[StructuredRecord]): 정형화된 레코드 목록.
        raw_response (Optional[str]): Gemini API 반환 원본 텍스트/JSON 문자열.
        error_message (Optional[str]): 실패 시 에러 메시지.
        execution_time_sec (float): 실행 소요 시간 (초).
        model_used (Optional[str]): 사용된 모델명.
    """

    success: bool
    records: List[StructuredRecord] = field(default_factory=list)
    raw_response: Optional[str] = None
    error_message: Optional[str] = None
    execution_time_sec: float = 0.0
    model_used: Optional[str] = None
