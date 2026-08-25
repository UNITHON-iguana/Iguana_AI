"""건설 현장 텍스트 정형화 및 공종 분류 모듈의 데이터 스키마 및 DTO 정의.

외부 입력 모델(InputRecord), 공종 메타데이터(WorkTypeItem),
LLM 출력 정형화 항목(StructuredItem, StructuredRecord, StructuringBatchResponse),
파이프라인 최종 결과 DTO(StructuringResult)를 정의합니다.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
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


class StructuredItem(BaseModel):
    """정형화된 단일 시공 품목 모델 (Pydantic / Structured Output 연동).

    Attributes:
        matchedWorkTypeId (Optional[int]): 매칭된 표준 WorkType ID (식별 불가 시 None).
        spec (Optional[str]): 호칭경(배관) 또는 가로*세로(덕트) 규격 문자열.
        quantity (float): 시공면 계수가 반영된 최종 수량 (개소).
    """

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
    """단일 입력 레코드에 대한 정형화 결과 모델.

    Attributes:
        location (Optional[str]): 정규화된 시공 위치 (동, 층, 세대, 구역 등).
        workDate (Optional[str]): 정규화된 작업 일자 (YYYY-MM-DD 형식).
        items (List[StructuredItem]): 분리 및 정형화된 세부 시공 품목 목록.
    """

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


class StructuringBatchResponse(BaseModel):
    """Gemini Structured Output을 위한 최상위 컨테이너 모델."""

    records: List[StructuredRecord] = Field(
        default_factory=list,
        description="입력 순서에 대응되는 정형화된 작업 레코드 목록",
    )


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
