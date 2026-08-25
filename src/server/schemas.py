"""AI 서버 요청 및 응답 데이터 모델 (Schemas).

메인 백엔드와의 API 통신을 위한 Pydantic DTO 모델들을 정의합니다.
"""

from typing import Any, Dict, List, Optional
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, HttpUrl


class WorkTypeInputItem(BaseModel):
    """표준 공종(WorkType) 입력 모델."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: int = Field(
        ...,
        validation_alias=AliasChoices("id", "workTypeId", "work_type_id"),
        description="표준 공종 고유 ID",
        examples=[101],
    )
    name: str = Field(
        ...,
        validation_alias=AliasChoices(
            "name", "workTypeName", "work_type_name", "workType", "work_type"
        ),
        description="표준 공종명",
        examples=["내화충전"],
    )


class AnalyzeRequest(BaseModel):
    """현장 사진 분석 요청 DTO.

    Attributes:
        image_url (str): 분석할 현장 사진의 S3 접근 URL.
        work_types (List[WorkTypeInputItem]): 공통 DB에서 조회한 표준 공종 목록.
        task_id (Optional[str]): 외부 시스템의 작업/사진 고유 식별자 ID.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    image_url: str = Field(
        ...,
        validation_alias=AliasChoices("image_url", "imageUrl", "url", "image"),
        description="분석할 현장 사진의 S3 접근 URL (Public 또는 Presigned URL)",
        examples=["https://fieldnote-bucket.s3.ap-northeast-2.amazonaws.com/photos/board_01.jpg"],
    )
    work_types: List[WorkTypeInputItem] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "work_types",
            "workTypes",
            "workTypeList",
            "work_type_list",
            "workType",
            "work_type",
        ),
        description="공통 DB에서 조회한 표준 공종 목록",
        examples=[
            [
                {"id": 101, "name": "내화충전"},
                {"id": 102, "name": "덕트"},
                {"id": 103, "name": "금속관벽체"},
            ]
        ],
    )
    task_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("task_id", "taskId", "photo_id", "photoId", "id"),
        description="메인 서버에서 전달한 작업 고유 식별자 ID (생략 시 자동 생성)",
        examples=["task_20260825_001"],
    )


class StructuredItemResponse(BaseModel):
    """정형화된 단일 시공 품목 응답 모델."""

    matchedWorkTypeId: Optional[int] = Field(
        default=None,
        description="공통 DB 기준 매칭된 표준 WorkType ID (미식별 시 null)",
        examples=[101],
    )
    spec: Optional[str] = Field(
        default=None,
        description="규격 (배관: 직경 숫자 문자열, 덕트: 가로*세로 단면 치수 등)",
        examples=["65"],
    )
    quantity: float = Field(
        ...,
        description="최종 시공 수량 (개소)",
        examples=[2.0],
    )


class StructuredRecordResponse(BaseModel):
    """정형화된 레코드 응답 모델."""

    location: Optional[str] = Field(
        default=None,
        description="정규화된 시공 위치 (동, 층, 세대, 구역 등, 미식별 시 null)",
        examples=["101동 3층"],
    )
    workDate: Optional[str] = Field(
        default=None,
        description="정규화된 작업 일자 (YYYY-MM-DD, 미식별 시 null)",
        examples=["2024-06-28"],
    )
    items: List[StructuredItemResponse] = Field(
        default_factory=list,
        description="정형화된 세부 시공/자재 품목 리스트",
    )


class AnalyzeResultData(BaseModel):
    """분석 결과 데이터 컨테이너."""

    ocr_raw_text: str = Field(
        ...,
        description="OCR로 추출된 보드판 원문 텍스트",
        examples=["2024.06.28 / 101동 3층 / 벽체D65*2 양면 시공"],
    )
    record: StructuredRecordResponse = Field(
        ...,
        description="정형화된 공사 내역 데이터",
    )


class ErrorDetail(BaseModel):
    """에러 상세 모델."""

    code: str = Field(..., description="에러 코드", examples=["BOARD_NOT_FOUND"])
    message: str = Field(..., description="에러 메시지", examples=["보드판 영역을 식별할 수 없습니다."])


class AnalyzeResponse(BaseModel):
    """현장 사진 분석 최종 응답 DTO."""

    success: bool = Field(..., description="성공 여부", examples=[True])
    task_id: str = Field(..., description="작업 고유 식별자 ID", examples=["task_20260825_001"])
    data: Optional[AnalyzeResultData] = Field(default=None, description="분석 결과 데이터 (성공 시)")
    error: Optional[ErrorDetail] = Field(default=None, description="에러 상세 정보 (실패 시)")
    execution_time_sec: float = Field(..., description="전체 처리 소요 시간 (초)", examples=[15.02])
