"""OCR 모듈 데이터 스키마 및 DTO(Data Transfer Object) 정의.

공사 보드판 표 항목(BoardTableItem), 이미지 로드 데이터(LoadedImageData),
OCR 수행 결과(OCRResult), 결과 저장 요약(SaveSummary) 등 모듈 전반에서
사용되는 정형화된 데이터 모델을 정의합니다.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class BoardTableItem(BaseModel):
    """공사 보드판 표(Table)에서 추출된 5대 핵심 항목을 표현하는 Pydantic 모델.

    무가공 원문 보존 원칙과 임의 추정(환각) 금지 원칙에 따라,
    표 내에서 식별 불가능하거나 누락된 항목은 임의로 값을 생성하지 않고 반드시 null(None)로 유지합니다.

    Attributes:
        공사명 (Optional[str]): 공사명 원문 텍스트 (식별 불가 시 None).
        공종 (Optional[str]): 공종 원문 텍스트 (식별 불가 시 None).
        위치 (Optional[str]): 위치 원문 텍스트 (식별 불가 시 None).
        내용 (Optional[str]): 내용 원문 텍스트 (식별 불가 시 None).
        일자 (Optional[str]): 일자 원문 텍스트 (식별 불가 시 None).
    """

    # [모델 설정] JSON Schema 생성 시 5개 핵심 필드가 모두 출력 스키마에 포함되도록 필수 키로 지정
    model_config = ConfigDict(
        json_schema_extra={
            "required": ["공사명", "공종", "위치", "내용", "일자"]
        }
    )

    # [필드 정의] 공사 보드판 5대 기본 항목
    공사명: Optional[str] = Field(
        default=None,
        description="공사명 텍스트 원문 (식별 불가 시 null)",
    )
    공종: Optional[str] = Field(
        default=None,
        description="공종 텍스트 원문 (식별 불가 시 null)",
    )
    위치: Optional[str] = Field(
        default=None,
        description="위치 텍스트 원문 (식별 불가 시 null)",
    )
    내용: Optional[str] = Field(
        default=None,
        description="내용 텍스트 원문 (식별 불가 시 null)",
    )
    일자: Optional[str] = Field(
        default=None,
        description="일자 텍스트 원문 (식별 불가 시 null)",
    )


@dataclass
class LoadedImageData:
    """OCR 엔진에 입력하기 위해 메모리에 로드된 이미지 데이터 컨테이너.

    Attributes:
        image_bytes (bytes): 원시 이미지 바이너리 바이트.
        mime_type (str): 이미지 MIME 타입 (예: 'image/jpeg', 'image/png').
        source_id (str): 원본 식별자 (파일 절대경로 또는 버퍼 식별 문자열).
        file_name (str): 원본 파일명 (예: 'board_01.jpg').
        metadata (Dict[str, Any]): 파일 크기, 경로 등 추가 메타데이터.
    """

    image_bytes: bytes
    mime_type: str
    source_id: str
    file_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        """로드된 이미지의 바이트 크기를 반환합니다.

        Returns:
            int: 이미지 바이너리 데이터의 바이트 크기.
        """
        # [속성 계산] 원시 바이트 데이터의 길이 반환
        return len(self.image_bytes)


@dataclass
class OCRResult:
    """단일 이미지에 대한 OCR 분석 및 전사 결과 컨테이너.

    Attributes:
        source_id (str): 원본 이미지 식별자 (파일 경로 등).
        file_name (str): 원본 이미지 파일명.
        success (bool): OCR 처리 성공 여부.
        data (Optional[BoardTableItem]): 성공 시 추출된 구조화 보드판 항목 데이터.
        raw_response (Optional[str]): Gemini API로부터 반환된 원본 텍스트/JSON 문자열.
        error_message (Optional[str]): 실패 시 발생한 에러 메시지 (민감정보 마스킹 처리됨).
        execution_time_sec (float): OCR 처리에 소요된 시간 (초 단위).
        model_used (Optional[str]): 실제 OCR 처리에 사용된 Gemini 모델명.
        metadata (Dict[str, Any]): 이미지 포맷, 크기 등 부가 메타데이터.
    """

    source_id: str
    file_name: str
    success: bool
    data: Optional[BoardTableItem] = None
    raw_response: Optional[str] = None
    error_message: Optional[str] = None
    execution_time_sec: float = 0.0
    model_used: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """OCR 결과를 JSON 직렬화가 가능한 Python 딕셔너리로 변환합니다.

        Returns:
            Dict[str, Any]: 딕셔너리 형태로 구조화된 OCR 결과 데이터.
        """
        # [변환 처리] Pydantic 모델인 data 필드를 딕셔너리로 덤프하여 직렬화 가능 구조 생성
        # [조건 검사] data 객체가 존재하는 경우 딕셔너리로 변환하고, 없거나 실패한 경우 None으로 설정
        parsed_data_dict = self.data.model_dump() if self.data else None

        # [소요 시간 포맷] 소수점 3자리까지 반올림하여 가독성 확보
        formatted_exec_time = round(self.execution_time_sec, 3)

        return {
            "source_id": self.source_id,
            "file_name": self.file_name,
            "success": self.success,
            "data": parsed_data_dict,
            "raw_response": self.raw_response,
            "error_message": self.error_message,
            "execution_time_sec": formatted_exec_time,
            "model_used": self.model_used,
            "metadata": self.metadata,
        }


@dataclass
class SaveSummary:
    """OCR 결과 저장 작업 수행 결과 요약 컨테이너.

    Attributes:
        destination (str): 저장이 수행된 대상 경로 또는 저장소 식별자.
        file_name (str): 저장된 결과 파일명.
        success (bool): 저장 성공 여부.
        bytes_written (int): 저장된 데이터의 바이트 크기.
        error_message (Optional[str]): 저장 실패 시 발생한 에러 메시지.
        metadata (Dict[str, Any]): 배치 저장 여부, 레코드 수 등 부가 정보.
    """

    destination: str
    file_name: str
    success: bool
    bytes_written: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
