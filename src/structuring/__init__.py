"""건설 현장 텍스트 정형화 및 공종 분류 패키지.

Gemini 3.7 Flash 모델과 Structured Output을 활용하여
현장 작업 기록(텍스트, 위치, 일자)을 표준 WorkType ID, 규격, 수량으로 정형화합니다.
"""

from .engine import BaseStructuringEngine, GeminiStructuringEngine
from .prompt import StructuringPromptBuilder
from .repository import BaseWorkTypeRepository, LocalJsonWorkTypeRepository
from .schemas import (
    InputRecord,
    StructuredItem,
    StructuredRecord,
    StructuringResult,
    WorkTypeItem,
)
from .service import BaseResponseHandler, StructuringService

__all__ = [
    "BaseStructuringEngine",
    "GeminiStructuringEngine",
    "StructuringPromptBuilder",
    "BaseWorkTypeRepository",
    "LocalJsonWorkTypeRepository",
    "InputRecord",
    "StructuredItem",
    "StructuredRecord",
    "StructuringResult",
    "WorkTypeItem",
    "BaseResponseHandler",
    "StructuringService",
]
