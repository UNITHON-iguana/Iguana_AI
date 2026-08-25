"""건설 현장 공사 보드판 표(Table) OCR 패키지.

공사 보드판 이미지로부터 표 영역을 전사(Transcription)하고 구조화된 데이터를 추출하기 위한
이미지 로더(Loader), OCR 엔진(Engine), 결과 저장기(Saver), 통합 파이프라인(Pipeline) 모듈을 제공합니다.
"""

from .engine import BaseOCREngine, GeminiOCREngine, SYSTEM_INSTRUCTION
from .loader import BaseImageLoader, BytesImageLoader, FileImageLoader
from .pipeline import OCRPipeline
from .saver import BaseOCRResultSaver, JsonFileOCRResultSaver, MemoryOCRResultSaver
from .schemas import BoardTableItem, LoadedImageData, OCRResult, SaveSummary

__all__ = [
    # 데이터 스키마 및 DTO
    "BoardTableItem",
    "LoadedImageData",
    "OCRResult",
    "SaveSummary",
    # 이미지 로더
    "BaseImageLoader",
    "FileImageLoader",
    "BytesImageLoader",
    # OCR 엔진
    "BaseOCREngine",
    "GeminiOCREngine",
    "SYSTEM_INSTRUCTION",
    # 결과 저장기
    "BaseOCRResultSaver",
    "JsonFileOCRResultSaver",
    "MemoryOCRResultSaver",
    # 통합 파이프라인
    "OCRPipeline",
]
