"""현장 사진 분석 오케스트레이션 서비스 모듈.

하위 모듈(table_crop, ocr, structuring)을 변경 없이 조율(Orchestration)하여
S3 이미지 다운로드 -> 표 크롭 -> OCR 전사 -> LLM 정형화를 수행하고
API 규격에 맞춘 최종 응답을 조립하여 반환합니다.
"""

import asyncio
import logging
import time
import uuid
from typing import List, Optional

import cv2
import httpx
import numpy as np

# --- 하위 모듈 임포트 ---
from ai.src.ocr.engine import GeminiOCREngine
from ai.src.ocr.schemas import LoadedImageData, OCRResult
from ai.src.structuring.engine import GeminiStructuringEngine
from ai.src.structuring.schemas import (
    InputRecord,
    StructuredItem,
    StructuredRecord,
    StructuringResult,
    WorkTypeItem,
)
from ai.src.table_crop.cropper import CropResult, TableCropper
from ai.src.table_crop.loader import BytesImageLoader, LoadedImage

# --- 서버 스키마 임포트 ---
from ai.src.server.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalyzeResultData,
    ErrorDetail,
    StructuredItemResponse,
    StructuredRecordResponse,
)

logger = logging.getLogger("fieldnote_ai.server")


import os


class PipelineService:
    """현장 사진 분석 통합 오케스트레이터 서비스."""

    def __init__(
        self,
        cropper: Optional[TableCropper] = None,
        ocr_engine: Optional[GeminiOCREngine] = None,
        structuring_engine: Optional[GeminiStructuringEngine] = None,
        download_timeout_sec: float = 15.0,
        is_stub: Optional[bool] = None,
        stub_delay_sec: float = 15.0,
    ) -> None:
        """하위 모듈 인스턴스를 주입받아 서비스를 초기화합니다.

        Args:
            cropper (Optional[TableCropper]): 표 영역 검출 및 크롭 인스턴스.
            ocr_engine (Optional[GeminiOCREngine]): Gemini OCR 전사 엔진.
            structuring_engine (Optional[GeminiStructuringEngine]): Gemini 텍스트 정형화 엔진.
            download_timeout_sec (float): 이미지 다운로드 타임아웃(초).
            is_stub (Optional[bool]): 스텁 모드 활성화 여부 (None일 경우 AI_SERVER_MODE 환경변수 기준, 기본 True).
            stub_delay_sec (float): 스텁 모드 지연 시간 (초). 기본 15초.
        """
        self.image_loader = BytesImageLoader()
        self.cropper = cropper or TableCropper()
        self.ocr_engine = ocr_engine or GeminiOCREngine()
        self.structuring_engine = structuring_engine or GeminiStructuringEngine()
        self.download_timeout_sec = download_timeout_sec
        self.stub_delay_sec = stub_delay_sec

        if is_stub is not None:
            self.is_stub = is_stub
        else:
            # 환경변수 AI_SERVER_MODE가 'integrated'가 아니면 기본적으로 stub 모드로 구동
            self.is_stub = os.getenv("AI_SERVER_MODE", "stub").lower() == "stub"

    async def download_image(self, image_url: str) -> bytes:
        """S3 또는 외부 HTTP URL로부터 이미지 바이너리를 비동기 다운로드합니다.

        Args:
            image_url (str): 이미지 접근 URL.

        Returns:
            bytes: 다운로드된 원시 이미지 바이트 데이터.

        Raises:
            httpx.HTTPError: 네트워크 또는 HTTP 상태 오류 시 발생.
            ValueError: 다운로드된 데이터가 비어있는 경우 발생.
        """
        logger.info(f"[Download] 이미지 다운로드 시작: {image_url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 FieldNoteAI/1.0",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        }
        async with httpx.AsyncClient(
            timeout=self.download_timeout_sec,
            headers=headers,
            follow_redirects=True,
        ) as client:
            response = await client.get(image_url)
            response.raise_for_status()
            image_bytes = response.content

        if not image_bytes:
            raise ValueError("다운로드된 이미지 데이터가 비어 있습니다.")

        logger.info(f"[Download] 이미지 다운로드 완료 ({len(image_bytes)} bytes)")
        return image_bytes

    def _crop_table(self, image_bytes: bytes, file_name: str) -> bytes:
        """table_crop 모듈을 호출하여 표 영역을 크롭하고 JPEG 바이트로 변환합니다.

        검출 실패(fallback) 시에는 원본 이미지를 그대로 반환하여 OCR 모듈이 전체 사진을 보도록 합니다.

        Args:
            image_bytes (bytes): 원본 이미지 바이트.
            file_name (str): 파일명.

        Returns:
            bytes: 크롭된 이미지(또는 원본 fallback) 바이트.
        """
        try:
            loaded_img: LoadedImage = self.image_loader.load(
                image_bytes, file_name=file_name
            )
            crop_result: CropResult = self.cropper.crop(loaded_img)

            if crop_result.is_fallback:
                logger.warning(
                    f"[Crop] 표 영역 미검출 (is_fallback=True), 원본 이미지로 OCR을 진행합니다."
                )
                return image_bytes

            # OpenCV ndarray -> JPEG 바이트 인코딩
            success, encoded_buf = cv2.imencode(".jpg", crop_result.cropped_image)
            if not success or encoded_buf is None:
                logger.warning("[Crop] 크롭 이미지 JPEG 인코딩 실패, 원본 이미지를 사용합니다.")
                return image_bytes

            cropped_bytes = encoded_buf.tobytes()
            logger.info(
                f"[Crop] 표 크롭 성공 (BBox={crop_result.bbox}, size={len(cropped_bytes)} bytes)"
            )
            return cropped_bytes

        except Exception as e:
            logger.warning(f"[Crop] 표 크롭 중 예외 발생 ({e}), 원본 이미지로 fallback 처리합니다.")
            return image_bytes

    async def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        """현장 사진 분석 전체 파이프라인(다운로드 -> 크롭 -> OCR -> 정형화)을 실행합니다.

        Args:
            request (AnalyzeRequest): 이미지 URL 및 work_types를 포함한 분석 요청.

        Returns:
            AnalyzeResponse: 분석 완료 및 정형화 결과 DTO.
        """
        start_time = time.time()
        task_id = request.task_id or f"task_{uuid.uuid4().hex[:8]}"
        file_name = f"{task_id}.jpg"

        logger.info(f"========== [Pipeline Start: {task_id}] ==========")
        logger.info(f"Target Image: {request.image_url}")
        logger.info(f"Work Types Provided: {len(request.work_types)} items")
        logger.info(f"Server Mode: {'STUB (Mock with delay)' if self.is_stub else 'INTEGRATED (Real Pipeline)'}")

        # -------------------------------------------------------------
        # [스텁 모드 처리] is_stub=True인 경우 약 15초 대기 후 Mock 응답 반환
        # -------------------------------------------------------------
        if self.is_stub:
            logger.info(
                f"[PipelineService] (Stub 모드) 실제 파이프라인 모사를 위해 {self.stub_delay_sec}초 동안 대기합니다..."
            )
            await asyncio.sleep(self.stub_delay_sec)

            matched_id = 101
            if request.work_types:
                matched_id = request.work_types[0].id

            mock_raw_text = "2024.06.28 / 101동 3층 / 벽체D65*2 양면 시공 완료"
            mock_record = StructuredRecordResponse(
                location="101동 3층",
                workDate="2024-06-28",
                items=[
                    StructuredItemResponse(
                        matchedWorkTypeId=matched_id,
                        spec="65",
                        quantity=2.0,
                    )
                ],
            )

            execution_time = round(time.time() - start_time, 2)
            logger.info(
                f"========== [Pipeline Success (Stub): {task_id} in {execution_time}s] =========="
            )

            return AnalyzeResponse(
                success=True,
                task_id=task_id,
                data=AnalyzeResultData(
                    ocr_raw_text=mock_raw_text,
                    record=mock_record,
                ),
                error=None,
                execution_time_sec=execution_time,
            )

        try:
            # -------------------------------------------------------------
            # [Step 1] S3 이미지 비동기 다운로드
            # -------------------------------------------------------------
            try:
                image_bytes = await self.download_image(request.image_url)
            except Exception as e:
                logger.error(f"[Pipeline Error] 이미지 다운로드 실패: {e}")
                return AnalyzeResponse(
                    success=False,
                    task_id=task_id,
                    data=None,
                    error=ErrorDetail(
                        code="IMAGE_DOWNLOAD_FAILED",
                        message=f"이미지를 다운로드할 수 없습니다: {str(e)}",
                    ),
                    execution_time_sec=round(time.time() - start_time, 2),
                )

            # -------------------------------------------------------------
            # [Step 2] table_crop 모듈 호출 (표/보드판 영역 크롭)
            # -------------------------------------------------------------
            # CPU 바운드 연산이므로 별도 스레드에서 실행
            cropped_image_bytes = await asyncio.to_thread(
                self._crop_table, image_bytes, file_name
            )

            # -------------------------------------------------------------
            # [Step 3] ocr 모듈 호출 (Gemini VLM 기반 보드판 텍스트 전사)
            # -------------------------------------------------------------
            loaded_ocr_data = LoadedImageData(
                image_bytes=cropped_image_bytes,
                mime_type="image/jpeg",
                source_id=f"url:{request.image_url}",
                file_name=file_name,
            )

            logger.info("[OCR] Gemini OCR 전사 엔진 호출 시작...")
            ocr_result: OCRResult = await asyncio.to_thread(
                self.ocr_engine.process, loaded_ocr_data
            )

            if not ocr_result.success:
                logger.error(f"[Pipeline Error] OCR 전사 실패: {ocr_result.error_message}")
                return AnalyzeResponse(
                    success=False,
                    task_id=task_id,
                    data=None,
                    error=ErrorDetail(
                        code="OCR_EXTRACTION_FAILED",
                        message=ocr_result.error_message or "OCR 텍스트 추출에 실패했습니다.",
                    ),
                    execution_time_sec=round(time.time() - start_time, 2),
                )

            # OCR 텍스트 요약 및 추출 데이터 파싱
            ocr_item = ocr_result.data
            ocr_raw_text = ""
            if ocr_item:
                parts = []
                if ocr_item.공사명:
                    parts.append(f"공사명: {ocr_item.공사명}")
                if ocr_item.공종:
                    parts.append(f"공종: {ocr_item.공종}")
                if ocr_item.위치:
                    parts.append(f"위치: {ocr_item.위치}")
                if ocr_item.내용:
                    parts.append(f"내용: {ocr_item.내용}")
                if ocr_item.일자:
                    parts.append(f"일자: {ocr_item.일자}")
                ocr_raw_text = " / ".join(parts)
            else:
                ocr_raw_text = ocr_result.raw_response or ""

            logger.info(f"[OCR] 추출 원문: {ocr_raw_text}")

            # -------------------------------------------------------------
            # [Step 4] structuring 모듈 호출 (LLM 공종 분류 및 정형화)
            # -------------------------------------------------------------
            # OCR 결과를 InputRecord로 조립
            work_text_parts = []
            if ocr_item:
                if ocr_item.공종:
                    work_text_parts.append(ocr_item.공종)
                if ocr_item.내용:
                    work_text_parts.append(ocr_item.내용)
            combined_work_text = " ".join(work_text_parts) if work_text_parts else ocr_raw_text

            input_record = InputRecord(
                text=combined_work_text or "공사 내역 없음",
                location=ocr_item.위치 if ocr_item else None,
                workDate=ocr_item.일자 if ocr_item else None,
                metadata={"task_id": task_id, "image_url": request.image_url},
            )

            # 요청받은 work_types를 WorkTypeItem으로 변환
            domain_work_types = [
                WorkTypeItem(id=item.id, name=item.name) for item in request.work_types
            ]

            logger.info(
                f"[Structuring] Gemini 정형화 엔진 호출 시작 (공종 후보 {len(domain_work_types)}개)..."
            )
            struct_result: StructuringResult = await asyncio.to_thread(
                self.structuring_engine.process, [input_record], domain_work_types
            )

            if not struct_result.success or not struct_result.records:
                logger.error(
                    f"[Pipeline Error] 정형화 실패: {struct_result.error_message}"
                )
                return AnalyzeResponse(
                    success=False,
                    task_id=task_id,
                    data=None,
                    error=ErrorDetail(
                        code="STRUCTURING_FAILED",
                        message=struct_result.error_message
                        or "데이터 정형화 및 공종 매핑에 실패했습니다.",
                    ),
                    execution_time_sec=round(time.time() - start_time, 2),
                )

            # -------------------------------------------------------------
            # [Step 5] 최종 응답 DTO 조립
            # -------------------------------------------------------------
            first_record: StructuredRecord = struct_result.records[0]

            response_items = [
                StructuredItemResponse(
                    matchedWorkTypeId=item.matchedWorkTypeId,
                    spec=item.spec,
                    quantity=item.quantity,
                )
                for item in first_record.items
            ]

            response_record = StructuredRecordResponse(
                location=first_record.location or (ocr_item.위치 if ocr_item else None),
                workDate=first_record.workDate or (ocr_item.일자 if ocr_item else None),
                items=response_items,
            )

            execution_time = round(time.time() - start_time, 2)
            logger.info(
                f"========== [Pipeline Success: {task_id} in {execution_time}s] =========="
            )

            return AnalyzeResponse(
                success=True,
                task_id=task_id,
                data=AnalyzeResultData(
                    ocr_raw_text=ocr_raw_text,
                    record=response_record,
                ),
                error=None,
                execution_time_sec=execution_time,
            )

        except Exception as e:
            logger.exception(f"[Pipeline Fatal] 예기치 못한 시스템 오류: {e}")
            return AnalyzeResponse(
                success=False,
                task_id=task_id,
                data=None,
                error=ErrorDetail(
                    code="INTERNAL_SERVER_ERROR",
                    message=f"서버 내부 처리 중 오류가 발생했습니다: {str(e)}",
                ),
                execution_time_sec=round(time.time() - start_time, 2),
            )


# 싱글톤 서비스 인스턴스
pipeline_service = PipelineService()
