"""AI 서버 API 라우터 정의.

엔드포인트 등록 및 요청 유효성 검사, 서비스 호출을 담당합니다.
"""

import logging
from fastapi import APIRouter, HTTPException, status

from ai.src.server.schemas import AnalyzeRequest, AnalyzeResponse, ErrorDetail
from ai.src.server.service import pipeline_service

logger = logging.getLogger("fieldnote_ai.server")

router = APIRouter(prefix="/api/v1", tags=["Analysis"])


@router.get("/health", summary="서버 헬스체크")
async def health_check() -> dict:
    """서버 가동 상태를 확인하는 헬스체크 엔드포인트."""
    mode = "stub" if pipeline_service.is_stub else "integrated"
    return {"status": "ok", "service": "fieldnote-ai-server", "mode": mode}


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="현장 사진 분석 및 데이터 정형화",
    description="S3 이미지 URL을 전달받아 표 크롭, OCR, 정형화를 수행한 후 최종 결과를 동기 반환합니다.",
)
async def analyze_photo(request: AnalyzeRequest) -> AnalyzeResponse:
    """현장 사진 분석 요청 핸들러.

    Args:
        request (AnalyzeRequest): 이미지 URL 및 작업 식별자 정보.

    Returns:
        AnalyzeResponse: 정형화 완료된 공사 내역 데이터.
    """
    logger.info(
        f"[Router Received] task_id={request.task_id}, image_url={request.image_url}, work_types_count={len(request.work_types)}"
    )
    if request.work_types:
        logger.info(
            f"[Router Received work_types 샘플] {[{'id': wt.id, 'name': wt.name} for wt in request.work_types[:3]]}"
        )
    else:
        logger.warning("[Router Received work_types] 빈 리스트([])로 수신되었습니다.")

    # 간단한 URL 형식 사전 검증
    if not request.image_url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "INVALID_IMAGE_URL",
                "message": "image_url은 http:// 또는 https:// 로 시작하는 유효한 URL이어야 합니다.",
            },
        )

    try:
        response = await pipeline_service.analyze(request)
        return response
    except Exception as e:
        logger.exception(f"[Router] 분석 처리 중 예외 발생: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "INTERNAL_SERVER_ERROR",
                "message": f"서버 내부 처리 중 오류가 발생했습니다: {str(e)}",
            },
        )
