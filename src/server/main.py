"""FastAPI 애플리케이션 진입점.

서버 앱 인스턴스 생성, 미들웨어 및 라우터 등록, 로깅 구성을 수행합니다.
"""

import logging
import sys
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai.src.server.router import router

# 로깅 기본 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("fieldnote_ai.server")


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 수명 주기(Lifespan) 이벤트 핸들러."""
    logger.info("==================================================")
    logger.info("🚀 현장노트 AI 분석 서버가 가동되었습니다. (Stub 모드: 15초 지연)")
    logger.info("   - Swagger Docs: http://localhost:8000/docs")
    logger.info("   - Health Check: http://localhost:8000/api/v1/health")
    logger.info("==================================================")
    yield


def create_app() -> FastAPI:
    """FastAPI 애플리케이션 팩토리 함수."""
    app = FastAPI(
        title="현장노트 AI 분석 서버 (Field Note AI)",
        description="건설 현장 사진(공사 보드판)의 표 크롭, OCR, 공종 분류 및 데이터 정형화 API 서버",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS 설정 (개발 및 외부 서버 연동용)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_raw_requests(request: Request, call_next):
        """인입되는 모든 HTTP 요청의 원본(Raw) 바디를 로깅하는 미들웨어."""
        if request.url.path == "/api/v1/analyze" and request.method == "POST":
            raw_body = await request.body()
            try:
                decoded_body = raw_body.decode("utf-8")
                logger.info("================ [RAW HTTP REQUEST BODY] ================")
                logger.info(f"Method: {request.method} {request.url.path}")
                logger.info(f"Headers: {dict(request.headers)}")
                logger.info(f"Raw Body: {decoded_body}")
                logger.info("=========================================================")
            except Exception as e:
                logger.warning(f"Failed to decode raw request body: {e}")

        response = await call_next(request)
        return response

    # 라우터 등록
    app.include_router(router)

    return app


app = create_app()
