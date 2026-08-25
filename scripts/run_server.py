"""AI 분석 서버 구동 스크립트.

Usage:
    python -m ai.scripts.run_server [--host 0.0.0.0] [--port 8000] [--reload]
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import uvicorn


def main() -> None:
    """서버 구동 메인 함수."""
    parser = argparse.ArgumentParser(description="현장노트 AI 분석 서버 구동")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="바인딩 호스트 (기본: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="바인딩 포트 (기본: 8000)")
    parser.add_argument("--reload", action="store_true", help="코드 변경 시 자동 리로드 활성화")

    args = parser.parse_args()

    uvicorn.run(
        "ai.src.server.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
