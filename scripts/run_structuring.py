"""건설 현장 텍스트 정형화 및 공종 분류 실행 데모 스크립트.

사용 예시:
    # 기본 샘플 세트 실행:
    python scripts/run_structuring.py

    # 단일 텍스트 직접 입력 실행:
    python scripts/run_structuring.py --text "6번 벽체 보 20*2 100 50 양면" --location "지하4" --date "2024-06-28"
"""

import argparse
import json
import os
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.structuring.repository import LocalJsonWorkTypeRepository
from src.structuring.schemas import InputRecord
from src.structuring.service import StructuringService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="건설 현장 텍스트 정형화 및 공종 분류 실행 스크립트"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="정형화할 작업 내용 원문 텍스트 (지정하지 않으면 샘플 세트 실행)",
    )
    parser.add_argument(
        "--location", type=str, default=None, help="시공 위치 (예: 지하4)"
    )
    parser.add_argument(
        "--date", type=str, default=None, help="작업 일자 (예: 2024-06-28)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.7-flash",
        help="사용할 Gemini 모델명 (기본: gemini-3.7-flash)",
    )
    args = parser.parse_args()

    # [서비스 초기화]
    repo = LocalJsonWorkTypeRepository()
    service = StructuringService(repository=repo)

    if args.text:
        # 단일 입력 실행
        inputs = [
            InputRecord(
                text=args.text, location=args.location, workDate=args.date
            )
        ]
    else:
        # 실제 dataset.json 기반 대표 샘플 5종 테스트
        print("=" * 70)
        print("🏗️ [현장노트 AI] 대표 케이스 5종 텍스트 정형화 테스트")
        print("=" * 70)
        inputs = [
            InputRecord(
                text="1-벽체D65 양면",
                location="3동 38층",
                workDate="2024-06-28",
            ),
            InputRecord(
                text="6번 벽체 보 20*2 100 50 양면",
                location="지하4",
                workDate="2024-06-28",
            ),
            InputRecord(
                text="18번 벽체 보 150 50*3 단면",
                location="3동 31층",
                workDate="2024-06-29",
            ),
            InputRecord(
                text="20번 벽체 무 2000*600 차열마감1번 보 700*400 차열마감2번 양면",
                location="지하4",
                workDate="2024-06-28",
            ),
            InputRecord(
                text="2번 벽체 보 2000*800 차열마감2번 오프구2000*200 단면",
                location="지하2층",
                workDate="2024-06-30",
            ),
        ]

    for idx, inp in enumerate(inputs, 1):
        print(f"\n[입력 {idx}] 원문: '{inp.text}' | 위치: {inp.location} | 일자: {inp.workDate}")

    print("\n⏳ Gemini 3.7 Flash (Thinking: Medium) 분석 중...")
    result = service.process_batch(inputs)

    if not result.success:
        print(f"\n❌ 정형화 실패: {result.error_message}")
        sys.exit(1)

    print(f"\n✅ 정형화 완료 (소요 시간: {result.execution_time_sec:.2f}초, 모델: {result.model_used})")
    print("-" * 70)

    for idx, record in enumerate(result.records, 1):
        print(f"\n📋 [결과 {idx}] 위치: {record.location} | 일자: {record.workDate}")
        for item in record.items:
            # 매칭된 WorkType 이름 조회
            wt_name = "미매칭"
            if item.matchedWorkTypeId:
                wt = repo.get_by_id(item.matchedWorkTypeId)
                wt_name = wt.name if wt else f"ID({item.matchedWorkTypeId})"

            print(
                f"  - WorkType: {wt_name} (ID: {item.matchedWorkTypeId}) | "
                f"규격: {item.spec} | "
                f"수량: {item.quantity}"
            )

    print("\n" + "=" * 70)
    print("📦 [최종 반환 JSON 구조]")
    print(
        json.dumps(
            [r.model_dump() for r in result.records],
            ensure_ascii=False,
            indent=2,
        )
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
