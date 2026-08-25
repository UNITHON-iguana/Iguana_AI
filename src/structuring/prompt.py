"""텍스트 정형화 시스템 프롬프트 및 Few-shot 예시 빌더 모듈.

WorkType 저장소로부터 조회한 최신 공종 목록과 도메인 파싱/수량 연산 규칙을
결합하여 Gemini 모델에 전달할 System Instruction 및 User Content를 동적으로 생성합니다.
"""

import json
from typing import List, Sequence

from .schemas import InputRecord, WorkTypeItem

FEW_SHOT_EXAMPLES = """
[Few-Shot 예시 1: 다중 배관 규격 및 단면/양면]
입력:
[
  {
    "text": "6번 벽체 보 20*2 100 50 양면",
    "location": "지하4",
    "workDate": "2024-06-28"
  },
  {
    "text": "18번 벽체 보 150 50*3 단면",
    "location": "3동 31층",
    "workDate": "2024-06-29"
  }
]
출력:
{
  "records": [
    {
      "location": "지하4",
      "workDate": "2024-06-28",
      "items": [
        { "matchedWorkTypeId": 101, "spec": "20", "quantity": 2.0 },
        { "matchedWorkTypeId": 101, "spec": "100", "quantity": 1.0 },
        { "matchedWorkTypeId": 101, "spec": "50", "quantity": 1.0 }
      ]
    },
    {
      "location": "3동 31층",
      "workDate": "2024-06-29",
      "items": [
        { "matchedWorkTypeId": 101, "spec": "150", "quantity": 0.5 },
        { "matchedWorkTypeId": 101, "spec": "50", "quantity": 1.5 }
      ]
    }
  ]
}

[Few-Shot 예시 2: 복합 덕트 및 차열재마감]
입력:
[
  {
    "text": "1번 벽체 무 2000*600 차열마감1번 단면",
    "location": "지하2층",
    "workDate": "2024-07-01"
  },
  {
    "text": "1번 벽체 보 1000*500*2 차열마감2번 양면",
    "location": "지하1층",
    "workDate": "2024-07-01"
  }
]
출력:
{
  "records": [
    {
      "location": "지하2층",
      "workDate": "2024-07-01",
      "items": [
        { "matchedWorkTypeId": 202, "spec": "2000*600", "quantity": 0.5 },
        { "matchedWorkTypeId": 301, "spec": "2000*600", "quantity": 0.5 }
      ]
    },
    {
      "location": "지하1층",
      "workDate": "2024-07-01",
      "items": [
        { "matchedWorkTypeId": 201, "spec": "1000*500", "quantity": 2.0 },
        { "matchedWorkTypeId": 301, "spec": "1000*500", "quantity": 4.0 }
      ]
    }
  ]
}

[Few-Shot 예시 3: 혼합 공종 (덕트 + 차열재 + 오픈구 + 배관)]
입력:
[
  {
    "text": "2번 벽체 보 2000*800 차열마감2번 오프구2000*200 단면",
    "location": "지하3층",
    "workDate": "2024-07-02"
  },
  {
    "text": "1벽체 보덕트 1000*500 차열두벌 보D 100. 단면",
    "location": "지하3층",
    "workDate": "2024-07-02"
  }
]
출력:
{
  "records": [
    {
      "location": "지하3층",
      "workDate": "2024-07-02",
      "items": [
        { "matchedWorkTypeId": 201, "spec": "2000*800", "quantity": 0.5 },
        { "matchedWorkTypeId": 301, "spec": "2000*800", "quantity": 1.0 },
        { "matchedWorkTypeId": 308, "spec": "2000*200", "quantity": 0.5 }
      ]
    },
    {
      "location": "지하3층",
      "workDate": "2024-07-02",
      "items": [
        { "matchedWorkTypeId": 201, "spec": "1000*500", "quantity": 0.5 },
        { "matchedWorkTypeId": 301, "spec": "1000*500", "quantity": 1.0 },
        { "matchedWorkTypeId": 101, "spec": "100", "quantity": 0.5 }
      ]
    }
  ]
}
"""


class StructuringPromptBuilder:
    """텍스트 정형화 System Instruction 및 User Content를 동적으로 생성하는 빌더."""

    @classmethod
    def build_system_instruction(
        cls, work_types: Sequence[WorkTypeItem]
    ) -> str:
        """WorkType 목록을 포함하여 완성된 System Instruction 텍스트를 빌드합니다.

        Args:
            work_types (Sequence[WorkTypeItem]): 표준 WorkType 목록.

        Returns:
            str: 완성된 System Instruction 프롬프트.
        """
        # [WorkType 테이블 구성]
        work_type_lines = []
        for wt in work_types:
            desc = f" ({wt.description})" if wt.description else ""
            work_type_lines.append(f"- ID {wt.id}: {wt.name}{desc}")

        work_types_block = "\n".join(work_type_lines)

        system_instruction = f"""당신은 건설 현장(내화채움구조, 설비, 배관, 덕트 공사 등)의 비정형 작업 기록 텍스트를 분석하여 구조화된 공사 데이터(JSON)로 정규화하는 전문 AI 파서입니다.
입력된 현장 텍스트(보드판 OCR 텍스트, 작업 일지 원문 등)로부터 작업 위치(location), 작업 일자(workDate), 그리고 세부 작업 품목 목록(items)을 추출하여 지정된 스키마의 `records` 목록에 맞게 정형화하십시오.

---

### [표준 WorkType (공종/품목) 정의 테이블]
원문의 작업 내용을 분석하여 가장 적합한 WorkType의 `id`를 `matchedWorkTypeId`에 매핑하십시오. 매칭되는 항목이 없거나 불확실한 경우 `null`로 지정하십시오.

{work_types_block}

---

### [필드별 파싱 및 계산 규칙]

1. **`location` (시공 위치)**:
   - 동, 층, 세대, 구역 정보가 원문에 포함된 경우 문자열로 추출 (예: "지하4", "3동 38층", "3동 1.2세대 31층").
   - 위치 정보가 전혀 없거나 식별 불가능한 경우 null.

2. **`workDate` (작업 일자)**:
   - "YYYY-MM-DD" 표준 형식으로 변환 (예: "2024-06-28").
   - 날짜 정보가 없으면 null.

3. **`items` (세부 품목 배열)**:
   원문에 여러 품목이 나열되어 있으면 개별 품목 객체로 분리하여 배열에 추가하십시오.

   - **`matchedWorkTypeId` (정수 또는 null)**:
     - 위 [표준 WorkType 정의 테이블]의 고유 ID 매핑.
     - 키워드 매핑 가이드:
       * `보` + 배관치수(D65, 50 등) ➔ 금속관벽체 또는 금속관입상 ID
       * `무` + 배관치수 ➔ 금속관벽체 또는 금속관입상 ID
       * `PVC` + 배관치수 ➔ PVC벽체 또는 PVC입상 ID
       * `보` + 사각치수(1600*600 등) / `보덕트` ➔ 보온덕트벽체 또는 보온덕트입상 ID
       * `무` + 사각치수 / `무덕트` ➔ 무보온덕트벽체 또는 무보온덕트입상 ID
       * `차열`, `차열마감`, `차열재` ➔ 차열재마감 ID
       * `오프`, `오프구`, `오픈`, `오픈구` ➔ 오픈구 ID
       * `단파`, `FVD`, `FD` ➔ 댐퍼팽창테이프 ID
       * `충전재마감`, `프레싱마감` ➔ 덕트마감 ID
       * `실리콘`, `실란트`, `구멍마감`, `틈새마감` ➔ 실란트마감벽체 ID

   - **`spec` (규격 문자열 또는 null)**:
     * **배관류**: 호칭경(직경 mm) 단일 정수 문자열 (예: "15", "20", "32", "50", "65", "80", "100", "125", "150", "200"). 'D', 'A', 'Ø', '파이', '보', '무' 등의 접두/접미사는 제거.
     * **덕트류**: `가로*세로` 단면 치수 (예: "1600*600", "800*400", "2000*500").
     * **특수/마감류**: 연결된 덕트/개구부의 `가로*세로` 치수 상속 (예: 차열재마감, 오픈구는 "2000*600", "800*200"). 치수가 없는 실란트/구멍마감/슬리브는 null.

   - **`quantity` (시공 수량, 부동소수점 Float)**:
     * **[공식]**: `[기본 개수]` × `[시공면 계수]`
     * **시공면 계수**:
       - `양면` (벽체): 1.0 (벽 앞/뒤 2면 시공 완료 시 1개소 인정)
       - `단면` (벽체): 0.5 (한쪽 면만 시공 시 0.5개소 인정)
       - `입상` (층간 관통): 1.0 (층당 1회 시공)
     * **기본 개수 산출**:
       - `*N` 또는 `N개` 표기가 있으면 N개 (예: `50*2` ➔ 2개, `150*3` ➔ 3개).
       - 수량 표기가 없으면 기본 1개.
     * **차열재마감(`차열재마감`) 수량 특수 공식**:
       - `[덕트 개수]` × `[도포 횟수(한벌/1번=1, 두벌/2번=2, N번=N)]` × `[시공면 계수(단면 0.5, 양면 1.0)]`
       - 예1: 차열 1번 + 단면 ➔ 1 × 1 × 0.5 = 0.5
       - 예2: 차열 2번 + 단면 ➔ 1 × 2 × 0.5 = 1.0
       - 예3: 차열 2번 + 양면 ➔ 1 × 2 × 1.0 = 2.0
       - 예4: 덕트 2개(*2) + 차열 2번 + 양면 ➔ 2 × 2 × 1.0 = 4.0

---

### [엄격한 제약 사항]
1. **임의 추정 금지 (No Hallucination)**: 원문에 명시되지 않은 치수, 수량, 날짜 등은 절대 임의로 추정하지 마십시오.
2. 입력된 각 레코드에 대해 반드시 순서를 유지하며 1:1로 대응되는 결과를 `records` 목록에 담아 출력하십시오.

---

### [Few-Shot 예시]
{FEW_SHOT_EXAMPLES}
"""
        return system_instruction.strip()

    @classmethod
    def build_user_content(cls, input_records: Sequence[InputRecord]) -> str:
        """입력 레코드 목록을 LLM에 전달할 JSON 형태 문자열로 직렬화합니다.

        Args:
            input_records (Sequence[InputRecord]): 입력 레코드 목록.

        Returns:
            str: 직렬화된 JSON 문자열.
        """
        payload = []
        for r in input_records:
            item = {"text": r.text}
            if r.location:
                item["location"] = r.location
            if r.workDate:
                item["workDate"] = r.workDate
            payload.append(item)

        return json.dumps(payload, ensure_ascii=False, indent=2)
