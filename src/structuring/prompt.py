"""텍스트 정형화 시스템 프롬프트 및 Few-shot 예시 빌더 모듈.

WorkType 저장소로부터 조회한 최신 공종 목록과 도메인 파싱/수량 연산 규칙을
결합하여 Gemini 모델에 전달할 System Instruction 및 User Content를 동적으로 생성합니다.
"""

import json
from typing import List, Sequence

from .schemas import InputRecord, WorkTypeItem

FEW_SHOT_EXAMPLES = """
[Few-Shot 예시 1: 다중 배관 규격 및 단면/양면 (완전 정상 케이스)]
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
        { "matchedWorkTypeId": 101, "workType": "금속관벽체", "spec": "20", "quantity": 2.0, "evidence": "보 20*2 양면", "confidence": "HIGH" },
        { "matchedWorkTypeId": 101, "workType": "금속관벽체", "spec": "100", "quantity": 1.0, "evidence": "100 양면", "confidence": "HIGH" },
        { "matchedWorkTypeId": 101, "workType": "금속관벽체", "spec": "50", "quantity": 1.0, "evidence": "50 양면", "confidence": "HIGH" }
      ]
    },
    {
      "location": "3동 31층",
      "workDate": "2024-06-29",
      "items": [
        { "matchedWorkTypeId": 101, "workType": "금속관벽체", "spec": "150", "quantity": 0.5, "evidence": "보 150 단면", "confidence": "HIGH" },
        { "matchedWorkTypeId": 101, "workType": "금속관벽체", "spec": "50", "quantity": 1.5, "evidence": "50*3 단면", "confidence": "HIGH" }
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
        { "matchedWorkTypeId": 202, "workType": "무보온덕트벽체", "spec": "2000*600", "quantity": 0.5, "evidence": "무 2000*600 단면", "confidence": "HIGH" },
        { "matchedWorkTypeId": 301, "workType": "차열재마감", "spec": "2000*600", "quantity": 0.5, "evidence": "차열마감1번 단면", "confidence": "HIGH" }
      ]
    },
    {
      "location": "지하1층",
      "workDate": "2024-07-01",
      "items": [
        { "matchedWorkTypeId": 201, "workType": "보온덕트벽체", "spec": "1000*500", "quantity": 2.0, "evidence": "보 1000*500*2 양면", "confidence": "HIGH" },
        { "matchedWorkTypeId": 301, "workType": "차열재마감", "spec": "1000*500", "quantity": 4.0, "evidence": "차열마감2번 양면", "confidence": "HIGH" }
      ]
    }
  ]
}

[Few-Shot 예시 3: Negative/노이즈 케이스 - 비공사 텍스트, 안전 슬로건, 공지사항]
입력:
[
  {
    "text": "현장 안전 제일 정리정돈 철저 및 안전모 착용",
    "location": "101동 1층",
    "workDate": "2024-07-02"
  },
  {
    "text": "오전 TBM 실시 및 위험성 평가 회의 진행",
    "location": null,
    "workDate": "2024-07-02"
  }
]
출력:
{
  "records": [
    {
      "location": "101동 1층",
      "workDate": "2024-07-02",
      "items": []
    },
    {
      "location": null,
      "workDate": "2024-07-02",
      "items": []
    }
  ]
}

[Few-Shot 예시 4: Negative/결측 케이스 - 규격이나 공종이 모호하여 파싱 불가]
입력:
[
  {
    "text": "벽체 배관 보수 작업 완료 및 점검 확인",
    "location": "지하 2층",
    "workDate": "2024-07-03"
  },
  {
    "text": "보강 작업 완료 (치수 미기재)",
    "location": "103동",
    "workDate": null
  }
]
출력:
{
  "records": [
    {
      "location": "지하 2층",
      "workDate": "2024-07-03",
      "items": []
    },
    {
      "location": "103동",
      "workDate": null,
      "items": []
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

================================================================================
🚨 [최우선 핵심 철학: 잘못된 정보 출력 < 미출력(Null / 빈 리스트)]
1. 건설 현장 데이터의 신뢰성을 위해, 잘못된 추측이나 허위 생성(Hallucination)은 시스템에 치명적입니다.
2. 확실하지 않거나 모호한 정보는 억지로 추측하여 채우지 말고, 반드시 `null` 또는 `items: []`로 포기하십시오. 사용자가 직접 검수 화면에서 확인하고 입력하도록 하는 것이 훨씬 더 안전하고 바람직합니다.
3. 원문에 명시적인 텍스트 근거(치수, 공종명 등)가 없다면 절대 임의로 상상하거나 유추하지 마십시오.
================================================================================

---

### [표준 WorkType (공종/품목) 정의 테이블]
원문의 작업 내용을 분석하여 가장 적합한 WorkType의 `id`를 `matchedWorkTypeId`에 매핑하십시오. 매칭되는 항목이 없거나 불확실한 경우 반드시 `null`로 지정하십시오.

{work_types_block}

---

### [필드별 파싱 및 엄격한 규칙]

1. **`location` (시공 위치)**:
   - 동, 층, 세대, 구역 정보가 원문에 명시된 경우에만 추출 (예: "지하4", "3동 38층", "3동 1.2세대 31층").
   - 위치 정보가 전혀 없거나 식별 불가능한 경우 반드시 null.

2. **`workDate` (작업 일자)**:
   - 원문에 날짜가 명시된 경우 "YYYY-MM-DD" 표준 형식으로 변환 (예: "2024-06-28").
   - 날짜 정보가 없으면 반드시 null.

3. **`items` (세부 품목 배열 - 올-오어-너씽 원칙)**:
   - 비공사 텍스트(안전구호, 정리정돈, TBM, 회의 등)이거나 공종/규격이 전혀 특정되지 않는 경우 반드시 **`items: []` (빈 리스트)**로 응답하십시오.
   - 여러 품목이 있을 때 명확한 근거가 있는 항목만 추출하되, 근거가 모호하거나 누락된 항목은 생성하지 마십시오.

   - **`matchedWorkTypeId` (정수 또는 null)**:
     - 위 [표준 WorkType 정의 테이블]의 고유 ID 매핑.
     - ⚠️ **단어 오인 주의**: 단일 글자 '보', '무'가 '보강', '보온재', '무소음', '정보' 등 일반 단어의 일부로 쓰인 경우 절대 공종으로 매핑하지 마십시오.
     - 명확한 배관/덕트 치수와 결합된 축약어(`보 50`, `무 1600*600`, `PVC 100` 등)일 때만 해당 공종 ID로 매핑하십시오.

   - **`spec` (규격 문자열 또는 null)**:
     * **배관류**: 호칭경(직경 mm) 단일 정수 문자열 (예: "15", "20", "32", "50", "65", "80", "100", "125", "150", "200"). 'D', 'A', 'Ø', '파이', '보', '무' 등의 접두/접미사는 제거.
     * **덕트류**: `가로*세로` 단면 치수 (예: "1600*600", "800*400", "2000*500").
     * **특수/마감류**: 연결된 덕트/개구부의 `가로*세로` 치수 상속. 원문에 명시된 치수가 없으면 반드시 null.

   - **`quantity` (시공 수량, 부동소수점 Float)**:
     * **[공식]**: `[기본 개수]` × `[시공면 계수]`
     * **시공면 계수**:
       - `양면` (벽체): 1.0 (벽 앞/뒤 2면 시공 완료 시 1개소 인정)
       - `단면` (벽체): 0.5 (한쪽 면만 시공 시 0.5개소 인정)
       - `입상` (층간 관통): 1.0 (층당 1회 시공)
     * **기본 개수 산출**:
       - `*N` 또는 `N개` 표기가 있으면 N개 (예: `50*2` ➔ 2개, `150*3` ➔ 3개). 수량 표기 없으면 기본 1개.

   - **`evidence` (추출 근거 텍스트)**:
     - 원문에서 이 품목과 규격, 수량을 추출한 실제 단어/문구 (예: "보 20*2 양면"). 근거가 없으면 "NONE".

   - **`confidence` (신뢰도: "HIGH" | "MEDIUM" | "LOW")**:
     - 원문에 공종, 규격, 수량이 모두 명확히 적혀 있으면 "HIGH".
     - 일부 단어가 모호하거나 추정 요소가 있으면 반드시 "LOW".

---

### [엄격한 제약 사항]
1. **임의 추정 절대 금지 (Zero Hallucination)**: 원문에 없는 정보는 절대 생성하지 마십시오.
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
