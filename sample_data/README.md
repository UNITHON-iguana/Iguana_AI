# 현장노트 AI 샘플 데이터셋 (`sample_data/`)

이 디렉토리는 건설 현장에서 수집된 실제 보드판 사진(`photos/`)과 사무 담당자가 최종 정리한 엑셀 파일(`BS.xlsm` 및 추출본)을 포함하는 **Ground Truth 벤치마크 데이터셋**입니다.

---

## 📂 디렉토리 구조

```text
ai/sample_data/
├── test_samples/                   # [단위 테스트 대상 샘플 데이터] 32건 Ground Truth
│   ├── photos/                     # 현장 원본 사진 32장 (378.jpg ~ 409.jpg)
│   ├── tables/                     # 크롭된 보드판 정답 이미지 32장
│   └── annotations/                # 정답 라벨링 및 실무자 엑셀 데이터
│       ├── BS.xlsm                 # 실무자 작성 원본 엑셀 장부 (매크로 포함)
│       ├── photo_daeji.json        # 사진대지 시트 정답 JSON (32건 레코드)
│       ├── photo_daeji.csv         # 사진대지 시트 정답 CSV
│       ├── daily_summary.json      # 일일 집계표 시트 정답 JSON (159개 항목)
│       ├── cumulative.json         # 기성 누계표 시트 정답 JSON (105개 항목)
│       ├── ground_truth_mapping.md # 32개 사진 ↔ 엑셀 1:1 정답 매핑표
│       └── sheet_structure.md      # 엑셀 시트 컬럼 ↔ 도메인 모델 매핑 명세
│
├── playground_prompt.md            # Gemini Playground 테스트용 프롬프트
├── 집계 방식.md                     # 공종별(배관/덕트/마감) 도메인 집계 규칙 분석서
└── README.md                       # 본 안내 문서
```

---

## 🎯 주요 활용 목적

1. **AI OCR 및 BBox 검증**:
   - `photos/` 내 사진에서 보드판 영역 검출 및 텍스트 인식 성능 평가.
2. **LLM 프롬프트 엔지니어링**:
   - `dataset_mapping.md`의 실제 보드판 텍스트 $\rightarrow$ 구조화 필드 매핑 관계를 Few-shot 예시로 활용.
3. **End-to-End 파이프라인 테스트**:
   - 업로드 $\rightarrow$ 영역 분리 $\rightarrow$ OCR $\rightarrow$ LLM 구조화 $\rightarrow$ 엑셀 내보내기 전체 파이프라인의 회귀 테스트 및 정확도 벤치마킹.
