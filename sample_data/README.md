# 현장노트 AI 샘플 데이터셋 (`sample_data/`)

이 디렉토리는 건설 현장에서 수집된 실제 보드판 사진(`photos/`)과 사무 담당자가 최종 정리한 엑셀 파일(`BS.xlsm` 및 추출본)을 포함하는 **Ground Truth 벤치마크 데이터셋**입니다.

---

## 📂 디렉토리 구조

```text
ai/sample_data/
├── photos/                         # 실제 현장 사진 32장 (378.jpg ~ 409.jpg)
│   ├── 378.jpg
│   ├── 379.jpg
│   └── ...
│
├── excel/
│   ├── BS.xlsm                     # 실무자 작성 원본 엑셀 (매크로 포함)
│   ├── BS_photo_daeji.json         # 사진대지 시트 파싱 JSON (photo_no & file_name 매핑 포함)
│   ├── BS_photo_daeji.csv          # 사진대지 시트 파싱 CSV
│   ├── BS_daily_summary.json       # 집계표 시트 파싱 JSON (159개 일일 집계 항목)
│   ├── BS_cumulative.json          # 누계 시트 파싱 JSON (105개 기성 누계 항목)
│   └── sheet_structure.md          # 엑셀 시트 컬럼 ↔ 도메인 모델 매핑 명세
│
├── dataset_mapping.md              # 32개 사진 ↔ 엑셀 1:1 매핑 정답표 (Ground Truth)
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
