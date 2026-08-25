# AI / OCR / LLM 백엔드 파이프라인 지침 (`ai/`)

이 디렉토리는 **"현장노트 AI"**의 AI 백엔드 코어가 위치하는 공간입니다.
사진 OCR, 보드판-작업사진 분리, LLM 기반 공종 분류 및 데이터 구조화를 담당합니다.

---

## 🎯 핵심 역할 및 파이프라인

1. **보드판/작업사진 영역 분리 (Object Detection / Cropping) ✅ 구현 완료**
   - OpenCV 기반 화이트 컬러 마스킹 + Canny Edge + Morphological Dilation 기법을 활용한 보드판 표 영역 BBox 자동 탐지 및 크롭.
   - 모듈: `src/table_crop/` (`loader.py`, `cropper.py`, `saver.py`, `pipeline.py`)
   - 실행 스크립트: `scripts/crop_tables.py`
   - 단위 테스트: `tests/test_table_cropper.py`
2. **OCR 엔진 연동 (Text Extraction) 🔜 예정**
   - 보드판 크롭 영역에서 텍스트(한글/숫자/기호)를 정밀하게 추출하고 원문(Raw OCR Text)을 보존.
3. **LLM 기반 공종 분류 및 데이터 구조화 (LLM Processing) 🔜 예정**
   - OCR 원문을 파싱하여 표준 도메인 필드로 구조화:
     - `작업일`, `현장명`, `동·층·구역`, `공종(골조/설비/마감/전기 등)`, `작업내용`, `시공단계(시공전/시공중/시공후)`, `자재명`, `수량`, `단위`, `특이사항`
4. **신뢰도 평가 및 검증**
   - 추출 필드별 신뢰도(Confidence) 계산.
   - 사진에 명시되지 않은 값은 절대 임의 추정(Hallucination)하지 않고 `None` 및 `확인 필요(needs_review)` 처리.
5. **API 인터페이스**
   - 웹 서버 및 프론트엔드와 통신할 수 있는 분석 요청/결과 API(FastAPI 등) 엔드포인트 제공.

---

## 📁 `ai/` 디렉토리 구조 및 구현 현황

```
ai/
├── src/
│   ├── table_crop/          # [완료] 표(보드판) 영역 검출 및 크롭 모듈
│   ├── ocr/                 # [완료] Gemini 비전 기반 보드판 OCR 추출 모듈
│   ├── structuring/         # [완료] LLM 공사 텍스트 정형화 및 공종 분류 모듈
│   └── server/              # [완료] FastAPI 기반 통합 분석 REST API 서버
├── scripts/                 # [Git 추적] 프로덕션 서비스 및 모듈 CLI 구동 스크립트
│   ├── run_server.py        # FastAPI 서버 구동
│   ├── crop_tables.py       # 표 크롭 일괄 처리 CLI
│   ├── run_ocr.py           # OCR 일괄 처리 CLI
│   ├── run_structuring.py   # 텍스트 정형화/공종 분류 CLI
│   └── experiments/         # [Git 제외] 모델 파인튜닝, 평가, 데이터셋 전처리, 모니터링 스크립트
├── tests/                   # [Git 추적] 순수 단위 테스트 코드 및 최소 Mock fixture
├── reports/                 # [Git 제외] 성능 평가/오차 분석 리포트 및 실행 산출물
├── sample_data/             # 현장 샘플 사진 및 공사 데이터
└── AGENTS.md                # AI 파이프라인 지침
```

---

## 📜 디렉토리 및 스크립트 관리 규칙

1. **서비스 실행용 스크립트 (`ai/scripts/`)**:
   - 프로덕션 배포, 통합 서버 구동(`run_server.py`), 핵심 모듈 CLI 등 실제 서비스 운영에 필요한 스크립트만 저장 (Git 관리 대상).
2. **개발/실험/파인튜닝 스크립트 (`ai/scripts/experiments/` 또는 `ai/scripts/dev/`)**:
   - LoRA/파인튜닝 학습, 체크포인트 평가, 데이터셋 정제/중복제거, Colab 연동/GPU 모니터링 등 연구/실험용 스크립트는 반드시 `scripts/experiments/`에 저장 (`.gitignore` 적용).
3. **단위 테스트 디렉토리 (`ai/tests/`)**:
   - 순수 단위/통합 테스트 코드(`test_*.py`)와 필수 Mock fixture 데이터만 유지합니다. 벤치마크 및 성능 평가 결과 파일은 `reports/eval/`로 격리 저장합니다.
4. **에이전트 준수 사항**:
   - 에이전트는 파일 및 스크립트 생성 시 목적을 명확히 판단하여 올바른 디렉토리에 생성해야 합니다.

---

## 🛠️ 모듈 사용 가이드 (Table Cropper)

```python
from src.table_crop.pipeline import TableCropPipeline
from src.table_crop.cropper import TableCropper

# 파이프라인 인스턴스 생성
pipeline = TableCropPipeline(cropper=TableCropper(filter_bottom_left=True))

# 1. 단일 이미지 처리
crop_res, save_res = pipeline.process_single(
    source="sample.jpg",
    destination="output_dir",
    file_name="crop_sample.jpg"
)

# 2. 디렉토리 일괄 처리
results = pipeline.process_directory(
    input_dir="input_photos/",
    output_dir="cropped_tables/"
)
```

---

## ⚠️ 핵심 구현 원칙
- **할루시네이션 엄격 방지**: LLM 프롬프트 및 응답 검증(Pydantic 등)을 통해 사진에 없는 정보의 임의 추정을 원천 차단.
- **예외 처리**: 비정형 보드판 서식, 저화질 이미지, 일부 필드 누락 시에도 전체 프로세스가 중단되지 않고 부분 성공/검수 대상 처리 가능하도록 설계.
- **스크립트 격리 원칙 준수**: 서비스 운영 스크립트 외의 연구/실험 코드가 Git 트리에 혼입되지 않도록 격리.
