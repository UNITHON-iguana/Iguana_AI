"""OCR 결과 저장기(Saver) 모듈.

OCR 수행 결과(OCRResult)를 JSON 파일로 직렬화하여 파일 시스템에 영구 저장하거나,
테스트 및 API 응답 처리를 위해 인메모리(Memory)에 보관하는 저장기들을 제공합니다.
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .schemas import OCRResult, SaveSummary


class BaseOCRResultSaver(ABC):
    """OCR 결과 저장기의 추상 기반 클래스.

    단일 및 다중 OCR 실행 결과를 대상 목적지(파일 시스템, 메모리 등)에 저장하는 공통 인터페이스를 정의합니다.
    """

    @abstractmethod
    def save(
        self,
        result: OCRResult,
        destination: Any,
        file_name: Optional[str] = None,
    ) -> SaveSummary:
        """단일 OCRResult를 지정된 목적지에 저장합니다.

        Args:
            result (OCRResult): 저장할 OCR 실행 결과 객체.
            destination (Any): 저장 대상 디렉토리 경로 또는 저장소 식별자.
            file_name (Optional[str]): 커스텀 출력 파일명 (미지정 시 자동 생성).

        Returns:
            SaveSummary: 저장 작업 결과 요약 객체.

        Raises:
            NotImplementedError: 하위 클래스에서 구현되지 않은 경우 발생.
        """
        pass

    def save_batch(
        self,
        results: Sequence[OCRResult],
        destination: Any,
        batch_summary_name: Optional[str] = "batch_ocr_summary.json",
    ) -> List[SaveSummary]:
        """여러 OCRResult를 저장하고 선택적으로 일괄 요약(Batch Summary) JSON 파일을 생성합니다.

        Args:
            results (Sequence[OCRResult]): 저장할 OCR 결과 객체들의 시퀀스.
            destination (Any): 저장 대상 디렉토리 경로 또는 저장소 식별자.
            batch_summary_name (Optional[str]): 일괄 요약 파일명 (기본값: 'batch_ocr_summary.json').

        Returns:
            List[SaveSummary]: 각 저장 작업 결과 요약 객체들의 리스트.
        """
        # [변수] summaries: 각 결과 저장에 대한 요약 정보를 수집할 리스트
        summaries: List[SaveSummary] = []

        # [개별 결과 저장] 각 OCR 결과를 순회하며 개별 파일로 저장
        for res in results:
            summary = self.save(res, destination)
            summaries.append(summary)

        # [조건 검사] 배치 요약 파일명이 지정되어 있고 대상 경로가 로컬 파일 경로(str 또는 Path)인지 확인
        # (인메모리 저장이 아닌 파일 시스템 저장 시에만 전체 결과를 하나로 합친 배치 요약 파일을 생성하기 위함)
        if batch_summary_name and isinstance(destination, (str, Path)):
            # [변수] dest_dir: 대상 디렉토리의 Path 객체
            dest_dir = Path(destination)
            dest_dir.mkdir(parents=True, exist_ok=True)

            # [변수] summary_path: 최종 요약 파일 경로
            summary_path = dest_dir / batch_summary_name

            # [변수] all_data: 전체 결과를 직렬화 가능한 딕셔너리 리스트로 변환
            all_data = [r.to_dict() for r in results]

            try:
                # [JSON 직렬화] 한글 깨짐 방지(ensure_ascii=False) 및 들여쓰기 2칸 적용
                # [변수] json_str: 포맷팅된 통합 JSON 문자열
                json_str = json.dumps(all_data, ensure_ascii=False, indent=2)
                summary_path.write_text(json_str, encoding="utf-8")

                # [성공 요약 추가]
                summaries.append(
                    SaveSummary(
                        destination=str(summary_path.resolve()),
                        file_name=batch_summary_name,
                        success=True,
                        bytes_written=len(json_str.encode("utf-8")),
                        metadata={"is_batch_summary": True, "count": len(results)},
                    )
                )
            except Exception as e:
                # [실패 요약 추가] 요약 파일 저장 실패 시 에러 내용 기록
                summaries.append(
                    SaveSummary(
                        destination=str(summary_path),
                        file_name=batch_summary_name,
                        success=False,
                        error_message=str(e),
                    )
                )

        return summaries


class JsonFileOCRResultSaver(BaseOCRResultSaver):
    """OCR 처리 결과를 UTF-8 인코딩의 정형화된 JSON 파일로 저장하는 저장기 클래스."""

    def save(
        self,
        result: OCRResult,
        destination: Union[str, Path],
        file_name: Optional[str] = None,
    ) -> SaveSummary:
        """단일 OCRResult를 JSON 파일로 저장합니다.

        Args:
            result (OCRResult): 저장할 OCR 결과 객체.
            destination (Union[str, Path]): 저장할 대상 디렉토리 경로.
            file_name (Optional[str]): 저장할 파일명. 미지정 시 '{원본파일명}_ocr.json' 형식 자동 지정.

        Returns:
            SaveSummary: 파일 저장 성공 여부 및 기록된 바이트 정보 요약 객체.
        """
        # [변수] target_dir: 출력 디렉토리의 Path 객체
        target_dir = Path(destination)

        try:
            # [디렉토리 생성] 상위 디렉토리가 존재하지 않을 경우 자동 생성
            target_dir.mkdir(parents=True, exist_ok=True)

            # [조건 검사] 사용자 지정 파일명이 전달되지 않았는지 확인 (미지정 시 원본 파일의 확장자를 제거한 이름에 '_ocr.json'을 붙여 자동 생성)
            if file_name is None:
                # [변수] stem: 원본 파일명에서 확장자를 제외한 기본 이름
                stem = Path(result.file_name).stem
                file_name = f"{stem}_ocr.json"

            # [변수] output_path: 최종 JSON 파일 절대/상대 경로
            output_path = target_dir / file_name

            # [변수] data_dict: OCRResult를 JSON 직렬화 가능한 딕셔너리로 변환
            data_dict = result.to_dict()

            # [변수] json_text: UTF-8 포맷의 들여쓰기된 JSON 텍스트 문자열
            json_text = json.dumps(data_dict, ensure_ascii=False, indent=2)

            # [파일 쓰기] UTF-8 인코딩으로 JSON 파일 작성
            output_path.write_text(json_text, encoding="utf-8")

            # [변수] bytes_count: 저장된 파일의 총 바이트 수 계산
            bytes_count = len(json_text.encode("utf-8"))

            return SaveSummary(
                destination=str(output_path.resolve()),
                file_name=file_name,
                success=True,
                bytes_written=bytes_count,
            )
        except Exception as e:
            # [예외 처리] 파일 시스템 접근 오류 또는 쓰기 실패 시 실패 정보 반환
            return SaveSummary(
                destination=str(target_dir / (file_name or "unknown.json")),
                file_name=file_name or "unknown.json",
                success=False,
                error_message=str(e),
            )


class MemoryOCRResultSaver(BaseOCRResultSaver):
    """OCR 처리 결과를 파일 시스템에 기록하지 않고 인메모리 리스트에 딕셔너리 형태로 보관하는 저장기 클래스."""

    def __init__(self) -> None:
        """인메모리 저장소 리스트를 초기화합니다."""
        # [변수] self.saved_results: 저장된 결과 딕셔너리들을 보관하는 인메모리 리스트
        self.saved_results: List[Dict[str, Any]] = []

    def save(
        self,
        result: OCRResult,
        destination: Any = None,
        file_name: Optional[str] = None,
    ) -> SaveSummary:
        """단일 OCRResult를 딕셔너리로 변환하여 메모리 리스트에 추가합니다.

        Args:
            result (OCRResult): 저장할 OCR 결과 객체.
            destination (Any): 인메모리 저장소이므로 사용되지 않음 (기본값: None).
            file_name (Optional[str]): 가상 파일명.

        Returns:
            SaveSummary: 메모리 저장 결과 요약 객체.
        """
        # [변수] res_dict: OCRResult 객체를 딕셔너리로 변환
        res_dict = result.to_dict()
        self.saved_results.append(res_dict)

        # [변수] target_fname: 파일명 결정 (사용자 지정값 또는 결과 객체의 원본 파일명)
        target_fname = file_name or result.file_name

        return SaveSummary(
            destination="memory",
            file_name=target_fname,
            success=True,
            bytes_written=0,
            metadata={"record_index": len(self.saved_results) - 1},
        )

    def get_results(self) -> List[Dict[str, Any]]:
        """메모리에 저장된 모든 OCR 결과 딕셔너리 목록을 반환합니다.

        Returns:
            List[Dict[str, Any]]: 저장된 결과 딕셔너리들의 리스트.
        """
        return self.saved_results

    def clear(self) -> None:
        """메모리에 저장된 모든 결과 데이터를 삭제하고 초기화합니다."""
        self.saved_results.clear()
