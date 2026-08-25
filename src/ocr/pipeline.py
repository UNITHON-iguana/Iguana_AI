"""OCR 파이프라인(Pipeline) 모듈.

이미지 로드(Loader), Gemini OCR 엔진 분석(Engine), 결과 저장(Saver) 과정을 하나로 통합하여
단일 이미지, 배치 이미지, 디렉토리 단위의 일괄 처리를 조율합니다.
"""

from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Union

from .engine import BaseOCREngine, GeminiOCREngine
from .loader import BaseImageLoader, FileImageLoader
from .saver import BaseOCRResultSaver, JsonFileOCRResultSaver
from .schemas import LoadedImageData, OCRResult, SaveSummary


class OCRPipeline:
    """이미지 로딩, Gemini OCR 처리, 결과 저장을 유기적으로 연결하는 고수준 파이프라인 클래스.

    Attributes:
        loader (BaseImageLoader): 이미지 로드 구현체.
        engine (BaseOCREngine): OCR 분석 엔진 구현체.
        saver (Optional[BaseOCRResultSaver]): 결과 저장기 구현체.
    """

    def __init__(
        self,
        loader: Optional[BaseImageLoader] = None,
        engine: Optional[BaseOCREngine] = None,
        saver: Optional[BaseOCRResultSaver] = None,
    ):
        """파이프라인 컴포넌트들을 주입받아 초기화합니다.

        Args:
            loader (Optional[BaseImageLoader]): 이미지 로더 인스턴스 (미지정 시 FileImageLoader 사용).
            engine (Optional[BaseOCREngine]): OCR 엔진 인스턴스 (미지정 시 GeminiOCREngine 사용).
            saver (Optional[BaseOCRResultSaver]): 결과 저장기 인스턴스 (미지정 시 JsonFileOCRResultSaver 사용).
        """
        # [컴포넌트 초기화] 주입된 객체가 없으면 기본 클래스 인스턴스 생성
        self.loader = loader or FileImageLoader()
        self.engine = engine or GeminiOCREngine()
        self.saver = saver or JsonFileOCRResultSaver()

    def process_image(
        self,
        source: Any,
        output_dir: Optional[Union[str, Path]] = None,
        output_file_name: Optional[str] = None,
    ) -> OCRResult:
        """단일 이미지 소스에 대해 로드 -> OCR 분석 -> (선택적) 저장을 수행합니다.

        Args:
            source (Any): 이미지 경로(str, Path), 바이트(bytes), 또는 이미 로드된 LoadedImageData.
            output_dir (Optional[Union[str, Path]]): 결과 JSON 파일을 저장할 디렉토리 경로. 미지정 시 파일 저장 생략.
            output_file_name (Optional[str]): 저장할 JSON 파일명.

        Returns:
            OCRResult: OCR 실행 및 전사 결과 객체.
        """
        # [조건 검사] 입력 소스가 이미 메모리에 로드된 LoadedImageData 객체인지 확인 (중복 로드를 방지하고 즉시 처리하기 위함)
        if isinstance(source, LoadedImageData):
            # [변수] loaded_image: 전달받은 이미지 컨테이너 객체
            loaded_image = source
        else:
            # [변수] loaded_image: 로더를 통해 소스로부터 로드한 이미지 컨테이너 객체
            loaded_image = self.loader.load(source)

        # [엔진 실행] Gemini OCR 분석 수행
        # [변수] result: OCR 분석 결과 객체
        result = self.engine.process(loaded_image)

        # [조건 검사] 출력 디렉토리 경로가 지정되어 있고 저장기(Saver)가 설정되어 있는지 확인 (결과 파일 저장이 요청된 경우에만 저장 수행)
        if output_dir is not None and self.saver is not None:
            self.saver.save(result, destination=output_dir, file_name=output_file_name)

        return result

    def process_batch(
        self,
        sources: Sequence[Any],
        output_dir: Optional[Union[str, Path]] = None,
        on_progress: Optional[Callable[[int, int, OCRResult], None]] = None,
    ) -> List[OCRResult]:
        """여러 이미지 소스에 대해 순차적으로 OCR 처리를 수행합니다.

        Args:
            sources (Sequence[Any]): 처리할 이미지 소스들의 시퀀스.
            output_dir (Optional[Union[str, Path]]): 결과 파일들을 저장할 출력 디렉토리 경로.
            on_progress (Optional[Callable[[int, int, OCRResult], None]]): 진행 상태 알림 콜백 함수 (현재인덱스, 전체개수, 결과객체).

        Returns:
            List[OCRResult]: 각 이미지별 처리 결과 객체들의 리스트.
        """
        # [변수] results: 순차 처리된 결과들을 수집할 리스트
        results: List[OCRResult] = []

        # [변수] total: 전체 처리 대상 소스의 수
        total = len(sources)

        # [순차 배치 처리]
        for idx, src in enumerate(sources):
            # [변수] res: 개별 이미지 처리 결과 객체
            res = self.process_image(src, output_dir=output_dir)
            results.append(res)

            # [조건 검사] 진행률 콜백 함수가 등록되어 있는지 확인 (등록된 경우 현재 완료 건수와 진행 상태 전달)
            if on_progress:
                on_progress(idx + 1, total, res)

        # [조건 검사] 출력 디렉토리가 지정되어 있고 저장기가 존재하며 처리된 결과가 1건 이상인지 확인
        # (전체 배치 작업이 완료된 후 통합 배치 요약 JSON 파일을 생성하기 위함)
        if output_dir is not None and self.saver is not None and len(results) > 0:
            self.saver.save_batch(results, destination=output_dir)

        return results

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        extensions: Optional[Sequence[str]] = None,
        recursive: bool = False,
        on_progress: Optional[Callable[[int, int, OCRResult], None]] = None,
    ) -> List[OCRResult]:
        """지정된 디렉토리 내의 지원되는 모든 이미지를 탐색하여 일괄 처리합니다.

        Args:
            input_dir (Union[str, Path]): 이미지가 위치한 대상 디렉토리 경로.
            output_dir (Optional[Union[str, Path]]): 결과 JSON을 저장할 출력 디렉토리 경로.
            extensions (Optional[Sequence[str]]): 탐색할 이미지 확장자 목록.
            recursive (bool): 하위 디렉토리까지 재귀적으로 탐색할지 여부.
            on_progress (Optional[Callable[[int, int, OCRResult], None]]): 진행 콜백 함수.

        Returns:
            List[OCRResult]: 처리된 모든 이미지의 OCR 결과 객체 리스트.
        """
        # [조건 검사] 주입된 로더 객체에 디렉토리 일괄 로드 메서드(load_directory)가 지원되는지 확인
        # (FileImageLoader 등 디렉토리 탐색 최적화 메서드가 구현된 경우 이를 활용하기 위함)
        if hasattr(self.loader, "load_directory"):
            # [변수] images: load_directory를 통해 한 번에 수집된 이미지 데이터 목록
            images = self.loader.load_directory(
                input_dir, extensions=extensions, recursive=recursive
            )
        else:
            # [조건 분기] 일반 로더일 경우 디렉토리 내 파일을 직접 탐색하여 순차 로드
            # [변수] p: 입력 디렉토리 Path 객체
            p = Path(input_dir)
            # [변수] images: 단일 파일 로드 방식으로 수집된 이미지 데이터 목록
            images = [self.loader.load(f) for f in p.glob("*") if f.is_file()]

        # [배치 처리 위임] 수집된 이미지 목록에 대해 배치 실행 수행
        return self.process_batch(images, output_dir=output_dir, on_progress=on_progress)
