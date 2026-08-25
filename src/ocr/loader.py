"""OCR 이미지 로더 모듈.

로컬 파일 경로, 디렉토리 일괄 검색, 인메모리 바이트 스트림 및 Base64 인코딩 문자열 등
다양한 소스로부터 이미지를 읽어 OCR 엔진 입력용 LoadedImageData 컨테이너로 변환합니다.
"""

from abc import ABC, abstractmethod
import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .schemas import LoadedImageData


class BaseImageLoader(ABC):
    """OCR 이미지 로더의 추상 기반 클래스.

    다양한 형태(파일, 바이트, 스트림 등)의 이미지 입력을 표준화된 LoadedImageData 객체로 변환하는 인터페이스를 정의합니다.
    """

    @abstractmethod
    def load(self, source: Any) -> LoadedImageData:
        """단일 소스로부터 이미지를 로드하여 LoadedImageData 객체를 생성합니다.

        Args:
            source (Any): 이미지 소스 (파일 경로 문자열, Path 객체, 바이너리 바이트 등).

        Returns:
            LoadedImageData: 로드된 이미지 데이터 컨테이너.

        Raises:
            NotImplementedError: 하위 클래스에서 구현되지 않은 경우 발생.
        """
        pass

    def load_batch(self, sources: Sequence[Any]) -> List[LoadedImageData]:
        """여러 이미지 소스를 순차적으로 로드하여 리스트로 반환합니다.

        Args:
            sources (Sequence[Any]): 이미지 소스들의 시퀀스(리스트, 튜플 등).

        Returns:
            List[LoadedImageData]: 성공적으로 로드된 LoadedImageData 객체들의 리스트.

        Raises:
            RuntimeError: 특정 소스 로드 중 예외가 발생한 경우 원본 예외와 함께 발생.
        """
        # [변수] loaded: 로드된 이미지 데이터 객체들을 수집할 리스트
        loaded: List[LoadedImageData] = []

        for src in sources:
            try:
                # [처리] 각 소스에 대해 load 메서드 수행 후 수집
                loaded_img = self.load(src)
                loaded.append(loaded_img)
            except Exception as e:
                # [예외 처리] 로드 실패한 소스 정보와 함께 런타임 오류 발생
                raise RuntimeError(f"Failed to load image from {src}: {e}") from e

        return loaded


class FileImageLoader(BaseImageLoader):
    """로컬 파일 시스템의 단일 파일 경로 또는 디렉토리로부터 이미지를 로드하는 로더 클래스."""

    # [상수] 지원하는 표준 이미지 확장자 목록
    SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def _infer_mime_type(self, path: Path) -> str:
        """파일 경로의 확장자를 분석하여 적절한 MIME 타입을 추론합니다.

        Args:
            path (Path): 대상 이미지 파일의 Path 객체.

        Returns:
            str: 추론된 MIME 타입 문자열 (예: 'image/jpeg', 'image/png').
        """
        # [변수] suffix: 소문자로 정규화된 파일 확장자
        suffix = path.suffix.lower()

        # [조건 검사] 확장자가 JPEG 포맷(.jpg, .jpeg)인지 확인 (가장 일반적인 사진 포맷 매핑)
        if suffix in (".jpg", ".jpeg"):
            return "image/jpeg"
        # [조건 검사] 확장자가 PNG 포맷(.png)인지 확인 (무손실 압축 포맷 매핑)
        elif suffix == ".png":
            return "image/png"
        # [조건 검사] 확장자가 WebP 포맷(.webp)인지 확인 (웹 최적화 이미지 포맷 매핑)
        elif suffix == ".webp":
            return "image/webp"
        # [조건 검사] 확장자가 BMP 포맷(.bmp)인지 확인 (비압축 비트맵 포맷 매핑)
        elif suffix == ".bmp":
            return "image/bmp"

        # [변수] guessed_type: 시스템 mimetypes 모듈을 통해 추측한 MIME 타입
        guessed_type, _ = mimetypes.guess_type(str(path))
        # [조건 검사] 추측된 MIME 타입이 유효한지 확인하고, 실패 시 기본값인 'image/jpeg' 반환
        return guessed_type or "image/jpeg"

    def load(self, source: Union[str, Path]) -> LoadedImageData:
        """단일 로컬 이미지 파일을 읽어 LoadedImageData 컨테이너로 반환합니다.

        Args:
            source (Union[str, Path]): 이미지 파일의 경로 (문자열 또는 Path 객체).

        Returns:
            LoadedImageData: 읽어들인 바이너리 데이터 및 파일 정보가 포함된 컨테이너.

        Raises:
            FileNotFoundError: 지정한 경로에 파일이 존재하지 않거나 디렉토리인 경우 발생.
        """
        # [변수] path: 전달받은 소스를 Path 객체로 정규화
        path = Path(source)

        # [조건 검사] 대상 경로가 실제 존재하는 파일인지 확인 (존재하지 않거나 디렉토리인 경우 파일 읽기가 불가능하므로 검증)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")

        # [변수] mime_type: 파일 확장자 기반으로 추론된 MIME 타입
        mime_type = self._infer_mime_type(path)

        # [파일 읽기] 바이너리 읽기 모드로 이미지 바이트 데이터 로드
        with open(path, "rb") as f:
            # [변수] image_bytes: 원시 바이너리 바이트 데이터
            image_bytes = f.read()

        # [변수] loaded_data: LoadedImageData 객체 생성 및 메타데이터 설정
        return LoadedImageData(
            image_bytes=image_bytes,
            mime_type=mime_type,
            source_id=str(path.resolve()),
            file_name=path.name,
            metadata={"file_size_bytes": len(image_bytes), "file_path": str(path)},
        )

    def load_directory(
        self,
        dir_path: Union[str, Path],
        extensions: Optional[Sequence[str]] = None,
        recursive: bool = False,
    ) -> List[LoadedImageData]:
        """지정된 디렉토리 내의 지원되는 모든 이미지 파일을 탐색하여 일괄 로드합니다.

        Args:
            dir_path (Union[str, Path]): 검색할 대상 디렉토리 경로.
            extensions (Optional[Sequence[str]]): 허용할 확장자 목록. 미지정 시 기본 지원 확장자 사용.
            recursive (bool): 하위 디렉토리까지 재귀적으로 탐색할지 여부.

        Returns:
            List[LoadedImageData]: 파일명 기준으로 정렬되어 로드된 이미지 데이터 리스트.

        Raises:
            NotADirectoryError: 지정한 디렉토리 경로가 존재하지 않거나 디렉토리가 아닌 경우 발생.
        """
        # [변수] target_dir: 대상 디렉토리의 Path 객체
        target_dir = Path(dir_path)

        # [조건 검사] 입력된 경로가 실제 존재하는 유효한 디렉토리인지 확인 (존재하지 않으면 파일 탐색을 진행할 수 없으므로 예외 발생)
        if not target_dir.is_dir():
            raise NotADirectoryError(f"Directory not found: {target_dir}")

        # [변수] exts: 소문자로 정규화된 유효 확장자 튜플
        # [조건 검사] 사용자 정의 확장자 목록이 제공되었는지 확인하고, 없으면 기본 지원 확장자(SUPPORTED_EXTENSIONS) 채택
        exts = tuple(e.lower() for e in (extensions or self.SUPPORTED_EXTENSIONS))

        # [변수] glob_pattern: 하위 탐색 범위 패턴
        # [조건 검사] recursive 플래그가 True인지 확인하여 하위 전체 탐색('**/*') 또는 현재 디렉토리 탐색('*') 선택
        glob_pattern = "**/*" if recursive else "*"

        # [변수] image_paths: 조건에 부합하는 파일들을 수집하고 파일명 기준으로 오름차순 정렬한 경로 리스트
        image_paths = sorted(
            [p for p in target_dir.glob(glob_pattern) if p.is_file() and p.suffix.lower() in exts],
            key=lambda p: p.name,
        )

        # [일괄 로드] 수집된 경로 각각을 로드하여 리스트로 반환
        return [self.load(p) for p in image_paths]


class BytesImageLoader(BaseImageLoader):
    """메모리 상의 원시 바이너리 바이트(bytes) 또는 Base64 인코딩 문자열로부터 이미지를 로드하는 로더 클래스."""

    def load(
        self,
        source: Union[bytes, str],
        mime_type: str = "image/jpeg",
        source_id: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> LoadedImageData:
        """바이트 스트림 또는 Base64 문자열을 디코딩하여 LoadedImageData 객체로 변환합니다.

        Args:
            source (Union[bytes, str]): 바이너리 바이트 데이터 또는 Base64 인코딩 문자열 (Data URI 포함 가능).
            mime_type (str): 이미지 MIME 타입 (기본값: 'image/jpeg').
            source_id (Optional[str]): 소스 식별자 (미지정 시 'memory_buffer').
            file_name (Optional[str]): 가상 파일명 (미지정 시 'image.jpg').

        Returns:
            LoadedImageData: 메모리에서 생성된 이미지 데이터 컨테이너.

        Raises:
            TypeError: source 매개변수가 bytes 또는 str 타입이 아닌 경우 발생.
        """
        # [조건 검사] 입력 소스가 문자열(Base64 인코딩 또는 Data URI 형태)인지 확인
        if isinstance(source, str):
            # [조건 검사] 'data:image/...;base64,...' 형태의 Data URI 스킴이 포함되어 있는지 확인 (웹 API나 프론트엔드 전송 포맷 지원)
            if source.startswith("data:") and ";base64," in source:
                # [변수] header, encoded: Data URI 헤더부와 실제 Base64 데이터부 분리
                header, encoded = source.split(";base64,", 1)
                # [변수] mime_type: 헤더에서 추출한 MIME 타입으로 갱신
                mime_type = header.replace("data:", "")
                # [변수] image_bytes: Base64 문자열을 원시 바이트로 디코딩
                image_bytes = base64.b64decode(encoded)
            else:
                # [변수] image_bytes: 순수 Base64 문자열 디코딩
                image_bytes = base64.b64decode(source)
        # [조건 검사] 입력 소스가 이미 바이너리 바이트(bytes) 형식인지 확인
        elif isinstance(source, bytes):
            # [변수] image_bytes: 별도 디코딩 없이 원시 바이트 그대로 사용
            image_bytes = source
        else:
            # [예외 처리] 지원하지 않는 타입인 경우 타입 에러 발생
            raise TypeError(f"Expected bytes or str, got {type(source)}")

        # [변수] src_id: 소스 식별자 결정 (미지정 시 기본값 'memory_buffer' 사용)
        src_id = source_id or "memory_buffer"

        # [변수] fname: 파일명 결정 (미지정 시 기본값 'image.jpg' 사용)
        fname = file_name or "image.jpg"

        return LoadedImageData(
            image_bytes=image_bytes,
            mime_type=mime_type,
            source_id=src_id,
            file_name=fname,
            metadata={"source_type": "bytes"},
        )
