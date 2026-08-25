"""표준 WorkType 메타데이터 저장소(Repository) 및 드라이버 모듈.

외부 서버(HTTP), 데이터베이스(DB), 또는 로컬 JSON 파일로부터 WorkType 목록을
조회하는 공통 인터페이스 및 구현체를 제공합니다.
"""

from abc import ABC, abstractmethod
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .schemas import WorkTypeItem

logger = logging.getLogger(__name__)


class BaseWorkTypeRepository(ABC):
    """WorkType 저장소의 추상 인터페이스.

    외부 서버 HTTP API, DB, 로컬 파일 등 다양한 데이터 소스로부터
    표준 WorkType 목록을 조회하는 메서드를 정의합니다.
    """

    @abstractmethod
    def get_work_types(self) -> List[WorkTypeItem]:
        """전체 표준 WorkType 목록을 반환합니다.

        Returns:
            List[WorkTypeItem]: 표준 WorkType 객체 목록.
        """
        pass

    @abstractmethod
    def get_by_id(self, work_type_id: int) -> Optional[WorkTypeItem]:
        """ID로 특정 WorkType을 조회합니다.

        Args:
            work_type_id (int): 조회할 WorkType ID.

        Returns:
            Optional[WorkTypeItem]: 일치하는 WorkType 객체 또는 None.
        """
        pass

    @abstractmethod
    def get_by_name(self, name: str) -> Optional[WorkTypeItem]:
        """공종명(이름)으로 특정 WorkType을 조회합니다.

        Args:
            name (str): 조회할 공종명.

        Returns:
            Optional[WorkTypeItem]: 일치하는 WorkType 객체 또는 None.
        """
        pass


class LocalJsonWorkTypeRepository(BaseWorkTypeRepository):
    """로컬 JSON 파일(work_types.json) 기반 WorkType 저장소 구현체.

    파일을 메모리에 캐싱하여 빠른 조회를 지원하며,
    파일이 갱신된 경우 `reload()` 메서드로 재로드가 가능합니다.
    """

    DEFAULT_JSON_PATH = (
        Path(__file__).resolve().parent.parent.parent / "dataset" / "work_types.json"
    )

    def __init__(self, json_path: Optional[Path] = None) -> None:
        """LocalJsonWorkTypeRepository 초기화.

        Args:
            json_path (Optional[Path]): work_types.json 파일 경로.
                지정되지 않을 경우 프로젝트 기본 경로(dataset/work_types.json)를 사용합니다.
        """
        self.json_path = Path(json_path) if json_path else self.DEFAULT_JSON_PATH
        self._cache_by_id: Dict[int, WorkTypeItem] = {}
        self._cache_by_name: Dict[str, WorkTypeItem] = {}
        self._items: List[WorkTypeItem] = []
        self.load()

    def load(self) -> None:
        """JSON 파일로부터 WorkType 목록을 메모리에 로드 및 캐싱합니다."""
        if not self.json_path.exists():
            logger.warning(
                f"WorkType JSON 파일이 존재하지 않습니다: {self.json_path}. 빈 목록으로 초기화합니다."
            )
            self._items = []
            self._cache_by_id = {}
            self._cache_by_name = {}
            return

        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            items: List[WorkTypeItem] = []
            cache_by_id: Dict[int, WorkTypeItem] = {}
            cache_by_name: Dict[str, WorkTypeItem] = {}

            for entry in data:
                item = WorkTypeItem(
                    id=entry["id"],
                    name=entry["name"],
                    description=entry.get("description"),
                )
                items.append(item)
                cache_by_id[item.id] = item
                cache_by_name[item.name] = item

            self._items = items
            self._cache_by_id = cache_by_id
            self._cache_by_name = cache_by_name
            logger.info(
                f"WorkType {len(self._items)}개 로드 완료 ({self.json_path})"
            )
        except Exception as e:
            logger.error(f"WorkType JSON 파일 로드 실패 ({self.json_path}): {e}")
            raise

    def reload(self) -> None:
        """캐시를 비우고 JSON 파일을 다시 로드합니다."""
        self.load()

    def get_work_types(self) -> List[WorkTypeItem]:
        """전체 표준 WorkType 목록을 반환합니다.

        Returns:
            List[WorkTypeItem]: 표준 WorkType 객체 목록.
        """
        return list(self._items)

    def get_by_id(self, work_type_id: int) -> Optional[WorkTypeItem]:
        """ID로 특정 WorkType을 조회합니다.

        Args:
            work_type_id (int): 조회할 WorkType ID.

        Returns:
            Optional[WorkTypeItem]: 일치하는 WorkType 객체 또는 None.
        """
        return self._cache_by_id.get(work_type_id)

    def get_by_name(self, name: str) -> Optional[WorkTypeItem]:
        """공종명(이름)으로 특정 WorkType을 조회합니다.

        Args:
            name (str): 조회할 공종명.

        Returns:
            Optional[WorkTypeItem]: 일치하는 WorkType 객체 또는 None.
        """
        return self._cache_by_name.get(name)
