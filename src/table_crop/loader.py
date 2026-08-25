"""Image loader module for table cropping.

Provides abstract and concrete image loaders to support various data sources
such as local files, directories, and in-memory byte streams (e.g. FastAPI/S3).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np


@dataclass
class LoadedImage:
    """Represents an image loaded into memory."""
    image: np.ndarray
    source_id: str
    file_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def channels(self) -> int:
        return self.image.shape[2] if self.image.ndim > 2 else 1


class BaseImageLoader(ABC):
    """Abstract base class for loading images."""

    @abstractmethod
    def load(self, source: Any) -> LoadedImage:
        """Load a single image from the given source.

        Args:
            source: Source identifier or object (file path, bytes, etc.)

        Returns:
            LoadedImage: Loaded image container.
        """
        pass

    def load_batch(self, sources: Sequence[Any]) -> List[LoadedImage]:
        """Load multiple images from given sources.

        Args:
            sources: Sequence of sources.

        Returns:
            List[LoadedImage]: List of loaded image containers.
        """
        loaded = []
        for src in sources:
            try:
                loaded.append(self.load(src))
            except Exception as e:
                raise RuntimeError(f"Failed to load image from {src}: {e}") from e
        return loaded


class FileImageLoader(BaseImageLoader):
    """Loads images from local file paths."""

    SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def load(self, source: Union[str, Path]) -> LoadedImage:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")

        # Use cv2.imdecode to handle potential non-ascii paths properly
        with open(path, "rb") as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Failed to decode image from path: {path}")

        return LoadedImage(
            image=image,
            source_id=str(path.resolve()),
            file_name=path.name,
            metadata={"file_size_bytes": path.stat().st_size},
        )

    def load_directory(
        self,
        dir_path: Union[str, Path],
        extensions: Optional[Sequence[str]] = None,
    ) -> List[LoadedImage]:
        """Load all supported images from a directory.

        Args:
            dir_path: Path to directory.
            extensions: Allowed file extensions. Default is SUPPORTED_EXTENSIONS.

        Returns:
            List[LoadedImage]: List of loaded images sorted by file name.
        """
        exts = tuple(extensions) if extensions else self.SUPPORTED_EXTENSIONS
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Directory not found: {path}")

        files = sorted(
            [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts],
            key=lambda p: p.name,
        )
        return self.load_batch(files)


class BytesImageLoader(BaseImageLoader):
    """Loads images from in-memory byte buffers."""

    def load(
        self,
        source: bytes,
        file_name: str = "image.jpg",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LoadedImage:
        if not source:
            raise ValueError("Input bytes are empty.")

        file_bytes = np.frombuffer(source, dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image from bytes.")

        meta = metadata.copy() if metadata else {}
        meta["file_size_bytes"] = len(source)

        return LoadedImage(
            image=image,
            source_id=f"bytes:{file_name}",
            file_name=file_name,
            metadata=meta,
        )
