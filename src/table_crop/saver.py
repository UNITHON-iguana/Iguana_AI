"""Image saver module for table cropping.

Provides abstract and concrete image savers to support saving cropped table images
to local directories, memory buffers, or cloud object storages.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np

from .cropper import CropResult


@dataclass
class SaveResult:
    """Result of saving a cropped image."""
    destination: str
    file_name: str
    success: bool
    bytes_written: int = 0
    data: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class BaseImageSaver(ABC):
    """Abstract base class for saving cropped images."""

    @abstractmethod
    def save(
        self,
        crop_result: CropResult,
        destination: Any,
        file_name: Optional[str] = None,
        **kwargs: Any,
    ) -> SaveResult:
        """Save a single CropResult to the given destination.

        Args:
            crop_result: The detection and crop result.
            destination: Destination directory, buffer, or storage path.
            file_name: Optional custom output file name.

        Returns:
            SaveResult: Details of the save operation.
        """
        pass

    def save_batch(
        self,
        crop_results: Sequence[CropResult],
        destination: Any,
        **kwargs: Any,
    ) -> List[SaveResult]:
        """Save multiple CropResults to the given destination.

        Args:
            crop_results: Sequence of CropResults.
            destination: Destination.

        Returns:
            List[SaveResult]: List of SaveResults.
        """
        return [self.save(cr, destination, **kwargs) for cr in crop_results]


class FileImageSaver(BaseImageSaver):
    """Saves cropped images to a local filesystem directory."""

    def __init__(
        self,
        jpeg_quality: int = 95,
        save_metadata_json: bool = False,
    ):
        """Initialize FileImageSaver.

        Args:
            jpeg_quality: JPEG compression quality (1-100). Default is 95.
            save_metadata_json: If True, writes an accompanying .json metadata file.
        """
        self.jpeg_quality = jpeg_quality
        self.save_metadata_json = save_metadata_json

    def save(
        self,
        crop_result: CropResult,
        destination: Union[str, Path],
        file_name: Optional[str] = None,
        **kwargs: Any,
    ) -> SaveResult:
        dest_dir = Path(destination)
        dest_dir.mkdir(parents=True, exist_ok=True)

        target_name = file_name or crop_result.file_name
        output_path = dest_dir / target_name

        try:
            ext = output_path.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            elif ext == ".png":
                encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
            else:
                encode_params = []

            # Use cv2.imencode + file.write to handle Unicode/special paths properly
            success, encoded_img = cv2.imencode(ext or ".jpg", crop_result.cropped_image, encode_params)
            if not success:
                raise ValueError(f"Failed to encode image to {ext}")

            img_bytes = encoded_img.tobytes()
            with open(output_path, "wb") as f:
                f.write(img_bytes)

            meta: Dict[str, Any] = {
                "source_id": crop_result.source_id,
                "file_name": target_name,
                "bbox": crop_result.bbox,
                "original_shape": crop_result.original_shape,
                "cropped_shape": crop_result.cropped_image.shape,
                "confidence": crop_result.confidence,
                "is_fallback": crop_result.is_fallback,
                "aspect_ratio": crop_result.aspect_ratio,
                "area_ratio": crop_result.area_ratio,
            }

            if self.save_metadata_json:
                json_path = output_path.with_suffix(".json")
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(meta, jf, ensure_ascii=False, indent=2)

            return SaveResult(
                destination=str(output_path.resolve()),
                file_name=target_name,
                success=True,
                bytes_written=len(img_bytes),
                metadata=meta,
            )
        except Exception as e:
            return SaveResult(
                destination=str(output_path),
                file_name=target_name,
                success=False,
                error_message=str(e),
            )


class BytesImageSaver(BaseImageSaver):
    """Encodes cropped images into in-memory byte buffers."""

    def __init__(self, image_format: str = ".jpg", jpeg_quality: int = 95):
        self.image_format = image_format if image_format.startswith(".") else f".{image_format}"
        self.jpeg_quality = jpeg_quality

    def save(
        self,
        crop_result: CropResult,
        destination: Any = None,
        file_name: Optional[str] = None,
        **kwargs: Any,
    ) -> SaveResult:
        target_name = file_name or crop_result.file_name
        try:
            params = []
            if self.image_format in (".jpg", ".jpeg"):
                params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]

            success, encoded_img = cv2.imencode(self.image_format, crop_result.cropped_image, params)
            if not success:
                raise ValueError(f"Failed to encode image to {self.image_format}")

            img_bytes = encoded_img.tobytes()
            meta = {
                "source_id": crop_result.source_id,
                "file_name": target_name,
                "bbox": crop_result.bbox,
                "confidence": crop_result.confidence,
                "is_fallback": crop_result.is_fallback,
            }

            return SaveResult(
                destination=f"memory:{target_name}",
                file_name=target_name,
                success=True,
                bytes_written=len(img_bytes),
                data=img_bytes,
                metadata=meta,
            )
        except Exception as e:
            return SaveResult(
                destination=f"memory:{target_name}",
                file_name=target_name,
                success=False,
                error_message=str(e),
            )
