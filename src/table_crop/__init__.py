"""Table cropping package for construction field note AI.

Provides modular image loading, OpenCV-based table board detection & cropping,
and flexible image saving implementations.
"""

from .cropper import CropResult, TableCropper
from .loader import BaseImageLoader, BytesImageLoader, FileImageLoader, LoadedImage
from .pipeline import TableCropPipeline
from .saver import BaseImageSaver, BytesImageSaver, FileImageSaver, SaveResult

__all__ = [
    "BaseImageLoader",
    "FileImageLoader",
    "BytesImageLoader",
    "LoadedImage",
    "TableCropper",
    "CropResult",
    "BaseImageSaver",
    "FileImageSaver",
    "BytesImageSaver",
    "SaveResult",
    "TableCropPipeline",
]
