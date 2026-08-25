"""Table cropping pipeline coordinating loader, cropper, and saver."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .cropper import CropResult, TableCropper
from .loader import BaseImageLoader, FileImageLoader, LoadedImage
from .saver import BaseImageSaver, FileImageSaver, SaveResult


class TableCropPipeline:
    """Pipeline coordinating image loading, table cropping, and image saving."""

    def __init__(
        self,
        loader: Optional[BaseImageLoader] = None,
        cropper: Optional[TableCropper] = None,
        saver: Optional[BaseImageSaver] = None,
    ):
        self.loader = loader or FileImageLoader()
        self.cropper = cropper or TableCropper()
        self.saver = saver or FileImageSaver()

    def process_single(
        self,
        source: Any,
        destination: Any,
        file_name: Optional[str] = None,
    ) -> Tuple[CropResult, SaveResult]:
        """Process a single image: load -> crop -> save.

        Args:
            source: Image source (file path, bytes, etc.).
            destination: Destination for saved crop (directory, buffer, etc.).
            file_name: Optional output file name.

        Returns:
            Tuple[CropResult, SaveResult]: Results from cropping and saving.
        """
        loaded_img = self.loader.load(source)
        crop_res = self.cropper.crop(loaded_img)
        save_res = self.saver.save(crop_res, destination, file_name=file_name)
        return crop_res, save_res

    def process_batch(
        self,
        sources: Sequence[Any],
        destination: Any,
    ) -> List[Tuple[CropResult, SaveResult]]:
        """Process multiple image sources."""
        results = []
        for src in sources:
            results.append(self.process_single(src, destination))
        return results

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extensions: Optional[Sequence[str]] = None,
    ) -> List[Tuple[CropResult, SaveResult]]:
        """Process all images in a directory.

        Args:
            input_dir: Directory containing original images.
            output_dir: Directory to save cropped table images.
            extensions: Allowed image extensions.

        Returns:
            List[Tuple[CropResult, SaveResult]]: Results for each image processed.
        """
        if not isinstance(self.loader, FileImageLoader):
            raise TypeError("Directory processing requires a FileImageLoader instance.")

        loaded_images = self.loader.load_directory(input_dir, extensions=extensions)
        results = []
        for img in loaded_images:
            crop_res = self.cropper.crop(img)
            save_res = self.saver.save(crop_res, output_dir)
            results.append((crop_res, save_res))

        return results
