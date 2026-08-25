"""Table cropper module using OpenCV for boundary and white rectangle detection.

Based on the validated approach in data_preparation.ipynb, detecting construction
board tables using white color masking, Canny edge detection, morphological dilation,
and spatial/shape filtering without deep learning models.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np

from .loader import LoadedImage


@dataclass
class CropResult:
    """Result of table detection and cropping."""
    cropped_image: np.ndarray
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    original_shape: Tuple[int, int, int]  # (height, width, channels)
    confidence: float
    is_fallback: bool
    source_id: str
    file_name: str
    metadata: Dict[str, Any]

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)

    @property
    def area_ratio(self) -> float:
        orig_area = self.original_shape[0] * self.original_shape[1]
        crop_area = self.width * self.height
        return crop_area / max(orig_area, 1)


class TableCropper:
    """Detects and crops table/board regions from construction site photos."""

    def __init__(
        self,
        lower_white: Tuple[int, int, int] = (253, 253, 253),
        upper_white: Tuple[int, int, int] = (255, 255, 255),
        canny_thresh1: int = 20,
        canny_thresh2: int = 70,
        dilate_iterations: int = 1,
        erode_iterations: int = 0,
        kernel_size: Tuple[int, int] = (5, 5),
        filter_bottom_left: bool = True,
        padding_px: int = 0,
        min_area_ratio: float = 0.005,
        max_area_ratio: float = 0.85,
    ):
        """Initialize table cropper with detection parameters.

        Args:
            lower_white: Lower BGR bound for white color mask.
            upper_white: Upper BGR bound for white color mask.
            canny_thresh1: First threshold for Canny edge detector.
            canny_thresh2: Second threshold for Canny edge detector.
            dilate_iterations: Number of dilation iterations to connect table borders.
            erode_iterations: Number of erosion iterations.
            kernel_size: Morphological kernel size.
            filter_bottom_left: Prioritize bottom-left region typical for construction boards.
            padding_px: Extra padding pixels around the detected bounding box.
            min_area_ratio: Minimum area ratio relative to full image size.
            max_area_ratio: Maximum area ratio relative to full image size.
        """
        self.lower_white = np.array(lower_white, dtype=np.uint8)
        self.upper_white = np.array(upper_white, dtype=np.uint8)
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2
        self.dilate_iterations = dilate_iterations
        self.erode_iterations = erode_iterations
        self.kernel = np.ones(kernel_size, dtype=np.uint8)
        self.filter_bottom_left = filter_bottom_left
        self.padding_px = padding_px
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def mask_white(self, image: np.ndarray) -> np.ndarray:
        """Mask out all non-white pixels as black."""
        mask = cv2.inRange(image, self.lower_white, self.upper_white)
        mask_inverse = cv2.bitwise_not(mask)
        masked_image = np.copy(image)
        masked_image[mask_inverse == 255] = [0, 0, 0]
        return masked_image

    def is_contour_at_bottom_left(self, image_shape: Tuple[int, int, int], contour: np.ndarray) -> bool:
        """Check if contour center or bounding box resides in the bottom-left region."""
        x, y, w, h = cv2.boundingRect(contour)
        img_h, img_w = image_shape[:2]

        center_x = x + w / 2.0
        center_y = y + h / 2.0

        # Center in the left half and bottom half of the image
        # Allow slight flexibility (center_x <= img_w * 0.55 and center_y >= img_h * 0.45)
        return center_x <= (img_w * 0.55) and center_y >= (img_h * 0.45)

    def is_valid_shape(self, image_shape: Tuple[int, int, int], contour: np.ndarray) -> bool:
        """Check if contour area is within reasonable bounds for a table board."""
        x, y, w, h = cv2.boundingRect(contour)
        img_h, img_w = image_shape[:2]
        total_area = img_h * img_w
        contour_area = w * h

        area_ratio = contour_area / max(total_area, 1)
        if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
            return False

        # Table boards are generally landscape (aspect ratio w/h >= 0.8)
        aspect_ratio = w / max(h, 1)
        if aspect_ratio < 0.6 or aspect_ratio > 6.0:
            return False

        return True

    def find_contours(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply white masking, Canny edge detection, and morphological ops to find contours."""
        masked = self.mask_white(image)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, self.canny_thresh1, self.canny_thresh2)

        processed = edged
        if self.dilate_iterations > 0:
            processed = cv2.dilate(processed, self.kernel, iterations=self.dilate_iterations)
        if self.erode_iterations > 0:
            processed = cv2.erode(processed, self.kernel, iterations=self.erode_iterations)

        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return list(contours)

    def crop(self, loaded_image: LoadedImage) -> CropResult:
        """Crop table region from the given loaded image.

        Args:
            loaded_image: LoadedImage instance containing image array and metadata.

        Returns:
            CropResult: Detection and cropping result.
        """
        image = loaded_image.image
        img_h, img_w, channels = image.shape
        contours = self.find_contours(image)

        is_fallback = False
        confidence = 0.9

        if not contours:
            # Fallback to bottom-left quadrant if no white contour detected
            is_fallback = True
            confidence = 0.2
            w = int(img_w * 0.35)
            h = int(img_h * 0.25)
            x = 0
            y = img_h - h
            selected_bbox = (x, y, w, h)
        else:
            # Filter contours by bottom-left position and valid shape
            candidates = contours
            if self.filter_bottom_left:
                pos_candidates = [
                    c for c in contours
                    if self.is_contour_at_bottom_left(image.shape, c) and self.is_valid_shape(image.shape, c)
                ]
                if not pos_candidates:
                    # Fallback to any bottom-left contour
                    pos_candidates = [c for c in contours if self.is_contour_at_bottom_left(image.shape, c)]

                if pos_candidates:
                    candidates = pos_candidates
                else:
                    # Fallback to valid shape anywhere in image
                    shape_candidates = [c for c in contours if self.is_valid_shape(image.shape, c)]
                    if shape_candidates:
                        candidates = shape_candidates
                        is_fallback = True
                        confidence = 0.6
                    else:
                        is_fallback = True
                        confidence = 0.4

            # Select largest contour among candidates
            best_contour = max(candidates, key=cv2.contourArea)
            selected_bbox = cv2.boundingRect(best_contour)

        # Apply padding if configured
        x, y, w, h = selected_bbox
        if self.padding_px > 0:
            x = max(0, x - self.padding_px)
            y = max(0, y - self.padding_px)
            w = min(img_w - x, w + 2 * self.padding_px)
            h = min(img_h - y, h + 2 * self.padding_px)
            selected_bbox = (x, y, w, h)

        # Extract cropped table region
        cropped = image[y : y + h, x : x + w]

        # Prevent empty crop
        if cropped.size == 0 or w == 0 or h == 0:
            is_fallback = True
            confidence = 0.1
            x, y, w, h = 0, int(img_h * 0.75), int(img_w * 0.35), int(img_h * 0.25)
            cropped = image[y : y + h, x : x + w]
            selected_bbox = (x, y, w, h)

        return CropResult(
            cropped_image=cropped,
            bbox=selected_bbox,
            original_shape=(img_h, img_w, channels),
            confidence=confidence,
            is_fallback=is_fallback,
            source_id=loaded_image.source_id,
            file_name=loaded_image.file_name,
            metadata=loaded_image.metadata.copy(),
        )
