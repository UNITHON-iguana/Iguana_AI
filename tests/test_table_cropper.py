"""Unit and integration tests for TableCropper and modular loader/saver."""

from pathlib import Path
import sys
import numpy as np
import pytest
import cv2

# Add ai/src to sys.path
test_dir = Path(__file__).resolve().parent
ai_root = test_dir.parent
if str(ai_root / "src") not in sys.path:
    sys.path.insert(0, str(ai_root / "src"))

from table_crop import (
    BaseImageLoader,
    BaseImageSaver,
    BytesImageLoader,
    BytesImageSaver,
    FileImageLoader,
    FileImageSaver,
    LoadedImage,
    TableCropper,
    TableCropPipeline,
)


@pytest.fixture
def sample_data_paths():
    sample_dir = ai_root / "sample_data" / "test_samples_1"
    if not sample_dir.exists():
        sample_dir = ai_root / "sample_data" / "test_samples"
    photos_dir = sample_dir / "photos"
    tables_dir = sample_dir / "tables"
    return photos_dir, tables_dir


def test_file_image_loader(sample_data_paths):
    photos_dir, _ = sample_data_paths
    loader = FileImageLoader()
    first_image_path = photos_dir / "378.jpg"

    loaded = loader.load(first_image_path)
    assert isinstance(loaded, LoadedImage)
    assert loaded.file_name == "378.jpg"
    assert loaded.width == 1440
    assert loaded.height == 1080
    assert loaded.channels == 3
    assert loaded.image.shape == (1080, 1440, 3)


def test_bytes_image_loader_and_saver():
    # Create a synthetic white rectangle on black background
    synthetic_img = np.zeros((300, 400, 3), dtype=np.uint8)
    synthetic_img[180:280, 20:180] = [255, 255, 255]

    _, encoded = cv2.imencode(".jpg", synthetic_img)
    img_bytes = encoded.tobytes()

    loader = BytesImageLoader()
    loaded = loader.load(img_bytes, file_name="synthetic.jpg")
    assert loaded.width == 400
    assert loaded.height == 300

    cropper = TableCropper(filter_bottom_left=True)
    crop_res = cropper.crop(loaded)

    assert crop_res.width > 0
    assert crop_res.height > 0
    assert not crop_res.is_fallback

    saver = BytesImageSaver()
    save_res = saver.save(crop_res)
    assert save_res.success
    assert save_res.data is not None
    assert len(save_res.data) > 0


def test_cropper_all_32_samples(sample_data_paths):
    photos_dir, tables_dir = sample_data_paths
    assert photos_dir.is_dir(), f"Photos directory not found: {photos_dir}"
    assert tables_dir.is_dir(), f"Tables directory not found: {tables_dir}"

    loader = FileImageLoader()
    cropper = TableCropper(dilate_iterations=1, filter_bottom_left=True)

    photo_files = sorted([p for p in photos_dir.iterdir() if p.suffix.lower() == ".jpg"])
    assert len(photo_files) == 32, f"Expected 32 sample images, found {len(photo_files)}"

    for photo_path in photo_files:
        target_path = tables_dir / photo_path.name
        assert target_path.is_file(), f"Ground truth table not found: {target_path}"

        target_img = cv2.imread(str(target_path))
        loaded = loader.load(photo_path)
        crop_res = cropper.crop(loaded)

        # Verify crop shape matches the ground-truth table
        assert crop_res.cropped_image.shape == target_img.shape, (
            f"Shape mismatch for {photo_path.name}: "
            f"Got {crop_res.cropped_image.shape}, Expected {target_img.shape}"
        )
        assert crop_res.confidence >= 0.8
        assert not crop_res.is_fallback


def test_table_cropper_with_padding(sample_data_paths):
    photos_dir, _ = sample_data_paths
    loader = FileImageLoader()
    cropper_no_pad = TableCropper(padding_px=0)
    cropper_pad = TableCropper(padding_px=10)

    loaded = loader.load(photos_dir / "378.jpg")
    res_no_pad = cropper_no_pad.crop(loaded)
    res_pad = cropper_pad.crop(loaded)

    # Padding should make the width and height larger
    assert res_pad.width >= res_no_pad.width
    assert res_pad.height >= res_no_pad.height
    # Ensure boundary clamping
    assert res_pad.x >= 0
    assert res_pad.y >= 0
    assert res_pad.x + res_pad.width <= loaded.width
    assert res_pad.y + res_pad.height <= loaded.height


def test_pipeline_directory_processing(tmp_path, sample_data_paths):
    photos_dir, _ = sample_data_paths
    output_dir = tmp_path / "cropped_output"

    pipeline = TableCropPipeline(
        loader=FileImageLoader(),
        cropper=TableCropper(),
        saver=FileImageSaver(save_metadata_json=True),
    )

    results = pipeline.process_directory(photos_dir, output_dir)
    assert len(results) == 32

    for crop_res, save_res in results:
        assert save_res.success
        saved_file = Path(save_res.destination)
        assert saved_file.is_file()
        assert saved_file.stat().st_size > 0

        # Check metadata json was created
        meta_file = saved_file.with_suffix(".json")
        assert meta_file.is_file()
