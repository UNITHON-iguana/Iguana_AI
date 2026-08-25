#!/usr/bin/env python3
"""CLI script to detect and crop table regions from construction photos."""

import argparse
import sys
from pathlib import Path

# Add project root and ai/src to sys.path so it can be run standalone
current_dir = Path(__file__).resolve().parent
ai_root = current_dir.parent
if str(ai_root / "src") not in sys.path:
    sys.path.insert(0, str(ai_root / "src"))

from table_crop import (
    FileImageLoader,
    FileImageSaver,
    TableCropper,
    TableCropPipeline,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect and crop construction board tables from photos using OpenCV."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Path to an input image file or directory of images.",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        type=str,
        help="Path to the output directory where cropped tables will be saved.",
    )
    parser.add_argument(
        "-d", "--dilate",
        type=int,
        default=1,
        help="Number of morphological dilation iterations (default: 1).",
    )
    parser.add_argument(
        "-e", "--erode",
        type=int,
        default=0,
        help="Number of morphological erosion iterations (default: 0).",
    )
    parser.add_argument(
        "-p", "--padding",
        type=int,
        default=0,
        help="Padding in pixels around detected table boundary (default: 0).",
    )
    parser.add_argument(
        "-q", "--quality",
        type=int,
        default=95,
        help="JPEG quality for saved images (1-100, default: 95).",
    )
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save accompanying .json metadata for each cropped image.",
    )
    parser.add_argument(
        "--no-bottom-left-filter",
        action="store_true",
        help="Disable prioritizing bottom-left position for table board detection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    cropper = TableCropper(
        dilate_iterations=args.dilate,
        erode_iterations=args.erode,
        padding_px=args.padding,
        filter_bottom_left=not args.no_bottom_left_filter,
    )
    loader = FileImageLoader()
    saver = FileImageSaver(
        jpeg_quality=args.quality,
        save_metadata_json=args.save_metadata,
    )
    pipeline = TableCropPipeline(loader=loader, cropper=cropper, saver=saver)

    print("=" * 60)
    print("Field Note AI - Table Cropper")
    print(f"Input:       {input_path}")
    print(f"Output:      {output_path}")
    print(f"Dilation:    {args.dilate}")
    print(f"Padding:     {args.padding}px")
    print("=" * 60)

    if input_path.is_file():
        crop_res, save_res = pipeline.process_single(input_path, output_path)
        if save_res.success:
            print(f"✓ Processed 1 file: {crop_res.file_name} -> BBox {crop_res.bbox} (Confidence: {crop_res.confidence})")
        else:
            print(f"✗ Failed to save {crop_res.file_name}: {save_res.error_message}", file=sys.stderr)
            sys.exit(1)
    else:
        results = pipeline.process_directory(input_path, output_path)
        success_count = sum(1 for _, sr in results if sr.success)
        fallback_count = sum(1 for cr, _ in results if cr.is_fallback)

        print(f"\nCompleted processing {len(results)} images:")
        print(f"  - Successfully saved: {success_count} / {len(results)}")
        print(f"  - Fallback used:      {fallback_count}")

        for cr, sr in results:
            status = "✓" if sr.success else "✗"
            fb_note = " [Fallback]" if cr.is_fallback else ""
            print(f"  {status} {cr.file_name:<15} BBox: {str(cr.bbox):<20} Shape: {str(cr.cropped_image.shape):<18}{fb_note}")

    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
