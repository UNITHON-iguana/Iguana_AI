#!/usr/bin/env python3
"""CLI script to run Gemini OCR on construction board table images.

Usage:
    python ai/scripts/run_ocr.py -i ai/sample_data/DG1/cropped_table -o ai/sample_data/DG1/ocr_results
    python ai/scripts/run_ocr.py -i path/to/board_crop.jpg
"""

import argparse
import os
from pathlib import Path
import sys
from typing import Optional

# Add project src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ocr import (
    BoardTableItem,
    FileImageLoader,
    GeminiOCREngine,
    JsonFileOCRResultSaver,
    OCRPipeline,
    OCRResult,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gemini OCR on construction board table images to extract raw unedited text."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Path to an input image file or a directory containing cropped table images.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Directory to save the JSON output files. Defaults to '<input_dir>/ocr_results'.",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Gemini model name to use (defaults to GEMINI_MODEL in .env or 'gemini-3.5-flash-lite').",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively scan subdirectories for images.",
    )
    return parser.parse_args()


def format_field(val: Optional[str]) -> str:
    if val is None:
        return "<null>"
    return f"'{val}'"


def progress_callback(idx: int, total: int, result: OCRResult) -> None:
    status_icon = "✅" if result.success else "❌"
    print(f"[{idx}/{total}] {status_icon} {result.file_name} ({result.execution_time_sec:.2f}s) [Model: {result.model_used}]")
    if result.success and result.data:
        data = result.data
        print(f"    ├─ 공사명: {format_field(data.공사명)}")
        print(f"    ├─ 공종  : {format_field(data.공종)}")
        print(f"    ├─ 위치  : {format_field(data.위치)}")
        print(f"    ├─ 내용  : {format_field(data.내용)}")
        print(f"    └─ 일자  : {format_field(data.일자)}")
    elif not result.success:
        print(f"    └─ [ERROR] {result.error_message}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output)
    else:
        if input_path.is_dir():
            output_dir = input_path / "ocr_results"
        else:
            output_dir = input_path.parent / "ocr_results"

    try:
        engine = GeminiOCREngine(model=args.model)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        print("Please configure GEMINI_API_KEY in ai/.env or as an environment variable.", file=sys.stderr)
        sys.exit(1)

    pipeline = OCRPipeline(
        loader=FileImageLoader(),
        engine=engine,
        saver=JsonFileOCRResultSaver(),
    )

    print("=" * 60)
    print(" 현장노트 AI - 보드판 무가공 OCR 실행")
    print(f" - 입력 소스: {input_path}")
    print(f" - 저장 위치: {output_dir}")
    print(f" - 사용 모델: {engine.model}")
    print("=" * 60)

    if input_path.is_file():
        result = pipeline.process_image(input_path, output_dir=output_dir)
        progress_callback(1, 1, result)
        results = [result]
    else:
        results = pipeline.process_directory(
            input_dir=input_path,
            output_dir=output_dir,
            recursive=args.recursive,
            on_progress=progress_callback,
        )

    success_count = sum(1 for r in results if r.success)
    print("=" * 60)
    print(f" 완료: 총 {len(results)}건 중 {success_count}건 성공 (결과 저장: {output_dir})")
    print("=" * 60)


if __name__ == "__main__":
    main()
