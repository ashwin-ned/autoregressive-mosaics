#!/usr/bin/env python3
"""Create a collage banner from all generated images in this folder."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

try:
    from PIL import Image
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: Pillow. Install it with `pip install Pillow` "
        "or your system package manager (e.g. `python3-pillow`)."
    ) from exc


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_size(size: str) -> tuple[int, int]:
    try:
        width_str, height_str = size.lower().split("x")
        width, height = int(width_str), int(height_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Size must look like WIDTHxHEIGHT, e.g. 1920x1080."
        ) from exc

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Width and height must be positive integers.")
    return width, height


def pick_grid(num_images: int, target_width: int, target_height: int) -> tuple[int, int]:
    if num_images <= 0:
        return 1, 1

    target_aspect = target_width / target_height
    cols = max(1, math.ceil(math.sqrt(num_images * target_aspect)))
    rows = math.ceil(num_images / cols)

    # Reduce empty cells while keeping all images in the grid.
    while cols > 1 and (cols - 1) * rows >= num_images:
        cols -= 1

    return cols, rows


def resize_and_crop(img: Image.Image, width: int, height: int) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    scale = max(width / img.width, height / img.height)
    resized = img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
        resample=resampling,
    )

    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    right = left + width
    bottom = top + height
    return resized.crop((left, top, right, bottom))


def collect_images(input_dir: Path, output_name: str) -> list[Path]:
    output_base = Path(output_name).stem
    images = [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and p.stem != output_base
    ]
    return images


def make_banner(input_dir: Path, output_name: str, size: tuple[int, int]) -> Path:
    target_width, target_height = size
    image_paths = collect_images(input_dir, output_name)
    if not image_paths:
        raise RuntimeError(f"No input images found in {input_dir}")

    cols, rows = pick_grid(len(image_paths), target_width, target_height)
    cell_width = max(1, target_width // cols)
    cell_height = max(1, target_height // rows)
    x_offset = (target_width - (cell_width * cols)) // 2
    y_offset = (target_height - (cell_height * rows)) // 2

    canvas = Image.new("RGB", (target_width, target_height), color=(0, 0, 0))

    for idx, img_path in enumerate(image_paths):
        row = idx // cols
        col = idx % cols
        if row >= rows:
            break

        with Image.open(img_path) as img:
            tile = resize_and_crop(img.convert("RGB"), cell_width, cell_height)
        x = x_offset + (col * cell_width)
        y = y_offset + (row * cell_height)
        canvas.paste(tile, (x, y))

    output_path = input_dir / output_name
    if not output_path.suffix:
        output_path = output_path.with_suffix(".png")
    canvas.save(output_path, optimize=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a collage banner from all images in a folder."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Folder containing source images (default: this script's folder).",
    )
    parser.add_argument(
        "--size",
        type=parse_size,
        default=parse_size("1920x1080"),
        help="Output size as WIDTHxHEIGHT (e.g. 1920x1080 or 1600x1068).",
    )
    parser.add_argument(
        "--output",
        default="banner_size.png",
        help="Output filename (default: banner_size.png).",
    )
    args = parser.parse_args()

    output_path = make_banner(args.input_dir, args.output, args.size)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
