#!/usr/bin/env python3
"""Create a quick visual contact sheet for sticker candidate folders."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def image_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"}],
        key=lambda p: p.name.lower(),
    )


def fit_image(path: Path, box: int) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((box, box))
        canvas = Image.new("RGB", (box, box), "white")
        left = (box - image.width) // 2
        top = (box - image.height) // 2
        canvas.paste(image, (left, top))
        return canvas


def draw_section(draw: ImageDraw.ImageDraw, x: int, y: int, text: str) -> None:
    draw.text((x, y), text, fill=(20, 20, 20))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cell", type=int, default=180)
    parser.add_argument("--cols", type=int, default=5)
    args = parser.parse_args()

    folders = [
        ("likely", args.candidate_dir / "likely_stickers"),
        ("review", args.candidate_dir / "review_stickers"),
    ]
    rows: list[tuple[str, Path]] = []
    for label, folder in folders:
        for path in image_files(folder):
            rows.append((label, path))

    if not rows:
        raise SystemExit("No candidate images found.")

    label_h = 38
    header_h = 34
    cell = args.cell
    cols = args.cols
    tile_h = cell + label_h
    grid_rows = math.ceil(len(rows) / cols)
    width = cols * cell
    height = header_h + grid_rows * tile_h
    sheet = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw_section(draw, 8, 10, f"Sticker candidates: {len(rows)} images")

    for idx, (label, path) in enumerate(rows):
        col = idx % cols
        row = idx // cols
        x = col * cell
        y = header_h + row * tile_h
        try:
            thumb = fit_image(path, cell - 12)
            sheet.paste(thumb, (x + 6, y + 6))
        except Exception:
            draw.rectangle((x + 6, y + 6, x + cell - 6, y + cell - 6), outline=(210, 60, 60), width=2)
        text = f"{label} {path.name[:24]}"
        draw.text((x + 6, y + cell + 4), text, fill=(30, 30, 30), font=font)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=92)
    print(str(args.out))


if __name__ == "__main__":
    main()
