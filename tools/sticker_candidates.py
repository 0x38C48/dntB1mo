#!/usr/bin/env python3
"""Find image files that are plausible custom sticker candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
EXCLUDED_DIRS = {"Data", "Emoji", "Portrait"}
THUMB_MARKERS = ("_thumb", "_thum", "thumb.")


def is_thumbnail(path: Path) -> bool:
    name = path.name.lower()
    return any(marker in name for marker in THUMB_MARKERS)


def iter_images(records_dir: Path) -> list[Path]:
    images: list[Path] = []
    for path in records_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_parts = set(path.relative_to(records_dir).parts[:-1])
        if relative_parts.intersection(EXCLUDED_DIRS):
            continue
        images.append(path)
    return sorted(images, key=lambda p: str(p).lower())


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_image(width: int, height: int, size_bytes: int, thumb: bool) -> tuple[float, str]:
    short = min(width, height)
    long = max(width, height)
    aspect = long / short if short else math.inf

    square_score = clamp01(1.0 - (aspect - 1.0) / 0.35)
    if 160 <= long <= 900 and 120 <= short <= 900:
        dimension_score = 1.0
    elif long <= 1200 and short >= 80:
        dimension_score = 0.72
    elif long <= 1600 and short >= 80:
        dimension_score = 0.45
    else:
        dimension_score = 0.15

    size_mb = size_bytes / (1024 * 1024)
    if size_mb <= 0.75:
        file_score = 1.0
    elif size_mb <= 1.5:
        file_score = 0.78
    elif size_mb <= 3.0:
        file_score = 0.45
    else:
        file_score = 0.12

    penalty = 0.35 if thumb else 0.0
    score = clamp01(0.62 * square_score + 0.24 * dimension_score + 0.14 * file_score - penalty)

    if not thumb and aspect <= 1.12 and 120 <= short and long <= 1000 and size_mb <= 1.5:
        bucket = "likely"
    elif not thumb and aspect <= 1.25 and 80 <= short and long <= 1400 and size_mb <= 3.0:
        bucket = "review"
    else:
        bucket = "unlikely"
    return round(score, 4), bucket


def inspect_image(path: Path, records_dir: Path) -> dict[str, Any] | None:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            mode = image.mode
    except Exception as exc:
        return {
            "relative_path": str(path.relative_to(records_dir)),
            "absolute_path": str(path),
            "error": str(exc),
        }

    size_bytes = path.stat().st_size
    thumb = is_thumbnail(path)
    score, bucket = score_image(width, height, size_bytes, thumb)
    short = min(width, height)
    long = max(width, height)
    return {
        "relative_path": str(path.relative_to(records_dir)),
        "absolute_path": str(path),
        "filename": path.name,
        "extension": path.suffix.lower(),
        "width": width,
        "height": height,
        "aspect_ratio_long_short": round(long / short, 4) if short else None,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 4),
        "is_thumbnail": thumb,
        "mode": mode,
        "score": score,
        "bucket": bucket,
    }


def copy_candidates(rows: list[dict[str, Any]], out_dir: Path, limit: int) -> None:
    likely_dir = out_dir / "likely_stickers"
    review_dir = out_dir / "review_stickers"
    likely_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    selected = [row for row in rows if row.get("bucket") == "likely"]
    selected.sort(key=lambda r: (-r["score"], r["size_bytes"], r["filename"]))
    for index, row in enumerate(selected[:limit], start=1):
        src = Path(row["absolute_path"])
        dst = likely_dir / f"{index:04d}_{src.name}"
        if not dst.exists():
            shutil.copy2(src, dst)

    review = [row for row in rows if row.get("bucket") == "review"]
    review.sort(key=lambda r: (-r["score"], r["size_bytes"], r["filename"]))
    for index, row in enumerate(review[: min(limit, 200)], start=1):
        src = Path(row["absolute_path"])
        dst = review_dir / f"{index:04d}_{src.name}"
        if not dst.exists():
            shutil.copy2(src, dst)


def write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (-r.get("score", -1), r.get("bucket", "z"), r.get("relative_path", "")))

    with (out_dir / "sticker_candidates.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = [
        "bucket",
        "score",
        "relative_path",
        "width",
        "height",
        "aspect_ratio_long_short",
        "size_mb",
        "is_thumbnail",
        "extension",
        "absolute_path",
    ]
    with (out_dir / "sticker_candidates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--copy-limit", type=int, default=500)
    args = parser.parse_args()

    records_dir = args.records_dir.resolve()
    out_dir = args.out_dir.resolve()
    rows = [row for row in (inspect_image(path, records_dir) for path in iter_images(records_dir)) if row]
    write_outputs(rows, out_dir)
    copy_candidates(rows, out_dir, args.copy_limit)

    counts: dict[str, int] = {}
    for row in rows:
        bucket = row.get("bucket", "error")
        counts[bucket] = counts.get(bucket, 0) + 1
    print(json.dumps({"out_dir": str(out_dir), "image_count": len(rows), "bucket_counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
