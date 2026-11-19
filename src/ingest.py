"""Step 1 ingest for the family-history prototype.

Walk a folder of scans/photos, extract EXIF DateTimeOriginal where present,
hash each file, and emit data/manifest.json. Photos without a usable EXIF date
are flagged so the later chronology step can place them by inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ExifTags
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Pillow is required: install it with 'pip install Pillow'"
    ) from exc

IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic"
}
EXIF_DATE_TAG = 36867  # DateTimeOriginal


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_exif_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%Y:%m:%d %H:%M:%S").isoformat()
    except (TypeError, ValueError):
        return None


def extract_exif_date(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            exif = image.getexif()
    except Exception:
        return None
    if not exif:
        return None
    raw = exif.get(EXIF_DATE_TAG)
    if raw is None:
        for tag_id, tag_name in ExifTags.TAGS.items():
            if tag_name == "DateTimeOriginal":
                raw = exif.get(tag_id)
                break
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    return parse_exif_date(raw if isinstance(raw, str) else None)


def image_dimensions(path: Path) -> list[int] | None:
    try:
        with Image.open(path) as image:
            return list(image.size)
    except Exception:
        return None


def record_for(path: Path, root: Path) -> dict[str, Any]:
    exif_date = extract_exif_date(path)
    return {
        "id": "sha256:" + sha256_file(path),
        "path": path.relative_to(root).as_posix(),
        "hash": sha256_file(path),
        "exif_date": exif_date,
        "dimensions": image_dimensions(path),
        "needs_inference": exif_date is None,
        # Populated by later pipeline steps; ingest leaves these empty.
        "scene_id": None,
        "person_ids": [],
        "inferred_order": None,
        "confidence": None,
    }


def ingest_folder(scans_dir: Path) -> dict[str, Any]:
    scans_dir = scans_dir.resolve()
    if not scans_dir.is_dir():
        raise SystemExit(f"not a directory: {scans_dir}")
    files = sorted(
        p for p in scans_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    records = [record_for(path, scans_dir) for path in files]
    return {
        "version": 1,
        "source": str(scans_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(records),
        "with_exif": sum(1 for r in records if r["exif_date"]),
        "needs_inference": sum(1 for r in records if r["needs_inference"]),
        "photos": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scans", type=Path, help="folder of scans/photos")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("data/manifest.json"),
        help="where to write manifest.json (default: data/manifest.json)",
    )
    args = parser.parse_args(argv)
    manifest = ingest_folder(args.scans)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"ingested {manifest['count']} photos "
        f"({manifest['with_exif']} with EXIF, "
        f"{manifest['needs_inference']} need inference) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
