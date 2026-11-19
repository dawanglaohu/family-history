"""Run the full pipeline (Steps 1-5) in one shot.

    python src/pipeline.py examples/scans -o data/manifest.json

Step 6 (verify.py) builds a synthetic sample and runs this end-to-end.
Each step is independently runnable too.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ingest import ingest_folder
from cluster import cluster_photos
from chronology import order_photos
from story import generate_stories
from tree import build_tree


def run(scans_dir: Path) -> dict:
    manifest = ingest_folder(scans_dir)
    cluster_photos(manifest)
    order_photos(manifest)
    generate_stories(manifest)
    build_tree(manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scans", type=Path, help="folder of scans/photos")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("data/manifest.json"),
        help="output manifest (default: data/manifest.json)",
    )
    args = parser.parse_args(argv)
    manifest = run(args.scans)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    chrono = manifest.get("chronology", {})
    print(
        f"pipeline: {manifest['count']} photos, "
        f"{len(manifest.get('scenes', {}))} scenes, "
        f"{chrono.get('anchors', 0)} anchors, "
        f"{chrono.get('inferred', 0)} inferred, "
        f"{len(manifest.get('tree', {}).get('nodes', []))} tree nodes -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
