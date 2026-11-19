"""Run the ingest -> cluster -> chronology pipeline in one shot.

    python src/pipeline.py examples/scans -o data/manifest.json

Each step is independently runnable too (ingest.py, cluster.py, chronology.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ingest import ingest_folder
from cluster import cluster_photos
from chronology import order_photos


def run(scans_dir: Path) -> dict:
    manifest = ingest_folder(scans_dir)
    cluster_photos(manifest)
    order_photos(manifest)
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
        f"{chrono.get('inferred', 0)} inferred -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
