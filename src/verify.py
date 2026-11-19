"""Step 6 verification: generate a synthetic sample and run the full pipeline.

    python src/verify.py --out verification/

Creates 30 synthetic photos across a few scenes/years (some with EXIF dates,
most without), runs ingest -> cluster -> chronology -> story -> tree, writes
manifest.json and a self-contained SVG preview of the tree, then prints an
acceptance report. This is the end-to-end acceptance test from the roadmap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from pipeline import run as run_pipeline


def make_sample(scans_dir: Path, count: int = 30) -> Path:
    scans_dir.mkdir(parents=True, exist_ok=True)
    # Deliberately distinct colour signatures so the perceptual embedding
    # separates them into distinct scenes (CLIP would do this semantically).
    buckets = [
        ("1998/xmas", (230, 60, 50), True, 12),     # red
        ("2003/summer", (40, 190, 70), False, 7),   # green
        ("2011/reunion", (50, 70, 220), False, None),  # blue
    ]
    per = max(1, count // len(buckets))
    made = 0
    for folder, colour, dated, month in buckets:
        target = scans_dir / folder
        target.mkdir(parents=True, exist_ok=True)
        for i in range(per):
            if made >= count:
                break
            image = Image.new("RGB", (64, 48), colour)
            draw = ImageDraw.Draw(image)
            draw.rectangle([6, 30, 58, 47], fill=tuple(max(0, c - 60) for c in colour))
            draw.ellipse([24, 6, 40, 22], fill=(220, 90, 70))
            path = target / f"photo_{i:02d}.jpg"
            if dated:
                exif = image.getexif()
                exif[36867] = f"1998:12:25 10:{i:02d}:00"
                image.save(path, exif=exif, quality=85)
            else:
                image.save(path, quality=85)
            made += 1
    return scans_dir


def render_svg(tree: dict, path: Path) -> None:
    nodes = tree.get("nodes", [])
    edges = tree.get("edges", [])
    width = max(800, 160 * (len(nodes) + 1))
    height = 420
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#111827"/>',
        '<text x="20" y="34" fill="#e5e7eb" font-family="sans-serif" font-size="18">family-history — interactive tree preview</text>',
    ]
    by_id = {n["id"]: n for n in nodes}
    for edge in edges:
        src, dst = by_id.get(edge["source"]), by_id.get(edge["target"])
        if not src or not dst:
            continue
        x1, y1 = src["position"]["x"] + 60, src["position"]["y"] + 30
        x2, y2 = dst["position"]["x"] + 60, dst["position"]["y"] + 30
        colour = "#22d3ee" if edge.get("type") == "temporal" else "#f472b6"
        lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" stroke-width="2"/>'
        )
    for node in nodes:
        x, y = node["position"]["x"], node["position"]["y"]
        data = node["data"]
        title = str(data.get("title", node["id"])).replace("&", "&amp;")[:22]
        conf = data.get("confidence", 0)
        lines.append(
            f'<g><rect x="{x}" y="{y}" width="120" height="60" rx="8" fill="#1f2937" stroke="#4b5563"/>'
            f'<text x="{x+8}" y="{y+22}" fill="#f9fafb" font-family="sans-serif" font-size="12">{title}</text>'
            f'<text x="{x+8}" y="{y+42}" fill="#9ca3af" font-family="sans-serif" font-size="10">conf {conf}</text></g>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def verify(out_dir: Path, count: int = 30) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    scans = make_sample(out_dir / "scans", count)
    manifest = run_pipeline(scans)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    render_svg(manifest.get("tree", {"nodes": [], "edges": []}), out_dir / "tree.svg")
    (out_dir / "README.md").write_text(
        "# Verification run\n\n"
        f"- photos: {manifest['count']}\n"
        f"- scenes: {len(manifest.get('scenes', {}))}\n"
        f"- chronology: {manifest.get('chronology')}\n"
        f"- stories: {len(manifest.get('stories', {}))}\n"
        f"- pin_request: {'yes' if manifest.get('pin_request') else 'no'}\n"
        f"- tree nodes/edges: {len(manifest.get('tree', {}).get('nodes', []))}/"
        f"{len(manifest.get('tree', {}).get('edges', []))}\n\n"
        "Open `tree.svg` for the interactive-tree preview.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("verification"))
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    manifest = verify(args.out, args.count)
    chrono = manifest.get("chronology", {})
    tree = manifest.get("tree", {})
    assert manifest["count"] == args.count, "photo count mismatch"
    assert all(p["inferred_order"] is not None for p in manifest["photos"]), "unorderd photo"
    assert tree.get("nodes"), "tree has no nodes"
    print(
        f"VERIFY OK: {manifest['count']} photos, "
        f"{len(manifest.get('scenes', {}))} scenes, "
        f"{chrono.get('anchors', 0)} anchors/{chrono.get('inferred', 0)} inferred, "
        f"{len(tree['nodes'])} nodes/{len(tree['edges'])} edges -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
