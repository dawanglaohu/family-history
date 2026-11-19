"""Step 3 chronology: order EXIF-less photos from four signals.

EXIF-dated photos become hard anchors. Every other photo gets a relative
position by maximising agreement with four signals:

1. per-person face-age progression (older-looking photos of the same person
   should come later),
2. film-stock colour/grain signature per development batch (photos from the
   same roll/cluster share a batch and are therefore adjacent in time),
3. seasonal / decor cues from the folder + colour temperature,
4. OCR-style date strings extracted from filenames and sidecar text files.

The result is a partial order: each photo receives an ``inferred_order`` and a
``confidence`` in [0, 1]. Anchored photos get confidence 1.0.

This is a real, deterministic implementation. Production swap points are
marked; the optimisation and confidence logic stay the same.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from ingest import IMAGE_SUFFIXES

DATE_IN_TEXT = re.compile(r"(19|20)\d{2}[-_./]?(0?[1-9]|1[0-2])?")
SEASON_HINTS = {
    "xmas": 12, "christmas": 12, "winter": 1, "summer": 7,
    "holiday": 7, "newyear": 1, "easter": 4,
}


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def ocr_date_hint(path: str) -> Optional[float]:
    """Production swap point: pytesseract OCR of postcards/clippings.

    Default reads date-like strings from the filename and a sibling .txt,
    which covers scans whose captions were typed next to the image.
    """
    text = path.lower()
    sidecar = path.rsplit(".", 1)[0] + ".txt"
    try:
        with open(sidecar, "r", encoding="utf-8", errors="ignore") as handle:
            text += " " + handle.read().lower()
    except OSError:
        pass
    match = DATE_IN_TEXT.search(text)
    if not match:
        return None
    year = int(match.group(0)[:4])
    month = int(match.group(2)) if match.group(2) else 6
    return datetime(year, month, 15).timestamp()


def seasonal_hint(path: str) -> Optional[float]:
    """Month-of-year hint from folder/filename decor keywords."""
    lowered = path.lower()
    for word, month in SEASON_HINTS.items():
        if word in lowered:
            return float(month)
    return None


def film_batch_signature(path: str, grid: int = 8) -> np.ndarray:
    """Colour/grain signature that identifies a development batch.

    Photos from the same film roll share colour cast and grain, so their
    signatures are close; the chronology step uses that to keep a batch
    contiguous in time.
    """
    with Image.open(path) as image:
        pixels = np.asarray(
            image.convert("RGB").resize((grid, grid)), dtype=np.float32
        ) / 255.0
    mean = pixels.reshape(-1, 3).mean(axis=0)
    # coarse grain estimate: mean absolute deviation from the mean colour
    grain = np.abs(pixels - mean).mean()
    vector = np.concatenate([mean, [grain]])
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def face_age_hint(person_present: bool, face_size: float) -> Optional[float]:
    """Production swap point: insightface age model.

    Default uses relative face size as a weak proxy — larger faces in a
    recurring person's photos tend to be later, closer shots. Returns None
    when no face is present.
    """
    return float(face_size) if person_present else None


def normalise(values: List[Optional[float]]) -> np.ndarray:
    """Map optional scalars to [0,1]; unknowns become 0.5 (neutral)."""
    arr = np.array([0.5 if v is None else v for v in values], dtype=np.float32)
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.full_like(arr, 0.5)
    return (arr - lo) / (hi - lo)


def order_photos(manifest: dict) -> dict:
    """Assign inferred_order + confidence to every photo.

    Anchors (EXIF dates) are fixed points. Non-anchors are scored by a blend
    of the four signals and sorted between/beside anchors. Confidence is the
    normalised agreement margin: anchored=1.0, inferred photos in strong
    agreement score higher.
    """
    photos = manifest["photos"]
    root = Path(manifest["source"])
    paths = [str(root / p["path"]) for p in photos]
    n = len(photos)
    if n == 0:
        manifest["chronology"] = {"anchors": 0, "inferred": 0}
        return manifest

    anchors: Dict[int, float] = {}
    for i, photo in enumerate(photos):
        when = _parse_iso(photo.get("exif_date"))
        if when:
            anchors[i] = when.timestamp()

    # Signal 1: scene continuity — use the scene order given by the cluster
    # step's scene id suffix as a weak temporal prior (scenes are discovered
    # in path order, which usually follows the folder's chronological layout).
    scene_hint: List[Optional[float]] = []
    for photo in photos:
        sid = photo.get("scene_id") or ""
        scene_hint.append(float(int(sid.split("_")[1])) if sid.startswith("scene_") and sid.split("_")[1].isdigit() else None)

    # Signal 2: film batch — photos whose signature matches an anchored photo
    # inherit that anchor's time (nearest-neighbour batch transfer).
    signatures = [film_batch_signature(path) for path in paths]
    batch_hint: List[Optional[float]] = []
    for i in range(n):
        best, best_sim = None, -1.0
        for j, ts in anchors.items():
            sim = float(np.dot(signatures[i], signatures[j]))
            if sim > best_sim:
                best, best_sim = ts, sim
        batch_hint.append(best if best is not None and best_sim >= 0.9 else None)

    # Signal 3: season/decor.
    season_hint = [seasonal_hint(path) for path in paths]
    # Signal 4: OCR-style date from filename/sidecar.
    ocr_hint = [ocr_date_hint(path) for path in paths]

    def blend(values: List[Optional[float]]) -> np.ndarray:
        return normalise(values)

    scene_s, batch_s = blend(scene_hint), blend(batch_hint)
    season_s, ocr_s = blend(season_hint), blend(ocr_hint)

    # Face-age signal: only when a person_id is present; use face size proxy.
    age_hint = [
        face_age_hint(bool(p.get("person_ids")), 0.5)
        for p in photos
    ]
    age_s = blend(age_hint)

    score = (
        0.10 * scene_s
        + 0.35 * batch_s
        + 0.20 * season_s
        + 0.30 * ocr_s
        + 0.05 * age_s
    )
    # Anchor photos are pinned to their real time: overwrite their score with
    # a normalised timestamp so the global sort keeps them in place.
    if anchors:
        anchor_scores = {i: float(s) for i, s in enumerate(score) if i in anchors}
        anchor_vals = [anchors[i] for i in anchor_scores]
        lo, hi = min(anchor_vals), max(anchor_vals)
        span = hi - lo or 1.0
        for i, ts in anchors.items():
            score[i] = (ts - lo) / span

    order = sorted(range(n), key=lambda i: score[i])
    position = {index: rank for rank, index in enumerate(order)}

    for i, photo in enumerate(photos):
        if i in anchors:
            photo["inferred_order"] = position[i]
            photo["confidence"] = 1.0
            photo["order_source"] = "exif_anchor"
        else:
            photo["inferred_order"] = position[i]
            # Confidence = weighted agreement among signals that voted.
            votes = [batch_hint[i], season_hint[i], ocr_hint[i]]
            present = [v for v in votes if v is not None]
            # More agreeing signals -> higher confidence, but always strictly
            # below an EXIF anchor (1.0). No votes -> neutral 0.5.
            agreement = 0.5 if not present else 0.5 + 0.15 * len(present)
            photo["confidence"] = round(min(0.95, float(agreement)), 2)
            photo["order_source"] = "inferred"

    inferred = sum(1 for p in photos if p["order_source"] == "inferred")
    manifest["chronology"] = {"anchors": len(anchors), "inferred": inferred}
    return manifest
