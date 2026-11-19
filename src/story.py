"""Step 4 story generation: caption clusters and write short stories.

A vision-language model captions each scene cluster and a prompt turns the
captions into a 2-3 sentence story. Low-confidence clusters are NOT interrogated
one by one — they are batched into a single ``pin_request`` with at most three
events for the user to confirm. That is the whole point: never 200 prompts.

Production swap point: ``caption_cluster`` (call a VLM on the actual images).
The batching / confidence logic stays the same.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

SEASON_MONTHS = {
    12: "winter holidays", 1: "winter", 2: "winter",
    3: "early spring", 4: "spring", 5: "spring", 6: "summer",
    7: "summer", 8: "late summer", 9: "autumn", 10: "autumn", 11: "late autumn",
}


def caption_cluster(photo_paths: list[str], folder_hint: str = "") -> list[str]:
    """Production swap point: run a vision-language model over the images.

    Default produces human-readable captions from path metadata so the repo
    runs offline. Each caption is one short phrase describing a photo.
    """
    captions = []
    for path in photo_paths:
        name = Path(path).stem.lower().replace("_", " ").replace("-", " ")
        year = ""
        for token in path.replace("\\", "/").split("/"):
            if len(token) == 4 and token.isdigit() and 1900 < int(token) < 2100:
                year = token
                break
        captions.append(f"{name}{(' in ' + year) if year else ''}")
    return captions


def _infer_month(manifest: dict, scene_id: str) -> Optional[int]:
    for photo in manifest["photos"]:
        if photo.get("scene_id") == scene_id:
            hint = photo.get("season_hint")
            if hint is not None:
                return int(hint)
            when = photo.get("exif_date")
            if when:
                return int(when[5:7])
    return None


def write_story(captions: list[str], month: Optional[int], folder_hint: str) -> str:
    """Turn captions into a 2-3 sentence story (template prompt stand-in).

    Production swap point: replace with an LLM prompt over the captions. The
    deterministic template keeps tests stable and the repo dependency-free.
    """
    season = SEASON_MONTHS.get(month, "") if month else ""
    subject = folder_hint.split("/")[-1] if folder_hint else "a family gathering"
    unique = list(dict.fromkeys(captions))[:3]
    what = ", ".join(unique)
    lead = f"A set of photos from {subject}"
    if season:
        lead += f" in {season}"
    return f"{lead}. Scenes include {what}. The stories behind them matter more than the dates."


def generate_stories(manifest: dict, confidence_threshold: float = 0.7) -> dict:
    """Annotate each scene with a story and collect batched pin requests."""
    root = Path(manifest["source"])
    scenes = manifest.get("scenes", {})
    # Map scene -> photos
    by_scene: dict[str, list[dict]] = {sid: [] for sid in scenes}
    for photo in manifest["photos"]:
        sid = photo.get("scene_id")
        if sid in by_scene:
            by_scene[sid].append(photo)

    stories: dict[str, dict] = {}
    low_confidence: list[dict] = []

    for sid, photos in by_scene.items():
        paths = [str(root / p["path"]) for p in photos]
        captions = caption_cluster(paths, sid)
        month = _infer_month(manifest, sid)
        conf = min([p.get("confidence", 0.5) for p in photos], default=0.5)
        story = write_story(captions, month, sid)
        stories[sid] = {
            "title": f"Scene {sid.split('_')[-1]}",
            "story": story,
            "captions": captions,
            "confidence": round(float(conf), 2),
        }
        if conf < confidence_threshold:
            low_confidence.append({
                "scene_id": sid,
                "suggested": story,
                "confidence": round(float(conf), 2),
            })

    # ONE batched pin request with at most 3 events, never one per photo.
    pin_request = None
    if low_confidence:
        pin_request = {
            "question": "Can you pin these events? (just confirm or correct)",
            "events": low_confidence[:3],
        }

    manifest["stories"] = stories
    manifest["pin_request"] = pin_request
    return manifest
