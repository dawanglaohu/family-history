"""Step 2 clustering: group photos into scenes and assign person IDs.

The pipeline is designed so heavy models can be swapped in:

* ``image_embedding`` by default uses a deterministic, dependency-free
  perceptual signature (average colour + luminance grid). Replace it with a
  CLIP embedding for production; the clustering code does not change.
* ``face_embeddings`` by default returns no faces. Wire in insightface / a
  face-recognition model and the ``person_ids`` grouping below works as-is.

Clustering uses cosine similarity + a simple connected-components threshold,
which is HDBSCAN-like for our purposes and keeps the repo runnable offline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from ingest import IMAGE_SUFFIXES

SCENE_SIMILARITY_THRESHOLD = 0.86


def _path_root(path: str) -> str:
    """Container-relative folder for the photo (e.g. 'scans/1998')."""
    return "/".join(path.split("/")[:-1])


def image_embedding(path: str, grid: int = 4) -> np.ndarray:
    """Return a normalised perceptual signature for an image.

    Production swap point: replace with ``clip_model.encode(image)``. The
    returned vector only needs to be a 1-D float array; similar scenes should
    have a high cosine similarity.
    """
    with Image.open(path) as image:
        image = image.convert("RGB").resize((grid, grid))
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    # 3 channels of mean colour + flattened luminance grid
    mean_rgb = pixels.reshape(-1, 3).mean(axis=0)
    luminance = pixels.mean(axis=2).reshape(-1)
    vector = np.concatenate([mean_rgb, luminance])
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def face_embeddings(path: str) -> list[np.ndarray]:
    """Return face embedding vectors found in an image.

    Production swap point: run insightface/face-recognition here. The default
    returns an empty list (no faces detected), which is fine for the sample
    folder and keeps the pipeline offline-runnable.
    """
    return []


def cosine_matrix(vectors: Sequence[np.ndarray]) -> np.ndarray:
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    stacked = np.vstack(vectors)
    return stacked @ stacked.T


def connected_components(similarity: np.ndarray, threshold: float) -> list[list[int]]:
    """HDBSCAN-like grouping: union photos whose cosine sim >= threshold."""
    n = similarity.shape[0]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def group_person_ids(vectors: Iterable[np.ndarray], threshold: float = 0.9) -> list[str]:
    """Cluster face embeddings into stable-ish person IDs."""
    vectors = list(vectors)
    if not vectors:
        return []
    sim = cosine_matrix(vectors)
    components = connected_components(sim, threshold)
    ids: dict[int, str] = {}
    for component in components:
        seed = np.vstack([vectors[i] for i in component]).mean(axis=0).tobytes()
        pid = "p_" + hashlib.sha256(seed).hexdigest()[:6]
        for i in component:
            ids[i] = pid
    return [ids[i] for i in range(len(vectors))]


def cluster_photos(manifest: dict) -> dict:
    """Annotate manifest records with scene_id and person_ids.

    Adds a top-level ``scenes`` summary: {scene_id: [photo ids]}.
    """
    photos = manifest["photos"]
    root = Path(manifest["source"])
    vectors = [image_embedding(str(root / p["path"])) for p in photos]
    sim = cosine_matrix(vectors)
    components = connected_components(sim, SCENE_SIMILARITY_THRESHOLD)

    scene_ids: dict[int, str] = {}
    scenes: dict[str, list[str]] = {}
    for component in components:
        # Use the most common folder as a readable scene name when possible.
        folders = [_path_root(photos[i]["path"]) for i in component]
        folder = max(set(folders), key=folders.count) or "scene"
        sid = "scene_" + hashlib.sha256(folder.encode()).hexdigest()[:4]
        # Disambiguate if two components land in the same folder.
        suffix = len([s for s in scenes if s.startswith(sid)])
        if suffix:
            sid = f"{sid}_{suffix}"
        scene_ids.update({i: sid for i in component})
        scenes.setdefault(sid, [])
        for i in component:
            scenes[sid].append(photos[i]["id"])

    for i, photo in enumerate(photos):
        photo["scene_id"] = scene_ids[i]
        photo["person_ids"] = group_person_ids(
            face_embeddings(str(root / photo["path"]))
        )

    manifest["scenes"] = scenes
    return manifest
