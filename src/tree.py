"""Step 5 tree: build a React Flow compatible graph from the manifest.

Node shape matches the schema shown in the design conversation:

    {
      "id": "scene_07",
      "type": "sceneNode",
      "data": {
        "title": "Christmas ~1998",
        "story": "...",
        "photo_refs": ["sha256:...", ...],
        "confidence": 0.72,
        "pin_request": null
      },
      "position": {"x": 340, "y": 120}
    }

Edges link scenes in temporal order (inferred_order of their photos) and,
when person_ids overlap, add a family/person link between scenes.
"""

from __future__ import annotations


NODE_SPACING_X = 320
NODE_SPACING_Y = 140


def _scene_order(manifest: dict, sid: str) -> float:
    orders = [
        p["inferred_order"] for p in manifest["photos"]
        if p.get("scene_id") == sid and p.get("inferred_order") is not None
    ]
    return min(orders) if orders else 1e9


def build_tree(manifest: dict) -> dict:
    scenes = manifest.get("scenes", {})
    stories = manifest.get("stories", {})
    photos_by_scene: dict[str, list[dict]] = {sid: [] for sid in scenes}
    for photo in manifest["photos"]:
        sid = photo.get("scene_id")
        if sid in photos_by_scene:
            photos_by_scene[sid].append(photo)

    ordered = sorted(scenes, key=lambda sid: _scene_order(manifest, sid))

    nodes = []
    for index, sid in enumerate(ordered):
        photos = photos_by_scene[sid]
        story = stories.get(sid, {})
        node = {
            "id": sid,
            "type": "sceneNode",
            "data": {
                "title": story.get("title", sid.replace("_", " ").title()),
                "story": story.get("story", ""),
                "photo_refs": [p["id"] for p in photos],
                "confidence": max(
                    [p.get("confidence", 0.5) for p in photos], default=0.0
                ),
                "pin_request": manifest.get("pin_request")
                if any(p.get("confidence", 1.0) < 0.7 for p in photos)
                else None,
            },
            "position": {
                "x": 80 + index * NODE_SPACING_X,
                "y": 80 + (index % 2) * NODE_SPACING_Y,
            },
        }
        nodes.append(node)

    edges = []
    for i in range(len(ordered) - 1):
        edges.append({
            "id": f"e_{ordered[i]}_{ordered[i+1]}",
            "source": ordered[i],
            "target": ordered[i + 1],
            "type": "temporal",
            "animated": False,
        })

    # Person links: scenes sharing a recurring person_id get a family edge.
    persons: dict[str, list[str]] = {}
    for photo in manifest["photos"]:
        sid = photo.get("scene_id")
        for pid in photo.get("person_ids", []):
            persons.setdefault(pid, [])
            if sid not in persons[pid]:
                persons[pid].append(sid)
    seen = set()
    for pid, sids in persons.items():
        for i in range(len(sids) - 1):
            a, b = sids[i], sids[i + 1]
            key = tuple(sorted((a, b)))
            if key in seen or a == b:
                continue
            seen.add(key)
            edges.append({
                "id": f"e_person_{pid}_{a}_{b}",
                "source": a,
                "target": b,
                "type": "family",
                "animated": False,
                "label": pid,
            })

    manifest["tree"] = {"nodes": nodes, "edges": edges}
    return manifest

