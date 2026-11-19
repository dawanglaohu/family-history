"""Tests for Step 4 (story), Step 5 (tree), Step 6 (verify)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingest import ingest_folder  # noqa: E402
from cluster import cluster_photos  # noqa: E402
from chronology import order_photos  # noqa: E402
from story import generate_stories, write_story, caption_cluster  # noqa: E402
from tree import build_tree  # noqa: E402
from verify import make_sample, verify  # noqa: E402
from pipeline import run  # noqa: E402


class StoryTests(unittest.TestCase):
    def test_story_is_2to3_sentences(self):
        story = write_story(["xmas tree 1998", "grandma 1998"], 12, "scans/1998")
        self.assertGreaterEqual(story.count("."), 2)

    def test_captions_read_metadata(self):
        caps = caption_cluster(["/photos/1998/xmas_001.jpg"])
        self.assertIn("1998", caps[0])

    def test_pin_request_is_batched(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            Image.new("RGB", (8, 8), "red").save(root / "a.jpg")
            m = ingest_folder(root)
            cluster_photos(m)
            order_photos(m)
            generate_stories(m, confidence_threshold=1.1)  # force all low
            pin = m.get("pin_request")
            self.assertIsNotNone(pin)
            self.assertLessEqual(len(pin["events"]), 3)


class TreeTests(unittest.TestCase):
    def test_node_shape_matches_schema(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            im = Image.new("RGB", (8, 8), "red")
            exif = im.getexif(); exif[36867] = "2000:01:01 00:00:00"
            im.save(root / "a.jpg", exif=exif)
            m = run(root)
            nodes = m["tree"]["nodes"]
            self.assertTrue(nodes)
            n = nodes[0]
            self.assertIn(n["type"], ("sceneNode",))
            for key in ("title", "story", "photo_refs", "confidence", "pin_request"):
                self.assertIn(key, n["data"])
            self.assertIn("x", n["position"])
            self.assertIn("y", n["position"])

    def test_edges_link_all_scenes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_sample(root / "scans", 9)
            m = run(root / "scans")
            nodes = m["tree"]["nodes"]
            temporal = [e for e in m["tree"]["edges"] if e["type"] == "temporal"]
            self.assertEqual(len(temporal), max(0, len(nodes) - 1))


class VerifyTests(unittest.TestCase):
    def test_verify_30_photos_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "verification"
            m = verify(out, count=30)
            self.assertEqual(m["count"], 30)
            self.assertTrue((out / "manifest.json").exists())
            self.assertTrue((out / "tree.svg").exists())
            self.assertTrue((out / "README.md").exists())
            self.assertTrue(m["tree"]["nodes"])
            self.assertTrue(all(p["inferred_order"] is not None for p in m["photos"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
