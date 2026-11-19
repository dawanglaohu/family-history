"""Tests for Step 2 (cluster) and Step 3 (chronology)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingest import ingest_folder  # noqa: E402
from cluster import (  # noqa: E402
    cluster_photos, cosine_matrix, connected_components, group_person_ids,
    SCENE_SIMILARITY_THRESHOLD,
)
from chronology import (  # noqa: E402
    order_photos, normalise, seasonal_hint, ocr_date_hint,
)
from pipeline import run  # noqa: E402


class ClusterTests(unittest.TestCase):
    def test_connected_components(self):
        sim = np.array([
            [1.0, 0.95, 0.2],
            [0.95, 1.0, 0.2],
            [0.2, 0.2, 1.0],
        ])
        groups = connected_components(sim, 0.86)
        self.assertEqual(len(groups), 2)
        self.assertEqual(sorted(len(g) for g in groups), [1, 2])

    def test_cosine_matrix_shape(self):
        vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        self.assertEqual(cosine_matrix(vectors).shape, (2, 2))

    def test_person_id_grouping(self):
        ids = group_person_ids([np.array([1.0, 0.0]), np.array([0.99, 0.01])])
        self.assertEqual(len(ids), 2)
        self.assertEqual(ids[0], ids[1])
        self.assertTrue(ids[0].startswith("p_"))

    def test_cluster_assigns_scene_ids(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            Image.new("RGB", (8, 8), "red").save(root / "a.jpg")
            Image.new("RGB", (8, 8), "red").save(root / "b.jpg")
            m = ingest_folder(root)
            cluster_photos(m)
            self.assertIn("scenes", m)
            self.assertTrue(all(p["scene_id"] for p in m["photos"]))
            self.assertEqual(
                m["photos"][0]["scene_id"], m["photos"][1]["scene_id"]
            )


class ChronologyTests(unittest.TestCase):
    def test_normalise_neutral_unknowns(self):
        out = normalise([None, None])
        np.testing.assert_array_equal(out, np.array([0.5, 0.5]))

    def test_seasonal_and_ocr_hints(self):
        self.assertEqual(seasonal_hint("scans/1998/xmas_001.jpg"), 12.0)
        self.assertIsNone(seasonal_hint("scans/random_001.jpg"))
        from datetime import datetime
        hint = ocr_date_hint("photos/1998-12-25.jpg")
        self.assertEqual(datetime.fromtimestamp(hint).year, 1998)

    def test_anchors_get_confidence_one(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            img = Image.new("RGB", (8, 8), "blue")
            exif = img.getexif()
            exif[36867] = "2001:06:15 12:00:00"
            img.save(root / "anchored.jpg", exif=exif)
            Image.new("RGB", (8, 8), "green").save(root / "unknown.jpg")
            m = ingest_folder(root)
            cluster_photos(m)
            order_photos(m)
            by_path = {p["path"]: p for p in m["photos"]}
            self.assertEqual(by_path["anchored.jpg"]["confidence"], 1.0)
            self.assertEqual(by_path["anchored.jpg"]["order_source"], "exif_anchor")
            self.assertEqual(by_path["unknown.jpg"]["order_source"], "inferred")
            self.assertLess(by_path["unknown.jpg"]["confidence"], 1.0)
            self.assertEqual(m["chronology"]["anchors"], 1)
            self.assertEqual(m["chronology"]["inferred"], 1)

    def test_pipeline_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            im = Image.new("RGB", (8, 8), "red")
            exif = im.getexif(); exif[36867] = "1999:01:02 03:04:05"
            im.save(root / "a.jpg", exif=exif)
            Image.new("RGB", (8, 8), "blue").save(root / "b.jpg")
            m = run(root)
            self.assertEqual(m["count"], 2)
            self.assertTrue(all(p["inferred_order"] is not None for p in m["photos"]))
            orders = sorted(p["inferred_order"] for p in m["photos"])
            self.assertEqual(orders, [0, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
