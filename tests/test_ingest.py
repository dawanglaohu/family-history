"""Minimal stdlib tests for the ingest step (no pytest dependency)."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingest import ingest_folder, parse_exif_date, sha256_file  # noqa: E402


class IngestTests(unittest.TestCase):
    def test_parse_exif_date(self):
        self.assertEqual(parse_exif_date("1998:12:25 10:30:00"), "1998-12-25T10:30:00")
        self.assertIsNone(parse_exif_date("not-a-date"))
        self.assertIsNone(parse_exif_date(None))

    def test_hash_is_stable(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.jpg"
            p.write_bytes(b"family photo bytes")
            self.assertEqual(sha256_file(p), sha256_file(p))
            self.assertEqual(len(sha256_file(p)), 64)

    def test_ingest_marks_exifless_photos_for_inference(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "1998").mkdir()
            # one JPEG with a DateTimeOriginal, one without
            with_exif = root / "1998" / "xmas.jpg"
            img = Image.new("RGB", (8, 6), "red")
            exif = img.getexif()
            exif[36867] = "1998:12:25 10:30:00"
            img.save(with_exif, exif=exif)
            without = root / "2003" / "polaroid.jpg"
            without.parent.mkdir()
            Image.new("RGB", (5, 7), "blue").save(without)

            manifest = ingest_folder(root)
            self.assertEqual(manifest["count"], 2)
            self.assertEqual(manifest["with_exif"], 1)
            self.assertEqual(manifest["needs_inference"], 1)
            by_path = {r["path"]: r for r in manifest["photos"]}
            self.assertEqual(by_path["1998/xmas.jpg"]["exif_date"], "1998-12-25T10:30:00")
            self.assertFalse(by_path["1998/xmas.jpg"]["needs_inference"])
            self.assertTrue(by_path["2003/polaroid.jpg"]["needs_inference"])
            self.assertEqual(by_path["2003/polaroid.jpg"]["dimensions"], [5, 7])

    def test_cli_writes_manifest(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            scans = root / "scans"
            scans.mkdir()
            Image.new("RGB", (3, 4), "green").save(scans / "a.jpg")
            out = root / "manifest.json"
            result = subprocess.run(
                [sys.executable, str(ROOT / "src" / "ingest.py"), str(scans), "-o", str(out)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["count"], 1)
            self.assertTrue(data["photos"][0]["needs_inference"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
