# family-history

A weekend prototype: feed it old family photos and it reconstructs a timeline,
fills in the gaps, and turns the result into an interactive family tree you can
actually explore. Not a genealogy database — a living scrapbook that writes
itself.

This repo currently contains **Step 1 (ingest)** of the pipeline. Later steps
(clustering, chronology inference, story generation, and the React Flow tree)
are planned in the roadmap below.

## Why this exists

Most photo-history apps demand that you manually tag hundreds of photos before
they show you anything interesting, so people abandon them. This prototype
aims to let the model do the heavy lifting from unorganized camera rolls and
old scans, without turning photo organizing into data entry. People care about
the stories and the vibe, not census records.

## Layout

```
src/ingest.py          # Step 1: scan folder -> data/manifest.json
tests/test_ingest.py   # stdlib unittest suite
examples/scans/        # sample photo folder (synthetic stand-ins)
data/                  # generated manifests (git-ignored except .gitkeep)
```

## Running Step 1

Uses Pillow for EXIF/dimension reads. With a Python that has Pillow installed:

```bash
python src/ingest.py examples/scans -o data/manifest.json
```

Example output:

```
ingested 2 photos (1 with EXIF, 1 need inference) -> data/manifest.json
```

Each record in `manifest.json`:

```json
{
  "id": "sha256:9f2c...",
  "path": "scans/1998/xmas_003.jpg",
  "hash": "9f2c1a...",
  "exif_date": null,
  "dimensions": [1200, 1600],
  "needs_inference": true,
  "scene_id": null,
  "person_ids": [],
  "inferred_order": null,
  "confidence": null
}
```

Photos with `exif_date: null` are the whole problem the later chronology step
solves: old polaroids and scans have no EXIF, so their order must be inferred.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Roadmap

- **Step 1 — Ingest** (done): walk `/scans`, pull EXIF `DateTimeOriginal`,
  SHA-256 each file, emit `manifest.json`, flag EXIF-less photos.
- **Step 2 — Cluster**: CLIP image embeddings + HDBSCAN for scenes; face
  detection/embeddings for recurring `person_ids`.
- **Step 3 — Chronology (the hard part)**: EXIF-dated photos become anchors.
  For the rest, build a partial order from face-age progression, film-stock
  color/grain signatures per development batch, seasonal/decor cues, and OCR
  dates on postcards/clippings; solve a global ordering respecting anchors and
  output a confidence score per placement.
- **Step 4 — Story**: caption clusters with a vision-language model, write
  2–3 sentence stories, and batch low-confidence clusters into a *single*
  "pin these events" question to the user — never 200 prompts.
- **Step 5 — Tree**: React Flow renders clusters as nodes and temporal/family
  links as edges; click a node to expand its photos and story.
- **Step 6 — Verify**: generate a synthetic 30-photo sample folder, run the
  pipeline end-to-end, and commit a README with screenshots.

## License

MIT
