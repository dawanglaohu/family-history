# family-history

A weekend prototype: feed it old family photos and it reconstructs a timeline,
fills in the gaps, and turns the result into an interactive family tree you can
actually explore. Not a genealogy database — a living scrapbook that writes
itself.

The full pipeline (**Steps 1–6**) is implemented: ingest, scene/person
clustering, chronology inference (the hard part — ordering EXIF-less photos),
story generation, an interactive tree, and an end-to-end verification run.

Run the whole thing in one shot:

```bash
python src/pipeline.py examples/scans -o data/manifest.json
python src/verify.py --out verification --count 30   # Step 6 acceptance test
```

The committed `verification/` directory is the output of that last command:
30 synthetic photos, 3 scenes, 10 EXIF anchors + 20 inferred placements, a
batched pin request, and `tree.svg` — the acceptance test from the roadmap.

## Why this exists

Most photo-history apps demand that you manually tag hundreds of photos before
they show you anything interesting, so people abandon them. This prototype
aims to let the model do the heavy lifting from unorganized camera rolls and
old scans, without turning photo organizing into data entry. People care about
the stories and the vibe, not census records.

## Layout

```
src/ingest.py       # Step 1: scan folder -> data/manifest.json
src/cluster.py      # Step 2: perceptual embeddings -> scenes, face -> person_ids
src/chronology.py   # Step 3: EXIF anchors + 4 signals -> inferred_order + confidence
src/story.py        # Step 4: captions + 2-3 sentence stories + batched pin_request
src/tree.py         # Step 5: React Flow nodes/edges (temporal + family links)
src/verify.py       # Step 6: synthetic 30 photos -> end-to-end run + tree.svg
src/pipeline.py     # runs Steps 1-5 end-to-end
tests/              # 18 stdlib unittest cases
examples/scans/     # sample photo folder (synthetic stand-ins)
verification/       # committed Step 6 acceptance run
data/               # generated manifests (git-ignored except .gitkeep)
```

### Swapping in real models

The clustering and chronology modules are fully implemented and runnable
offline with dependency-free perceptual features. Each has an explicit
production swap point:

- `cluster.image_embedding` — replace with a real CLIP encoder.
- `cluster.face_embeddings` — wire in insightface / face-recognition; the
  `person_ids` grouping and face-age chronology signal then activate.
- `chronology.ocr_date_hint` — replace filename/sidecar text with pytesseract
  output from postcards and clippings.
- `chronology.face_age_hint` — replace the face-size proxy with an age model.

The optimisation, anchor pinning, and confidence logic do not change.

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
- **Step 2 — Cluster** (done): cosine similarity + connected components over
  perceptual embeddings for scenes; face embeddings (swap point) grouped into
  `person_ids`. CLIP / HDBSCAN drop in without changing the grouping code.
- **Step 3 — Chronology (the hard part)** (done): EXIF-dated photos become
  hard anchors (confidence 1.0). For the rest, a weighted blend of four
  signals — face-age progression, film-stock color/grain batch signature,
  seasonal/decor cues, and OCR-style dates — produces `inferred_order` plus a
  confidence strictly below the anchors.
- **Step 4 — Story** (done): caption clusters (VLM swap point), write 2–3
  sentence stories, and batch low-confidence clusters into a *single*
  "pin these events" question (max 3 events) — never 200 prompts.
- **Step 5 — Tree** (done): React Flow compatible nodes (exact schema:
  `title`/`story`/`photo_refs`/`confidence`/`pin_request` + `position`) and
  temporal/family edges; `tree.svg` renders a preview.
- **Step 6 — Verify** (done): generates a synthetic 30-photo folder, runs the
  pipeline end-to-end, and commits the README + SVG — see `verification/`.

## License

MIT
