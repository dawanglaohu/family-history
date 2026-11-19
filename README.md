# family-history

A weekend prototype: feed it old family photos and it reconstructs a timeline,
fills in the gaps, and turns the result into an interactive family tree you can
actually explore. Not a genealogy database — a living scrapbook that writes
itself.

This repo currently contains **Steps 1–3** of the pipeline: ingest, scene/person
clustering, and chronology inference (the hard part — ordering EXIF-less
photos). Story generation and the React Flow tree are planned in the roadmap
below.

Run the whole thing in one shot:

```bash
python src/pipeline.py examples/scans -o data/manifest.json
```

## Why this exists

Most photo-history apps demand that you manually tag hundreds of photos before
they show you anything interesting, so people abandon them. This prototype
aims to let the model do the heavy lifting from unorganized camera rolls and
old scans, without turning photo organizing into data entry. People care about
the stories and the vibe, not census records.

## Layout

```
src/ingest.py       # Step 1: scan folder -> data/manifest.json
src/cluster.py      # Step 2: CLIP-style embeddings -> scenes, face -> person_ids
src/chronology.py   # Step 3: EXIF anchors + 4 signals -> inferred_order + confidence
src/pipeline.py     # runs Steps 1-3 end-to-end
tests/              # 12 stdlib unittest cases
examples/scans/     # sample photo folder (synthetic stand-ins)
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
- **Step 4 — Story**: caption clusters with a vision-language model, write
  2–3 sentence stories, and batch low-confidence clusters into a *single*
  "pin these events" question to the user — never 200 prompts.
- **Step 5 — Tree**: React Flow renders clusters as nodes and temporal/family
  links as edges; click a node to expand its photos and story.
- **Step 6 — Verify**: generate a synthetic 30-photo sample folder, run the
  pipeline end-to-end, and commit a README with screenshots.

## License

MIT
