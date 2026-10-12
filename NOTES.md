# Notes

Running log. Nothing here is a commitment; newest at the top.

## 2026-10-11

### Decided (for now)
- A Coruña only. No galerías (cameras point forward from cars; upper floors barely appear).
- Scraping is fine, paying for map APIs is not.
- Straightening: later.
- Time dimension: interesting, keep looking for data that supports it.
- Keep docs loose. Iterate.

### Sources
| Source | Coruña | Access | Notes |
|---|---|---|---|
| Mapillary | 86k images, 7.5k panos, 21.6k detected objects | official API, free | 73% is one 2019 dashcam pass. Low-res, blurry, forward-facing. Comes with object detections and 3D-placed features, which is what the crops use. |
| Apple Look Around | 105k car panoramas, 6 faces each at 5632×4864 | `streetlevel` (reverse-engineered, no key) | Superb quality. Dates: 2019-04 14k, 2023-05 5k, 2024-07 14k, 2025-03/04 11k, 2026-07 61k. Apple only serves the latest pass per street, so dates are a patchwork, not a time machine. No detections, we bring our own. Faces use Apple's own projection, needs reprojecting before straightening. |
| Google Street View | 44k panos, history back to 2008 (median 7 captures per spot) | `streetlevel` coverage tiles + per-id metadata, no key | Only the location-search endpoint is dead (`SingleImageSearch`); tiles, metadata with history and full-res 16384×8192 equirect downloads all work. The time machine. |
| Bing Streetside | not checked properly | `streetlevel` | Spain coverage is thin anyway. |
| KartaView, Panoramax | nothing useful | | |
| Wikimedia Commons | 500+ geotagged files within 2 km of the Mariña | API | Could feed a historical layer for a few famous objects. |
| OSM | lamps, benches, bins tagged as nodes | Overpass | Possible ground truth for counts. Not checked yet. |

### What the Mapillary crops look like
14.4k crops from 21.6k features, 59 classes with 48px+ crops. Top: store signs 1.5k, ads 1.3k, poles 715,
trash cans 700, street lights 510, zebra crossings 476, straight arrows 412, utility poles 370, traffic
signs 307, traffic lights 260, benches 192, manholes 167. Viewer: `data/coruna/typologies.html`.
- Lots of duplicates (same sign from several frames). Dedupe comes before any grid.
- Quality all over the place. Night, blur, tiny.
- A few percent are the wrong thing (windows, laundry as "store sign").

### Models
- SAM 3 loads from an ungated mirror (`JobHarvest/sam3`, transformers format) since Meta's repo gates
  access. Runs fine in 8 GB, a few seconds per image for ten prompts.
- SAM 3 vs Mapillary on 24 images (`data/coruna/sam3_vs_mapillary.html`): SAM 3 wins clearly.
  Mapillary's polygons are fragments (a lamp head, not the lamp post) and it misses most instances
  (1 traffic sign vs 51, 2 manholes vs 31, 32 store signs vs 135 at threshold 0.5). SAM 3 gives whole
  objects as single instances, and works on the Apple face too. Caveats: prompts overlap ("pole" grabs
  lamp posts and sign posts), and the lower-score boxes need a look. Mapillary still has one thing SAM
  doesn't: a 3D position per object, which is what makes cross-image dedupe easy. With SAM we need our
  own (camera pose plus bearing, or embedding dedupe).
- Moondream 3 (`moondream/moondream3-preview`, ungated) still to try, for checking crops and naming clusters.
- The GPU has another process using 16 GB (`uvicorn app:app` on port 8000). 8 GB is enough for SAM 3.

### Apple pipeline (evening)
- `apple_detect.py`: faces stream from Apple (25/s with 16 threads) through SAM 3 at 0.22 s per face
  all-in. Side faces only for signs. Thinned to one pano per 6 m. Crops plus quarter-res masks kept,
  faces discarded. Whole 2026 set at this rate: about 20k panos, 40k faces, 2.5 hours.
- Prompt sensitivity: "store sign" works, "shop sign" and "storefront sign" return nothing. Worth a
  small prompt sweep per object type before each big run.
- "store sign" precision is about half: traffic signs, notices, street plates and the tunnel's
  emergency exit signs (noisy, dark) come along. `filter_crops.py` (SigLIP zero-shot) sorts them.
- OCR: Florence-2 large (`<OCR>`) reads clean signs well, 0.4 s per crop. EasyOCR and RapidOCR were
  much worse on stylised lettering. Moondream 3 would be the upgrade but needs more VRAM than is free.
- Dedupe still open: the same banner appears from 3 consecutive panos even after thinning.

### Open threads
- Which object first? Store signs have the most variety; street lights and trash cans are more Becher.
  Apple's quality might make small things (manholes, doorbells, door handles) viable that Mapillary can't do.
- Dedupe: GPS clustering plus visual embedding (DINOv2) seems the obvious move.
- Time: Mapillary 2019 dashcam vs Apple 2026 for the same streets? Low quality on the 2019 side.
  Google history would be far better if an endpoint comes back.
- Presentation: web grid, print, or something playful (see Yufeng Zhao in PRIOR_ART.md).

## Sign histories (2026-10-12)
- `match_sources.py`: Google detection ray vs Apple sign position. Keeps perp ≤ max(1 m, 5 % of range), range 2–28 m,
  height within 1.5 m, angular width ratio 0.4–2.5, and the ray within 75° of the sign's facing (mean Apple ray).
- Known weakness: on a busy corner (poster column in front of a fascia, banner above it) the per-date pick
  alternates between objects. Next idea: pick per-date candidates as a chain (Viterbi) maximising DINOv2
  similarity between consecutive picks plus geometry, so a run of the same object wins over alternation.
- `history_strip.py` cuts 4:3 windows 3.2 sign-widths wide at the detection's bearing/pitch; undetected dates
  with a pano ≤ 22 m on the facing side get a geometric window flagged `detected=false`.
- Viewer sprites: one vertical sprite of 480×360 windows per sign plus one horizontal strip of 240×160 crops,
  because an artifact publish takes at most 255 files / 64 MB (two publishes for 160 signs, 86 MB).
