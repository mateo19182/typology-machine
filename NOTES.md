# Notes

Running log. Nothing here is a commitment; newest at the top.

## 2026-10-31 · static viewer on Cloudflare, graffiti retired

- User approved moving the static part to Cloudflare, keeping the API on this server, and
  retiring graffiti. Static shell + `ctx/` are a Worker with assets on route
  `typology.m19182.dev/*`; script-less routes keep `/api`, `/proposal-context` and the local
  crop fallback on the tunnel. Chosen over a proxy Worker + `typology-origin` hostname: no
  Worker requests at all (map thumbnails alone are hundreds of `/api/thumb` hits per visit),
  no DNS/tunnel change, rollback = delete one route.
- R2 crops: listing check found all 1,164,241 crops (33.4 GB) with matching sizes; 50 random
  public SHA-256 checks matched. Delivery was already activated in the export on 2026-10-22.
  No new crop uploads were needed. R2 CORS added for the alphabet canvas (`?cors` cache key).
- Map and wall thumbnails still come from `/api/thumb` on this server (edge-cached by
  extension). Pre-rendering them to R2 would be ~2.3M writes (~$10).

## 2026-10-12 · categories before appearance clusters

- The DINOv2/v3 comparison prompted a shift toward browsing recognizable object categories.
  Add graffiti and street-name plates alongside store signs and the earlier street-object types:
  advertisement posters, banners, bins, street lights, poles, benches, manhole covers, traffic signs,
  and junction boxes. These are intended categories, not a claim of complete city coverage.
- The comparison displays padded JPEG rectangles and embeds those same rectangles. SAM masks
  exist separately at quarter resolution. Keep saving masks, but leave them out of the current
  display and classification pipeline; use the rectangular crops for now.
- A `store sign` detection is a candidate, not a verified class. Incidental graffiti and street plates
  can seed galleries, but dedicated prompt trials on source imagery are needed to find missed objects.
  Graffiti must not be discarded merely because an `is_sign` check says no.
- Keep object category separate from optional appearance clusters within a category. Preserve the
  detector prompt, classification provenance, masks, source image, date, and position quality.
- Qwen embedding evaluation is deferred.

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
- Dedupe: prioritize avoiding false merges over finding every duplicate. The old proximity + DINO
  connected components merged different signs at the same storefront. Keep uncertain observations separate.
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

## Deduplication review feedback (2026-10-12)
- User found the original groups mostly incorrect, particularly different signs at the same location.
- V1's local-feature checks and conservative pairs looked better. User requested larger groups and no
  camera-angle filtering. V2 requires all pairs within each group to pass independent visual checks.
- User said the displayed v2 examples were correct and requested wider matching. This is feedback on
  reviewed examples, not a measured precision claim for the entire dataset.
- V3 searches across old group boundaries, expands the radius from 20 to 35 m, relaxes cosine similarity
  from 0.95 to 0.92 and competitor margin from 0.03 to 0.02. Local-feature checks stay unchanged; existing
  v2 groups remain intact. No camera-angle filter or group-size cap. Same-date constraints remain.
- User also approved the displayed v3 examples and requested further widening. V4 uses cosine 0.88,
  competitor margin 0.01 and 50 m search radius, preserving v3 groups and the same local-feature gates.
  Feedback refers to the displayed sample; new matches still need inspection.
- User approved the displayed v4 examples and requested more widening. V5 uses cosine 0.84,
  competitor margin 0.005 and radius 65 m, preserving v4 groups and local-feature thresholds.

- User approved the displayed v5 duplicate examples and requested much wider matching.
  V6 lowers similarity to 0.78 and competitor margin to 0.002; local-feature checks
  remain unchanged and v5 groups are preserved. User rejected a proposed 100 m
  radius, so v6 retains 65 m. Stop the initial 100 m experiment and regenerate both
  sources before publishing the review. This is sample feedback, not measured precision.
- User requested concurrent SAM inference. Workers must have disjoint assignments,
  preserve completed predictions, and continue sharing each image encoding across prompts.
- User requested local Trace-it stay stopped. Its four live CI containers were stopped
  with restart=no; data volumes retained. Local launcher is blocked by
  ~/.config/trace-it/disabled and trace-pay/AGENTS.md records the stop request.

- User approved the displayed v6 duplicates and requested wider matching on both
  sources. V7 uses similarity 0.70 and competitor margin 0, retaining the 65 m
  maximum camera distance, same-date restriction, local-feature thresholds and
  all-pairs group checks. All v6 groups remain together.

- Mateo chose 10 m sampling, all existing categories, and inference optimization.
  Preserve all old predictions, reuse compatible selected views, cache constant text
  embeddings, and transfer postprocessed tensors to CPU once per prompt. The old
  6 m run stopped cleanly with 6,143 completed views; 3,237 overlap the new selection.

- User explicitly approved increasing the duplicate search to 100 m after finding
  displayed v7 groups correct. This supersedes the earlier 65 m preference. V8
  also lowers similarity from 0.70 to 0.60, retaining a zero competitor margin,
  same-date requirement, local-feature thresholds and all-pairs group consistency.
  Preserve v7 groups and save a separate review; sample feedback is not a measured
  precision estimate.

- User reports all 161 v8 review examples are correct and requests more widening.
  V9 keeps 100 m / 0.60 similarity / zero competitor margin, and relaxes local
  inlier ratio from 0.85 to 0.80 and coverage from 0.35 to 0.25 on both crops.
  Minimum 20 inliers, descriptor ratio 0.65, RANSAC 2.5 px, same-date and all-pairs
  checks remain. Reevaluate saved local evidence, preserve every v8 group.

- User again reports the displayed duplicate examples look correct and asks to
  widen further. V10 preserves v9 groups and keeps 100 m / 0.60 similarity / zero
  competitor margin, lowering local agreement to 0.75 and coverage to 0.20.
  All other geometric and all-pairs checks remain. Feedback is qualitative.
- User requested graffiti examples. `explore/graffiti_review.py` builds 60 saved
  examples from the active multicategory run, with crops, boxed street context,
  exact prompts, scores, local browser labels and JSON export. Per prompt, sample
  10 random, 10 low-score and 10 high-score proposals across distinct panoramas.
  Exclude within-view duplicate proposals. These are unclassified SAM proposals.

- User says v10 review examples still look correct; v11 lowers local feature
  agreement from 0.75 to 0.70 and per-crop coverage from 0.20 to 0.15. Preserve
  every v10 group; keep 100 m, DINO 0.60, 20 inliers, same-date and all-pairs checks.
  This is sample approval, not measured precision.
- Graffiti review intentionally selected one proposal per panorama. Of its 60
  sampled views, 48 contain multiple unflagged graffiti proposals; the first has
  8, plus 1 flagged overlap. No one-object detection cap is imposed by our loop.
  `graffiti_scene_review.py` displays all unflagged proposals in 12 selected scenes
  to inspect omissions and cases where one box covers multiple marks. Current
  SAM threshold is 0.5 and minimum detection width/height is 32 pixels.

- User reports v11 duplicate examples still look correct and requests further
  widening. V12 lowers local agreement to 0.65 and coverage to 0.10, preserving
  v11 groups. Keep 100 m, similarity 0.60, minimum 20 inliers and all-pairs checks.
  Review sample feedback does not establish full-dataset precision.
- User approves graffiti results and requests object-type layers in the main
  frontend, with subagents. Show per-category detections and inspection controls;
  distinguish SAM proposals at camera coordinates from verified object locations.

- Object-layer snapshot published from 36,480 completed views: 323,846 raw SAM
  proposals, 299,223 after omitting within-view duplicate flags in the UI. Default
  graffiti + street names = 38,389 observations. Independent validation checked
  every exported ID, coordinate, crop link and count; sampled crop URLs return 200.
  Layers use camera coordinates, remain unclassified/cross-view undeduplicated,
  and are a manual snapshot. SAM service continues unchanged.

- User reports v12 duplicate examples still correct and asks whether 50% local
  feature agreement is enough. V13 tests exactly 0.50, retaining 0.10 coverage,
  20 inliers, 100 m, 0.60 similarity and all-pairs checks. Preserve every v12 group.
  Output: Apple 17,667 / GSV 1,791 duplicate groups. Review: 150 new/expanded groups,
  649 crops. Sample feedback still does not establish full-dataset precision.
- User noticed no full-image button on graffiti. Initial layer export set raw to
  null. Add per-view context URL and normalized boxes, with a lazy CPU JPEG cache
  behind Caddy. No bulk HEIC conversion or GPU changes. Same context serves all
  categories from a view; originals and masks remain unchanged.

- User authorized all four downstream tasks with subagents: category evaluation,
  new-category duplicate review, OCR/search and scheduled map refreshes. Category
  pilot freezes 96 crops and preserves the original 43 human labels. Intern2
  predictions stay separate; no accuracy without new human labels. A contact
  sheet shows many street-name proposals are actually traffic/shop signs, and
  the classifier itself has some errors. Do not auto-override map categories.
- New-category dedupe exposed different parking signs sharing an emblem and
  identical sign designs on different buildings. Crop checks alone cannot
  establish physical identity. Add background scene alignment before accepting
  new-category groups. Keep failed pilot versions for audit; don't seed their
  groups or collapse them in the frontend.
- CPU OCR Latin PP-OCRv5 pilot: 200 balanced crops, 149 nonempty results, around
  98 seconds. Chinese/English baseline quarantined, not exported. Enabled bounded
  OCR timer (1,000/15 minutes) and layer snapshot timer (every 15 minutes). OCR is
  draft text, not verified transcription. No GPU work beyond the completed
  96-crop Intern2 evaluation; SAM continues.
- Map updates are atomic, skip unchanged inputs, retain at least 24 hours/eight
  snapshots, and include OCR text. Browser polls every minute, offers explicit
  update, preserves map/layers/search and old data if refresh fails. Capture dates
  are separate from snapshot publication time. Actual accent-insensitive search
  for 'ramon del cueto' found three street-plate observations.
- New-category v3 finished with 17 groups (8 graffiti, 9 street-name proposals),
  35 grouped observations. Both known false merges fail the new gates. All
  52,942 snapshot observations are preserved. Full-scene spot checks found
  matching physical objects whose category proposals are still wrong; keep
  classification and object identity as separate review questions. Five dedupe
  tests and review browser checks passed. No frontend group collapse applied.

- User reports new category reviews and duplicate groups look correct. Saved
  category labels confirm 24/24 classifier agreement, across nine human category
  labels but only one street-name plate; this is a reviewed subset, not a citywide
  accuracy estimate. Duplicate feedback is qualitative, not per-case server data.
- At 19:31 on Oct12, SAM completed41,987/71,752 views at1.91views/s, slowest-worker
  ETA23:57. GPU100%, ~15GBused; CPU85%idle. OCR batch1000 took415.5seconds but the
  15-minute timer left about8minutes idle. Changed OCR timer to30seconds after
  each bounded batch, preserving locking/resume and CPU-only inference. Map export
  remains every15minutes. Expected OCR throughput roughlydouble pending measurement.

- User requested more detail on scaling category classification/deduplication and
  authorized grouped objects on the map now. Applied latest selected proposals
  to new map layers: legacy shop v13 by exact reused-crop IDs, multicategory v3 by
  exact observation IDs. No fuzzy linking or automatic category changes. All raw
  observations remain in the export. Client defaults to grouped mode, provides
  a raw toggle and every viewpoint, combines OCR for search, uses representative
  camera coordinates. Initial shop layer76,383 observations ->68,682 items;
  5,655 actual multi-view groups present in this snapshot, plus17 newcategorygroups.

## 2026-10-12: people/pets in ongoing SAM; backfill pending

- User approved full-person crops, no extra classification. Main run `765844a02f75` now adds `person` and `pet` (`dog`, `cat`) after the 48,009-view checkpoint. Original config and all pre-transition predictions remain unchanged.
- Runtime additions are pinned in `additional_prompts.json`; new results record attempted categories/prompts even for zero detections. Four workers continue the existing 10 m selection.
- **Pending after the current pass:** backfill person/pet detection on pre-transition views. Inventory: `data/coruna/apple/multicategory/765844a02f75/person_pet_backfill.json`; pilot's 50 person-covered views are recorded separately. Do not rerun existing category detections, replace approved duplicate groups, or treat unattempted categories as absent. Backfill has not started.
- User abandoned the independent street-text detector after review. Keep its artifacts; do not expand or publish its proposals.
- Current map: one Shop signs layer combining old Apple/Google representatives with new observations using exact source-crop IDs and accepted groups. People and Pets are regular layers, without a classifier or cross-image person/pet matching.
- UI: compact capture-year range, Select all/Clear all, ES/EN language preference, technical bilingual `method.html`. Removed explanatory panel subtitles.

## 2026-10-12 · Google GPU run prepared

- User requested a full-category Google run and parallel downloads of remaining
  indexed images, prioritizing total dollar cost. Frozen first bundle: 4,408 current
  panoramas, 35,264 views, 21.7 GB; run `9d984d64aa7e`. All 13 categories included
  from the start. Model revision matches Apple; source hashes and portable paths
  allow local import. No inference has run on the rented GPU yet.
- Google workers now shard/skip completed views before panorama projection.
  Existing Apple run configuration and saved predictions are preserved.
- Remaining current-image downloads started as `typology-google-fetch.service`,
  four workers, no thinning, no history expansion, 60 GiB disk reserve.
- Vast API account access works. Account currently has zero credit and no payment
  method. No rental purchased. Live compatible candidates were about $0.206/hour
  for on-demand RTX 3090 or $0.137/hour minimum interruptible price, including
  100 GB disk. Offers change; recheck before renting. Credential stored outside
  the repository and excluded from the GPU bundle.
- Local CUDA and bundle preflight passed. Remote smoke inference and throughput
  measurement remain. Current Torch wheels require CUDA 13 compatible hosts.

2026-10-12: User funded Vast and authorized the on-demand Google run. Rented one
RTX 3090 (instance 55604140, Norway) at $0.205185/hour including 100 GB storage;
network fees $0.001302/GB each way. Frozen run 9d984d64aa7e still contains 4,408
panoramas / 35,264 views. Local `typology-vast-watch.service` retrieves results
and checks final checksums and referenced assets before deleting this rental.
`typology-vast-hard-stop.timer` independently stops compute at 06:17 CEST Oct 13
if cleanup has not finished; a stopped rental retains billable disk for recovery.
Credentials and the dedicated SSH key remain outside the repository.
Google projection now advances in its loader thread and overlaps GPU inference;
two projected views matched the original pixels. Remote dependency pins include
OpenCV to preserve the fast projection backend. A 1/2/4-worker benchmark selects
concurrency from completed views; benchmark predictions are retained in the run.
Live state: `data/coruna/vast_run.json`; remote logs will be copied to
`data/coruna/vast_logs/`. Initial container setup and remote inference verification
were still pending when this note was written.

Vast launch follow-up: remote Python/Torch/CUDA checks and eight-view inference
passed. Local upload was only ~1–2 MB/s, so frozen inputs are fetched directly on
the host (roughly 3 panoramas/second), with exact SHA256 acceptance. The local
watcher uploads cached originals for checksum exceptions. Downloader threads have
lower CPU priority than inference. One/two/four-worker benchmarks completed 224
additional views and measured 0.862/1.247/1.298 views/second after warmup. Four
workers were selected; all benchmark views remain in the production output.
The full run launched at ~21:35 CEST with eight hours maximum inference and the
original nine-hour rental deadline. Projection/backend pins, configuration and
staged input checks passed remotely; retrieved sample outputs included all 13
attempted categories and valid crop/mask paths. A visual shop-sign crop check
matched the proposed category. Current estimate is ~7.5 hours of inference and
$1.70–$2 total for this frozen batch, subject to actual sustained throughput.
The watcher now also ends a run with no new completed views for 15 minutes.
Newer Google downloads continue locally and are outside this frozen inventory.

## 2026-10-12: map browsing, barrio filters and downloads

- User narrowed cleanup work to downloads and map navigation, adding barrio filters.
- Added 109 municipal Barrios / AA VV polygons from Concello da Coruña, with source
  attribution and a refresh script. Membership follows each record's displayed
  coordinates; new observations generally use camera positions.
- Map thumbnails follow the viewport, numbered points reveal nearby objects, and
  inspection keeps the map visible. Added source/area/barrio filters, reset, street
  basemap, URL state and sharing. ES/EN controls retained.
- CSV/GeoJSON/JSONL exports use the current selection and grouping mode, preserve
  retained viewpoints and provenance, and link existing images. A manifest records
  filters and snapshot metadata. Layer exports now include per-record run IDs and
  model revisions/prompts in their manifest.
- Implementation review only; no automated or browser tests requested or run.


## 2026-10-12: queued completion, backfill and duplicate review

- User authorized ongoing progress checks, Google completion/import, people/pets
  backfill, relevant deduplication, and pushing all project changes to GitHub.
- Installed five-minute follow-up timer. Apple completion is required before the
  one-worker people/pets backfill starts. Verified all 48,009 source files exist;
  none of those source results already attempted person/pet categories.
- Google publishing waits for the watcher's verified final copy and rental cleanup.
  Stopped rentals remain a recovery task. No new rental or batch was purchased.
- Started a separate CPU duplicate proposal pass for 4,500 sampled observations
  across shop signs, graffiti and street-name plates. Initial snapshot contains
  186,352 observations and 290,753 eligible pairs. Existing approved groups are
  unchanged; the service generates a review page after matching finishes.
- Existing 30 unittest cases and four new backfill/handoff cases passed. JavaScript
  and shell syntax checks and systemd unit verification passed. GPU backfill
  inference is queued, not yet smoke-tested on the GPU.


## 2026-10-13: data disk migration

- User authorized moving project data to `/mnt/data` and removing clearly redundant
  material. Destination: `/mnt/data/projects/typology-machine/data`, with the old
  repository `data` path retained as a symlink after verification.
- Removed 52 generated map snapshots, retaining the eight newest and the live
  manifest. Freed 26.0 GB decimal. Removed 4,408 completed rental staging image
  links only after confirming they were hard links to retained Google originals.
  Frozen inventories, provenance and rental logs remain.
- `typology-storage-migration.service` performs the copy, final writer pause and
  sync, inventory/read-back verification, path switch, HTTP health checks, and
  removal of the verified old copy. It preserves copies on failure. Progress:
  `~/.local/state/typology-machine/storage-migration.json` and `.log`.
- Refresh and follow-up timers are paused during the initial copy; backfill and
  OCR continue. The migration resumes timers and any paused writers after the
  switch. No original imagery, detections, masks, OCR, or reviewed groups deleted.


## 2026-10-13: R2 crops and webpage performance options

- User authorized R2 delivery for crops and requested options for other webpage
  improvements. Created `typology-crops` with custom domain
  `crops.typology.m19182.dev`; domain ownership and TLS are active.
- S3 credentials are stored outside the repository with owner-only permissions.
  A public sample matched local SHA-256; a 32-crop S3 smoke upload succeeded.
- Full resumable upload started under `typology-r2-crops.service`, with hourly
  follow-up. Browser activation waits for the complete inventory and public
  checks. Thumbnail/detail fallback was tested by aborting remote image requests
  in a headless browser; local images loaded without JavaScript errors.
- Map options proposed separately: reduce main-thread work with precomputed barrio
  membership and compact indexes; add viewport/paginated queries backed by SQLite
  spatial/text indexes; or switch map rendering to MapLibre with PMTiles and a
  separate search/detail mechanism. No map architecture change implemented yet.
