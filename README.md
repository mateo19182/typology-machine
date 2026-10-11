# Typology Machine

Pick a city and an object type (doors, manhole covers, shop signs, galerías…). Find every instance in
street-level imagery, straighten it to a frontal view, remove duplicates, and lay the results out as a
Becher-style typology grid. Then cluster them to find the city's sub-types.

## Status
Exploring the data. No models run yet.

## Data
Mapillary (free, needs a token). Put `MAPILLARY_TOKEN='MLY|...'` in `.env` (quoted: the token contains `|`).
Everything under `data/` is gitignored.

## Scripts
- `uv run explore/coverage.py coruna` pulls image points and Mapillary's own detected objects from the
  z14 vector tiles, writing `data/coruna/{images,features}.csv` plus a summary.
- `uv run explore/contact_sheet.py coruna 200` builds `data/coruna/contact_sheet.html`, a random sample
  stratified by year and pano.

## Findings: A Coruña (2026-10-11)
- 86k images, 7.5k panos, only 15 contributors. 73% are from 2019 (a dashcam pass). 2025–26 adds about
  10k, mostly car-roof GoPro panos.
- Mostly forward-facing car shots on arterial roads. Old town and Mariña have a few hundred images each,
  mostly from 2016–2019 phones.
- Avenida da Mariña sample: tunnels, night shots and the road; the galerías are barely in frame. Upper
  floors need panos or upward-angled shots, so galerías look weak with this data.
- 21.6k Mapillary detections. Best typology candidates by count: store signs (2.5k), ad signs (2.1k),
  street lights (3.8k), trash cans (1k), manholes (943), benches (305).

## Model plan
SAM 3 for exhaustive text-prompted masks; Moondream 3.1 to check crops and name clusters. Compare both on
about 50 hand-checked images before picking.
