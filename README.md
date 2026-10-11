# Typology Machine

Find every instance of some object type in a city's street-level photos and lay them out as a Becher-style
typology grid. Dedupe, cluster into sub-types, maybe watch them change over the years. A Coruña first.

This is an exploration, not a spec. Scripts under `explore/` are throwaway probes; `NOTES.md` has the
running log of what we know and what's open. `IDEAS.md` and `PRIOR_ART.md` are reference.

## Setup
`uv sync`. Put `MAPILLARY_TOKEN='MLY|...'` in `.env` (quoted: the token contains `|`). `data/` is gitignored.

## Scripts
- `explore/coverage.py` pulls Mapillary image points and its detected objects for a city bbox.
- `explore/contact_sheet.py` random sample of Mapillary images, stratified by year and pano.
- `explore/crop_features.py` crops every Mapillary-detected object from the best image that sees it.
- `explore/typology_grid.py` local HTML viewer, one grid per class, from the crops.
- `explore/lookaround_coverage.py` indexes Apple Look Around panoramas (no key needed).
- `explore/sam3_vs_mapillary.py` side-by-side of Mapillary's detections and SAM 3 text prompts.

All take the city name (`coruna`) and write under `data/<city>/`.
