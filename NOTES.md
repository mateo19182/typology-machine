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
| Google Street View | yes, with history back to ~2009 | unofficial endpoints only | `streetlevel` and `streetview` both fail everywhere right now (Google changed something). Re-check later; it is the only real time machine. |
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
  access. Comparison with Mapillary's detections: `data/coruna/sam3_vs_mapillary.html`.
- Moondream 3 (`moondream/moondream3-preview`, ungated) still to try, for checking crops and naming clusters.
- The GPU has another process using 16 GB (`uvicorn app:app` on port 8000). 8 GB is enough for SAM 3.

### Open threads
- Which object first? Store signs have the most variety; street lights and trash cans are more Becher.
  Apple's quality might make small things (manholes, doorbells, door handles) viable that Mapillary can't do.
- Dedupe: GPS clustering plus visual embedding (DINOv2) seems the obvious move.
- Time: Mapillary 2019 dashcam vs Apple 2026 for the same streets? Low quality on the 2019 side.
  Google history would be far better if an endpoint comes back.
- Presentation: web grid, print, or something playful (see Yufeng Zhao in PRIOR_ART.md).
