# Data sources and scope

The MIT license covers this project's code. It does not grant rights to external
imagery, datasets, map tiles, model weights, or dependencies.

- Street imagery comes from Apple Look Around, Google Street View, and Mapillary.
  Downloads, extracted crops, detections, OCR, and review results stay under the
  ignored `data/` directory. None of that imagery is included in the repository.
- `site/barrios.geojson` contains municipal barrio polygons from Concello da
  Coruña / Ayuntamiento da Coruña. Its `metadata` preserves the source service,
  attribution, and retrieval date. These external boundaries are not covered by
  the project's MIT license. `explore/fetch_barrios.py` reproduces the download.
- The viewer loads basemap tiles from external providers and displays their
  attribution. The repository does not contain those tiles.
- Detection and embedding experiments use separately obtained models, including
  SAM 3 and DINO models. Model weights are not included. Obtain model access and
  follow the terms supplied with each model and data source.

Publishing the source code does not publish a reusable city image dataset. The
hosted viewer is a separate deployment; visibility there does not establish
permission to redistribute its underlying images.

Object labels and OCR can be wrong. Locations often represent the camera rather
than the exact object. Duplicate groups are proposals until reviewed. People and
pets are detection categories; the current deduplication workflow excludes them
from identity matching.

## Viewer quality filters

Map, wall, object details and downloads use a minimum detection confidence of 0.65 by default. The filter offers 0.50, 0.65, 0.75 and 0.85 and travels with shared links. Detector scores are not calibrated category accuracy.

The query index excludes proposals with a short side below 24 pixels or area below 1,024 pixels. It uses object dimensions when available, otherwise the exported crop dimensions. Within the same source image and category, boxes with intersection-over-union of at least 0.65 are suppressed in descending confidence order; larger area breaks confidence ties. Different categories and viewpoints are preserved. Original detections and images remain untouched; selecting 0.50 restores lower-confidence candidates, but not size/overlap exclusions. No blur classifier or category verification is applied.
