"""Pull Mapillary coverage (image points + detected map features) for a bbox from vector tiles.

Writes data/<city>/images.csv and data/<city>/features.csv and prints a summary.
Usage: uv run explore/coverage.py [city]
"""
import csv
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import mapbox_vector_tile as mvt
import mercantile
import requests
from dotenv import load_dotenv

CITIES = {
    # west, south, east, north
    "coruna": (-8.455, 43.335, -8.375, 43.390),
}
Z = 14
TILES = {
    "images": "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}",
    "features": "https://tiles.mapillary.com/maps/vtp/mly_map_feature_point/2/{z}/{x}/{y}",
}

load_dotenv()
TOKEN = os.environ["MAPILLARY_TOKEN"]


def fetch(kind, tile):
    url = TILES[kind].format(z=tile.z, x=tile.x, y=tile.y)
    r = requests.get(url, params={"access_token": TOKEN}, timeout=60)
    r.raise_for_status()
    return mvt.decode(r.content) if r.content else {}


def lonlat(tile, geom, extent):
    # vector tile coords are 0..extent with y flipped by the decoder (origin bottom-left)
    x, y = geom["coordinates"]
    b = mercantile.bounds(tile)
    return b.west + (b.east - b.west) * x / extent, b.south + (b.north - b.south) * y / extent


def in_bbox(lon, lat, bbox):
    w, s, e, n = bbox
    return w <= lon <= e and s <= lat <= n


def main(city):
    bbox = CITIES[city]
    out = Path("data") / city
    out.mkdir(parents=True, exist_ok=True)
    tiles = list(mercantile.tiles(*bbox, zooms=Z))
    images, features = [], []
    for t in tiles:
        layer = fetch("images", t).get("image")
        if layer:
            for f in layer["features"]:
                lon, lat = lonlat(t, f["geometry"], layer["extent"])
                if in_bbox(lon, lat, bbox):
                    images.append({**f["properties"], "lon": lon, "lat": lat})
        layer = fetch("features", t).get("point")
        if layer:
            for f in layer["features"]:
                lon, lat = lonlat(t, f["geometry"], layer["extent"])
                if in_bbox(lon, lat, bbox):
                    features.append({**f["properties"], "lon": lon, "lat": lat})
        print(f"tile {t.x},{t.y}: {len(images)} images, {len(features)} features so far", file=sys.stderr)

    for name, rows in (("images", images), ("features", features)):
        keys = sorted({k for r in rows for k in r})
        with open(out / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    year = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).year
    print(f"\n== {city}: {len(images)} images in {len(tiles)} tiles")
    print("panos:", sum(bool(r.get("is_pano")) for r in images))
    print("sequences:", len({r.get("sequence_id") for r in images}))
    print("creators:", len({r.get("creator_id") for r in images}))
    print("by year:", sorted(Counter(year(r["captured_at"]) for r in images).items()))
    qs = sorted(r.get("quality_score", 0) for r in images)
    print("quality_score p25/p50/p75:", [round(qs[int(len(qs) * p)], 2) for p in (0.25, 0.5, 0.75)])
    print(f"\n== {len(features)} map features; top classes:")
    for v, n in Counter(r.get("value") for r in features).most_common(40):
        print(f"{n:6d}  {v}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "coruna")
