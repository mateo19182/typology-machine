"""Index Apple Look Around panoramas in a city bbox (no API key; uses streetlevel's reverse-engineered client).

Writes data/<city>/lookaround.csv with one row per pano: id, build_id, lat, lon, date, heading.
Note: Apple only serves the latest pass per street, so dates are a patchwork, not a time machine.
Usage: uv run explore/lookaround_coverage.py [city]
"""
import csv
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mercantile
from streetlevel import lookaround

from coverage import CITIES

city = sys.argv[1] if len(sys.argv) > 1 else "coruna"
W, S, E, N = CITIES[city]
tiles = list(mercantile.tiles(W, S, E, N, 17))


def fetch(t):
    for _ in range(3):
        try:
            return lookaround.get_coverage_tile(t.x, t.y).panos
        except Exception:
            time.sleep(1)
    print("failed tile", t, file=sys.stderr)
    return []


with ThreadPoolExecutor(8) as ex:
    panos = [p for r in ex.map(fetch, tiles) for p in r]

out = Path("data") / city / "lookaround.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "build_id", "lat", "lon", "date", "heading", "coverage_type"])
    for p in panos:
        w.writerow([p.id, p.build_id, p.lat, p.lon, p.date.isoformat(), round(p.heading, 4), p.coverage_type.name])
print(f"{len(panos)} panos from {len(tiles)} tiles -> {out}")
print(sorted(Counter(p.date.strftime("%Y-%m") for p in panos).items()))
