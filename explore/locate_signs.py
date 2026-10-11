"""Give every sign a street and a barrio, from OpenStreetMap: nearest named highway segment to the pano
position (within --max-dist metres) and nearest place=neighbourhood/suburb node (OSM has points, not
polygons, for Coruña's barrios). Writes location.jsonl next to instances.jsonl.

Inputs: data/<city>/osm_streets.json and osm_places.json from Overpass (way[highway][name] out geom;
node[place~neighbourhood|suburb|quarter] out).
Usage: uv run explore/locate_signs.py coruna store_sign
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("slug", nargs="?", default="store_sign")
    ap.add_argument("--max-dist", type=float, default=30)
    a = ap.parse_args()
    data = Path("data") / a.city
    base = data / "apple" / a.slug
    lat0 = 43.36

    def xy(lat, lon):
        return (lon * 111320 * math.cos(math.radians(lat0)), lat * 111320)

    # streets: densify each way into points every ~4 m, remember the name per point
    pts, names = [], []
    for w in json.load(open(data / "osm_streets.json"))["elements"]:
        if w["type"] != "way":
            continue
        name = w["tags"]["name"]
        g = [xy(p["lat"], p["lon"]) for p in w["geometry"]]
        for (x0, y0), (x1, y1) in zip(g, g[1:]):
            n = max(1, int(math.hypot(x1 - x0, y1 - y0) / 4))
            for t in range(n + 1):
                pts.append((x0 + (x1 - x0) * t / n, y0 + (y1 - y0) * t / n))
                names.append(name)
    street_tree = cKDTree(np.array(pts))

    places = [(e["tags"]["name"], e["tags"].get("place"), xy(e.get("lat") or e["center"]["lat"], e.get("lon") or e["center"]["lon"]))
              for e in json.load(open(data / "osm_places.json"))["elements"]
              if e["tags"].get("place") in ("neighbourhood", "suburb", "quarter") and e["tags"].get("name")]
    place_tree = cKDTree(np.array([p[2] for p in places]))

    out = base / "location.jsonl"
    c_street, c_barrio = Counter(), Counter()
    with open(out, "w") as f:
        for r in map(json.loads, open(base / "instances.jsonl")):
            p = xy(r["lat"], r["lon"])
            d, i = street_tree.query(p)
            street = names[i] if d <= a.max_dist else None
            _, j = place_tree.query(p)
            barrio = places[j][0]
            f.write(json.dumps({"id": r["id"], "street": street, "street_dist": round(float(d), 1), "barrio": barrio}, ensure_ascii=False) + "\n")
            c_street[street] += 1
            c_barrio[barrio] += 1
    print("->", out)
    print("barrios:", c_barrio.most_common(12))
    print("streets:", c_street.most_common(15))


if __name__ == "__main__":
    main()
