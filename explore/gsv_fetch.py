"""Fetch Google Street View panoramas for a city area: current pass plus, with --history, every earlier capture
at the same spot. No key: coverage tiles give ids (gsv.csv from gsv_index), metadata by id gives date, heading
and the historical list, tiles give the image.

Writes data/<city>/gsv/panos/<id>.jpg (equirectangular, zoom 4 = 8192x4096) and data/<city>/gsv/panos.jsonl
(one row per downloaded pano: id, lat, lon, heading, date, spot (the current pano id it belongs to), is_current).
Resumable.

Usage: uv run explore/gsv_fetch.py coruna [--bbox W,S,E,N] [--thin 6] [--zoom 4] [--history] [--workers 8] [--limit N]
"""
import argparse
import csv
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from streetlevel import streetview as sv


class TimeoutSession(requests.Session):
    def get(self, *a, **kw):
        kw.setdefault("timeout", 30)
        return super().get(*a, **kw)


def thin(rows, d):
    seen, out = set(), []
    for r in rows:
        k = (round(float(r["lat"]) * 111320 / d), round(float(r["lon"]) * 111320 * math.cos(math.radians(float(r["lat"]))) / d))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("--bbox")
    ap.add_argument("--thin", type=float, default=6)
    ap.add_argument("--zoom", type=int, default=4)
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    data = Path("data") / a.city
    out = data / "gsv"
    (out / "panos").mkdir(parents=True, exist_ok=True)
    log_path = out / "panos.jsonl"
    done = {json.loads(l)["id"] for l in open(log_path)} if log_path.exists() else set()

    rows = list(csv.DictReader(open(data / "gsv.csv")))
    if a.bbox:
        W, S, E, N = map(float, a.bbox.split(","))
        rows = [r for r in rows if W <= float(r["lon"]) <= E and S <= float(r["lat"]) <= N]
    rows = thin(rows, a.thin)
    if a.limit:
        rows = rows[: a.limit]
    print(f"{len(rows)} current panos selected, {len(done)} already fetched", flush=True)

    lock = threading.Lock()
    local = threading.local()

    def session():
        s = getattr(local, "s", None)
        if s is None:
            s = local.s = TimeoutSession()
        return s

    def fetch_one(pano, spot, is_current):
        """pano: StreetViewPanorama with image_sizes (from find_panorama_by_id)."""
        dst = out / "panos" / f"{pano.id}.jpg"
        if pano.id in done:
            return 0
        for _ in range(3):
            try:
                if not dst.exists():
                    im = sv.get_panorama(pano, zoom=a.zoom, session=session())
                    im.save(dst, quality=88)
                rec = {"id": pano.id, "lat": pano.lat, "lon": pano.lon, "heading": round(math.degrees(pano.heading), 3) if pano.heading is not None else None,
                       "date": str(pano.date)[:7] if pano.date else None, "spot": spot, "is_current": is_current,
                       "size": list(pano.image_sizes[-1]) if pano.image_sizes else None}
                with lock:
                    open(log_path, "a").write(json.dumps(rec) + "\n")
                    done.add(pano.id)
                return 1
            except Exception as e:
                err = e
                time.sleep(2)
        print("failed", pano.id, repr(err)[:120], flush=True)
        return 0

    def spot(row):
        n = 0
        for _ in range(3):
            try:
                p = sv.find_panorama_by_id(row["id"], download_depth=False, session=session())
                break
            except Exception:
                time.sleep(2)
                p = None
        if p is None:
            print("meta failed", row["id"], flush=True)
            return 0
        n += fetch_one(p, p.id, True)
        if a.history:
            for h in p.historical or []:
                if h.id in done:
                    continue
                try:
                    hp = sv.find_panorama_by_id(h.id, download_depth=False, session=session())
                except Exception:
                    hp = None
                if hp is not None:
                    n += fetch_one(hp, p.id, False)
        return n

    t0 = time.time()
    total = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, n in enumerate(ex.map(spot, rows), 1):
            total += n
            if i % 100 == 0:
                print(f"{i}/{len(rows)} spots, {total} panos, {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"done: {total} panos fetched in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
