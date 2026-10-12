"""The same sign through the years, by geometry alone. A sign has a map position and a height (from
triangulate.py). Every Google panorama near it, current or historical, is a known camera at a known place:
compute the bearing, pitch and angular size of the sign from that camera, cut that window out of the
panorama. No detector involved, so it works on 2008 imagery just as well as on 2024.

Writes data/<city>/history/<sign>/<date>_<panoid>.jpg and history.jsonl.
Usage: uv run explore/history_strip.py coruna store_sign --source apple [--sign s00123 ...] [--max-dist 25] [--limit N]
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import py360convert
from PIL import Image
from scipy.spatial import cKDTree

CAM_H = 2.5
OUT_HW = (600, 800)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("slug", nargs="?", default="store_sign")
    ap.add_argument("--source", default="apple")
    ap.add_argument("--sign", nargs="*")
    ap.add_argument("--max-dist", type=float, default=25)
    ap.add_argument("--min-rays", type=int, default=2)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    data = Path("data") / a.city
    base = data / a.source / a.slug
    out = data / "history"
    out.mkdir(exist_ok=True)

    signs = [json.loads(l) for l in open(base / "sign_positions.jsonl")]
    signs = [s for s in signs if s["n_rays"] >= a.min_rays]
    if a.sign:
        signs = [s for s in signs if s["sign"] in a.sign]
    if a.limit:
        signs = signs[: a.limit]
    rows = {r["id"]: r for r in map(json.loads, open(base / "instances.jsonl"))}
    panos = [json.loads(l) for l in open(data / "gsv" / "panos.jsonl")]
    panos = [p for p in panos if (data / "gsv" / "panos" / f"{p['id']}.jpg").exists() and p["heading"] is not None]
    lat0 = signs[0]["lat"] if signs else 43.36
    kx, ky = 111320 * math.cos(math.radians(lat0)), 111320
    pxy = np.array([(p["lon"] * kx, p["lat"] * ky) for p in panos])
    tree = cKDTree(pxy)
    print(len(signs), "signs,", len(panos), "google panos")

    log = open(out / "history.jsonl", "a")
    n_sign = n_img = 0
    for s in signs:
        rep = rows[s["rep"]]
        # angular width of the sign as seen from the representative view, times its distance = physical width
        x0, y0, x1, y1 = rep["box"]
        if a.source == "apple":
            fov_s = 1.178097245 if rep["face"] in (1, 3) else 2.159844949
            ang_w = (x1 - x0) / rep["face_size"][0] * fov_s
        else:
            ang_w = 2 * math.atan((x1 - x0) / rep["view_size"][0] * math.tan(math.radians(rep["fov"][0] / 2)))
        width_m = max(0.4, s["dist_m"] * ang_w)
        sxy = np.array([s["lon"] * kx, s["lat"] * ky])
        near = tree.query_ball_point(sxy, a.max_dist)
        if not near:
            continue
        by_date = defaultdict(list)
        for i in near:
            by_date[panos[i]["date"]].append(i)
        d_out = out / s["sign"]
        d_out.mkdir(exist_ok=True)
        wrote = 0
        for date, idxs in sorted(by_date.items()):
            i = min(idxs, key=lambda i: np.linalg.norm(pxy[i] - sxy))  # closest pano per date
            p = panos[i]
            v = sxy - pxy[i]
            dist = float(np.linalg.norm(v))
            if dist < 3:
                continue
            bearing = math.degrees(math.atan2(v[0], v[1])) % 360
            pitch = math.degrees(math.atan2(s["height_m"] - CAM_H, dist))
            fov_h = max(18.0, min(90.0, 3 * math.degrees(2 * math.atan(width_m / 2 / dist))))
            fov_v = fov_h * OUT_HW[0] / OUT_HW[1]
            dst = d_out / f"{date}_{p['id']}.jpg"
            if not dst.exists():
                e = np.asarray(Image.open(data / "gsv" / "panos" / f"{p['id']}.jpg").convert("RGB"))
                u = (bearing - p["heading"] + 540) % 360 - 180  # yaw relative to the pano's own heading
                view = py360convert.e2p(e, fov_deg=(fov_h, fov_v), u_deg=u, v_deg=pitch, out_hw=OUT_HW, mode="bilinear")
                Image.fromarray(view).save(dst, quality=88)
            log.write(json.dumps({"sign": s["sign"], "date": date, "pano": p["id"], "dist_m": round(dist, 1), "bearing": round(bearing, 1),
                                  "pitch": round(pitch, 1), "fov": round(fov_h, 1), "file": str(dst.relative_to(data))}) + "\n")
            wrote += 1
        if wrote:
            n_sign += 1
            n_img += wrote
    print(f"{n_sign} signs, {n_img} dated views -> {out}")


if __name__ == "__main__":
    main()
