"""Match Apple signs (triangulated positions) to Google detections of any date by geometry: a Google
detection is a ray from its pano through the sign; if the ray passes within a metre or so of an Apple sign
position, at a compatible height and angular size, it is the same sign seen on that date. No embedding
needed across sources and years; Google's own detector framing is kept.

Writes data/<city>/gsv/<slug>/matches.jsonl (one row per matched Google detection: gsv id, sign, perp_m,
t_m, dh_m) and data/<city>/history/index.jsonl (one row per sign: every dated view, Apple and Google).
Usage: uv run explore/match_sources.py coruna store_sign [--min-rays 2] [--max-t 35]
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from triangulate import CAM_H, ray  # noqa: E402


def ang_width(r, source):
    x0, y0, x1, y1 = r["box"]
    if source == "apple":
        fov_s = 1.178097245 if r["face"] in (1, 3) else 2.159844949
        return (x1 - x0) / r["face_size"][0] * fov_s
    return 2 * math.atan((x1 - x0) / r["view_size"][0] * math.tan(math.radians(r["fov"][0] / 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("slug", nargs="?", default="store_sign")
    ap.add_argument("--min-rays", type=int, default=2)
    ap.add_argument("--max-t", type=float, default=28)
    ap.add_argument("--facing", type=float, default=75, help="max angle (deg) between the Google ray and the mean Apple ray")
    ap.add_argument("--perp", type=float, default=0.05, help="max ray-to-sign distance as a fraction of range (floor 1 m)")
    ap.add_argument("--dh", type=float, default=1.5, help="max height disagreement (m)")
    a = ap.parse_args()
    data = Path("data") / a.city
    abase, gbase = data / "apple" / a.slug, data / "gsv" / a.slug

    signs = [s for s in map(json.loads, open(abase / "sign_positions.jsonl")) if s["n_rays"] >= a.min_rays]
    arows = {r["id"]: r for r in map(json.loads, open(abase / "instances.jsonl"))}
    members = {g["sign"]: g["members"] for g in map(json.loads, open(abase / "groups.jsonl"))}
    for s in signs:
        s["width_m"] = max(0.3, s["dist_m"] * ang_width(arows[s["rep"]], "apple"))
        # which way the sign faces: the mean direction of the Apple rays that saw it. A Google ray coming from
        # the other side of the wall cannot be looking at this sign.
        ds = [ray(arows[m], "apple")[0] for m in members[s["sign"]] if m in arows]
        f = np.array([sum(math.sin(b) for b in ds), sum(math.cos(b) for b in ds)])
        s["facing"] = f / (np.linalg.norm(f) + 1e-9)
    lat0 = signs[0]["lat"]
    kx, ky = 111320 * math.cos(math.radians(lat0)), 111320
    sxy = np.array([(s["lon"] * kx, s["lat"] * ky) for s in signs])
    tree = cKDTree(sxy)
    print(len(signs), "apple signs with >=", a.min_rays, "rays")

    grows = [r for r in map(json.loads, open(gbase / "instances.jsonl")) if r["heading"] is not None]
    print(len(grows), "google detections")
    matches = []
    for r in grows:
        p = np.array([r["lon"] * kx, r["lat"] * ky])
        b, e = ray(r, "gsv")
        d = np.array([math.sin(b), math.cos(b)])
        aw = ang_width(r, "gsv")
        best = None
        for i in tree.query_ball_point(p, a.max_t):
            v = sxy[i] - p
            t = float(v @ d)
            if t < 2 or t > a.max_t:
                continue
            perp = float(np.linalg.norm(v - t * d))
            if perp > max(1.0, a.perp * t):
                continue
            if float(d @ signs[i]["facing"]) < math.cos(math.radians(a.facing)):
                continue
            dh = CAM_H["gsv"] + math.tan(e) * t - signs[i]["height_m"]
            if abs(dh) > a.dh:
                continue
            wr = (t * aw) / signs[i]["width_m"]
            if wr < 0.4 or wr > 2.5:
                continue
            key = perp / t + abs(dh) / 10
            if best is None or key < best[0]:
                best = (key, i, perp, t, dh)
        if best:
            _, i, perp, t, dh = best
            matches.append({"gsv": r["id"], "sign": signs[i]["sign"], "perp_m": round(perp, 2), "t_m": round(t, 1), "dh_m": round(dh, 2)})
    with open(gbase / "matches.jsonl", "w") as f:
        for m in matches:
            f.write(json.dumps(m) + "\n")
    print(f"{len(matches)} google detections matched to {len({m['sign'] for m in matches})} apple signs -> {gbase / 'matches.jsonl'}")

    # one history row per sign: all Apple views (with dates) plus the best Google view per date
    gid = {r["id"]: r for r in grows}
    by_sign = defaultdict(lambda: defaultdict(list))
    for m in matches:
        r = gid[m["gsv"]]
        by_sign[m["sign"]][r["date"]].append((m, r))
    hist = data / "history"
    hist.mkdir(exist_ok=True)
    n = 0
    with open(hist / "index.jsonl", "w") as f:
        for s in signs:
            views = []
            for mid in members[s["sign"]]:
                r = arows[mid]
                views.append({"source": "apple", "date": r["date"][:7], "id": mid, "pano": r["pano"], "crop": f"apple/{a.slug}/{r['crop']}",
                              "is_rep": mid == s["rep"], "score": r["score"]})
            for date, cands in by_sign[s["sign"]].items():
                def k(c):
                    m, r = c
                    return m["perp_m"] / m["t_m"] + 0.02 * max(0, m["t_m"] - 20) + 0.05 * max(0, 5 - m["t_m"]) - 0.1 * r["score"]
                cands.sort(key=k)
                for j, (m, r) in enumerate(cands):
                    views.append({"source": "gsv", "date": date, "id": m["gsv"], "pano": r["pano"], "crop": f"gsv/{a.slug}/{r['crop']}",
                                  "is_rep": j == 0, "score": r["score"], "t_m": m["t_m"], "perp_m": m["perp_m"]})
            views.sort(key=lambda v: (v["date"] or "", v["source"]))
            dates = sorted({v["date"] for v in views if v["date"]})
            f.write(json.dumps({"sign": s["sign"], "lat": s["lat"], "lon": s["lon"], "height_m": s["height_m"], "width_m": round(s["width_m"], 2),
                                "n_dates": len(dates), "first": dates[0] if dates else None, "last": dates[-1] if dates else None,
                                "facing": round(math.degrees(math.atan2(*s["facing"])) % 360),
                                "views": views}) + "\n")
            n += 1
    print(n, "signs ->", hist / "index.jsonl")


if __name__ == "__main__":
    main()
