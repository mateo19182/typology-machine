"""The same sign through the years. Reads history/index.jsonl (from match_sources.py): for every sign with
enough dated views, cut a context window around each dated detection, framed the same way every time
(sign in the centre, window about three sign-widths wide, 4:3). Google views come out of the equirect
pano at the detection's bearing and pitch; Apple views are a padded crop of the original face.

Dates with a Google pano near the sign but no detection matched get a geometric window at the sign's
computed bearing, flagged detected=false: those are the "before the sign existed" views.

Writes data/<city>/history/<sign>/<date>_<source>_<id>.jpg and history/strips.jsonl.
Usage: uv run explore/history_strip.py coruna store_sign [--min-dates 4] [--sign s00123 ...] [--size 640] [--workers 6]
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pillow_heif
import py360convert
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from match_sources import ang_width  # noqa: E402
from triangulate import ray  # noqa: E402

pillow_heif.register_heif_opener()
CAM_H = 2.5
WIDEN = 3.2  # window width in sign widths


def plan_view(v, r, s, data):
    """what to cut for one dated view -> dict for the worker"""
    aw = math.degrees(ang_width(r, v["source"]))
    fov_h = max(22.0, min(80.0, WIDEN * aw))
    if v["source"] == "gsv":
        x0, y0, x1, y1 = r["box"]
        W, H = r["view_size"]
        fy = 0.5 / math.tan(math.radians(r["fov"][1] / 2))
        pitch = r["view_pitch"] + math.degrees(math.atan((0.5 - (y0 + y1) / 2 / H) / fy))
        b, _ = ray(r, "gsv")
        u = (math.degrees(b) - r["heading"] + 540) % 360 - 180
        return {"kind": "equirect", "src": str(data / "gsv" / "panos" / f"{r['pano']}.jpg"), "u": u, "v": pitch, "fov": fov_h}
    x0, y0, x1, y1 = r["box"]
    W, H = r["face_size"]
    fov_s = 1.178097245 if r["face"] in (1, 3) else 2.159844949
    half = fov_h / 2 / math.degrees(fov_s) * W  # px
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return {"kind": "face", "src": str(data / "apple" / "faces" / f"{r['pano']}_{r['face']}.heic"),
            "box": [cx - half, cy - half * 0.75, cx + half, cy + half * 0.75]}


def cut(job):
    size = job["size"]
    out_hw = (size * 3 // 4, size)
    try:
        if job["kind"] == "equirect":
            e = np.asarray(Image.open(job["src"]).convert("RGB"))
            fov_v = job["fov"] * 0.75
            im = Image.fromarray(py360convert.e2p(e, fov_deg=(job["fov"], fov_v), u_deg=job["u"], v_deg=job["v"], out_hw=out_hw, mode="bilinear"))
        else:
            im = Image.open(job["src"]).convert("RGB")
            x0, y0, x1, y1 = job["box"]
            w, h = min(x1 - x0, im.width), min(y1 - y0, im.height)
            x0, y0 = min(max(0, x0), im.width - w), min(max(0, y0), im.height - h)  # slide the window inside the face
            im = im.crop((int(x0), int(y0), int(x0 + w), int(y0 + h))).resize((size, size * 3 // 4), Image.LANCZOS)
        im.save(job["dst"], quality=82)
        return True
    except Exception as e:
        print("failed", job["dst"], repr(e)[:100], flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("slug", nargs="?", default="store_sign")
    ap.add_argument("--sign", nargs="*")
    ap.add_argument("--min-dates", type=int, default=4, help="google dates needed for a sign to get a strip")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    data = Path("data") / a.city
    hist = data / "history"
    arows = {r["id"]: r for r in map(json.loads, open(data / "apple" / a.slug / "instances.jsonl"))}
    grows = {r["id"]: r for r in map(json.loads, open(data / "gsv" / a.slug / "instances.jsonl"))}
    index = [json.loads(l) for l in open(hist / "index.jsonl")]
    if a.sign:
        index = [s for s in index if s["sign"] in a.sign]
    index = [s for s in index if len({v["date"] for v in s["views"] if v["source"] == "gsv"}) >= a.min_dates]
    if a.limit:
        index = index[: a.limit]
    # panos near each sign for the undetected dates
    panos = [p for p in map(json.loads, open(data / "gsv" / "panos.jsonl")) if p["heading"] is not None and (data / "gsv" / "panos" / f"{p['id']}.jpg").exists()]
    lat0 = index[0]["lat"] if index else 43.36
    kx, ky = 111320 * math.cos(math.radians(lat0)), 111320
    pxy = np.array([(p["lon"] * kx, p["lat"] * ky) for p in panos])
    tree = cKDTree(pxy)
    print(len(index), "signs with >=", a.min_dates, "google dates")

    jobs, strips = [], []
    for s in index:
        d_out = hist / s["sign"]
        d_out.mkdir(parents=True, exist_ok=True)
        frames = []
        seen_dates = set()
        for v in s["views"]:
            if not v["is_rep"]:
                continue
            r = (grows if v["source"] == "gsv" else arows).get(v["id"])
            if r is None:
                continue
            job = plan_view(v, r, s, data)
            dst = d_out / f"{v['date']}_{v['source']}_{v['id']}.jpg"
            job.update(dst=str(dst), size=a.size)
            if not dst.exists():
                jobs.append(job)
            frames.append({"date": v["date"], "source": v["source"], "id": v["id"], "pano": v["pano"], "detected": True,
                           "file": str(dst.relative_to(data)), "crop": v["crop"]})
            seen_dates.add(v["date"])
        # geometric fallback for nearby dated panos with no matched detection, on the facing side
        sxy = np.array([s["lon"] * kx, s["lat"] * ky])
        fac = math.radians(s["facing"])
        fvec = np.array([math.sin(fac), math.cos(fac)])
        by_date = defaultdict(list)
        for i in tree.query_ball_point(sxy, 22):
            v = sxy - pxy[i]
            dist = float(np.linalg.norm(v))
            if dist < 4 or (v / dist) @ fvec < math.cos(math.radians(60)):
                continue
            by_date[panos[i]["date"]].append((dist, i))
        for date, cands in by_date.items():
            if date in seen_dates or not date:
                continue
            dist, i = min(cands)
            p = panos[i]
            v = sxy - pxy[i]
            bearing = math.degrees(math.atan2(v[0], v[1])) % 360
            pitch = math.degrees(math.atan2(s["height_m"] - CAM_H, dist))
            fov_h = max(22.0, min(80.0, WIDEN * math.degrees(2 * math.atan(s["width_m"] / 2 / dist))))
            dst = d_out / f"{date}_gsv_{p['id']}_geo.jpg"
            job = {"kind": "equirect", "src": str(data / "gsv" / "panos" / f"{p['id']}.jpg"), "u": (bearing - p["heading"] + 540) % 360 - 180,
                   "v": pitch, "fov": fov_h, "dst": str(dst), "size": a.size}
            if not dst.exists():
                jobs.append(job)
            frames.append({"date": date, "source": "gsv", "id": None, "pano": p["id"], "detected": False, "file": str(dst.relative_to(data)), "crop": None})
        frames.sort(key=lambda f: (f["date"], f["source"]))
        strips.append({"sign": s["sign"], "lat": s["lat"], "lon": s["lon"], "facing": s["facing"], "n_dates": len({f["date"] for f in frames}),
                       "n_detected": sum(f["detected"] for f in frames), "frames": frames})
    print(len(jobs), "windows to cut")
    # decode each pano once per worker call order: sort jobs by source so neighbours share files in page cache
    jobs.sort(key=lambda j: j["src"])
    with ProcessPoolExecutor(a.workers) as ex:
        for k, ok in enumerate(ex.map(cut, jobs, chunksize=4), 1):
            if k % 200 == 0:
                print(f"{k}/{len(jobs)}", flush=True)
    with open(hist / "strips.jsonl", "w") as f:
        for s in strips:
            f.write(json.dumps(s) + "\n")
    print(f"{len(strips)} signs, {sum(len(s['frames']) for s in strips)} frames -> {hist / 'strips.jsonl'}")


if __name__ == "__main__":
    main()
