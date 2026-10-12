"""Build the static site: one tile per triangulated sign (its sharpest view), positions for the map, street and barrio, and tags when tags.jsonl exists.

Writes <out>/data.json, <out>/img/<sign>.jpg and copies site/index.html there.
Usage: uv run explore/export_site.py coruna store_sign --out data/coruna/site [--min-rays 2] [--size 256]
"""
import argparse
import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pillow_heif
from PIL import Image
from streetlevel import lookaround

pillow_heif.register_heif_opener()

ROOT = Path(__file__).resolve().parent.parent


def load_clusters(base):
    path = base / "cluster_metadata_dinov2.json"
    if not path.exists():
        return {}, None
    report = json.loads(path.read_text())
    assignments = {a["crop_id"]: a for a in report["assignments"]}
    return assignments, {k: report[k] for k in ("run_id", "embedding", "clusters", "coverage")}


def source_link(r, source):
    """where to see this view in the provider's own viewer"""
    if source == "gsv":
        x0, y0, x1, y1 = r["box"]
        W, H = r["view_size"]
        fy = 0.5 / math.tan(math.radians(r["fov"][1] / 2))
        pitch = r["view_pitch"] + math.degrees(math.atan((0.5 - (y0 + y1) / 2 / H) / fy))
        return f"https://www.google.com/maps/@?api=1&map_action=pano&pano={r['pano']}&heading={r['bearing']:.1f}&pitch={pitch:.1f}&fov=40"
    yaw = {0: 0, 1: 90, 2: 180, 3: -90}.get(r["face"], 0)
    return lookaround.build_permalink(lat=r["lat"], lon=r["lon"], heading=math.degrees(r["heading"]) + yaw, radians=False)


def thumb(job):
    src, dst, size, face, ctx, ctx_size = job
    try:
        im = Image.open(src).convert("RGB")
        im.thumbnail((size, size))
        im.save(dst, quality=80, optimize=True)
        if ctx and not ctx.exists() and face.exists():  # the whole source face, downscaled, for the detail toggle
            f = Image.open(face).convert("RGB")
            f.thumbnail((ctx_size, ctx_size))
            f.save(ctx, quality=75, optimize=True)
        return im.width, im.height
    except Exception as e:
        print("failed", src, repr(e)[:80])
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("slug", nargs="?", default="store_sign")
    ap.add_argument("--source", default="apple")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-rays", type=int, default=2)
    ap.add_argument("--min-score", type=float, default=0.6)
    ap.add_argument("--min-side", type=int, default=48)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ctx-size", type=int, default=1400, help="long side of the full-face context image; 0 to skip")
    a = ap.parse_args()
    base = Path("data") / a.city / a.source / a.slug
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "ctx").mkdir(exist_ok=True)

    rows = {r["id"]: r for r in map(json.loads, open(base / "instances.jsonl"))}
    groups = {g["sign"]: g for g in map(json.loads, open(base / "groups.jsonl"))}
    pos = [p for p in map(json.loads, open(base / "sign_positions.jsonl")) if p["n_rays"] >= a.min_rays]
    loc = {r["id"]: r for r in map(json.loads, open(base / "location.jsonl"))} if (base / "location.jsonl").exists() else {}
    tags = {t["id"]: t for t in map(json.loads, open(base / "tags.jsonl"))} if (base / "tags.jsonl").exists() else {}
    assignments, cluster_meta = load_clusters(base)

    picks = []
    for p in pos:
        r = rows[p["rep"]]
        if r["score"] < a.min_score:
            continue
        if min(r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]) < a.min_side:
            continue
        t = tags.get(r["id"])
        if t and t.get("is_sign") == "no":
            continue
        picks.append((p, r, t))
    picks.sort(key=lambda x: -x[1]["score"])
    print(len(picks), "signs of", len(pos), "triangulated")

    faces = Path("data") / a.city / a.source / "faces"
    jobs = [(base / r["crop"], out / "img" / f"{p['sign']}.jpg", a.size, faces / f"{r['pano']}_{r['face']}.heic",
             (out / "ctx" / f"{p['sign']}.jpg") if a.ctx_size else None, a.ctx_size) for p, r, _ in picks]
    with ProcessPoolExecutor(a.workers) as ex:
        sizes = list(ex.map(thumb, jobs, chunksize=32))

    data = []
    for (p, r, t), wh in zip(picks, sizes):
        if not wh:
            continue
        l = loc.get(r["id"], {})
        W, H = r["face_size"] if a.source == "apple" else r["view_size"]
        x0, y0, x1, y1 = r["box"]
        o = {"bx": [round(x0 / W, 4), round(y0 / H, 4), round(x1 / W, 4), round(y1 / H, 4)],
             "ctx": (out / "ctx" / f"{p['sign']}.jpg").exists(),"id": p["sign"], "s": r["score"], "d": r["date"][:7], "seen": len(groups[p["sign"]]["members"]),
             "la": round(p["lat"], 6), "lo": round(p["lon"], 6), "hm": p["height_m"], "w": wh[0], "h": wh[1],
             "st": l.get("street"), "ba": l.get("barrio"),
             "src": a.source[0], "u": source_link(r, a.source)}
        if t:
            o.update(k=t.get("kind"), b=t.get("business"), t=t.get("text"))
        assignment = assignments.get(r["id"])
        if assignment:
            o.update(cluster_id=assignment["cluster_id"], cluster_color=assignment["color"])
        data.append(o)
    meta = {"city": "A Coruña", "n": len(data), "groups": len(groups), "instances": len(rows), "dates": sorted({d["d"] for d in data}),
            "tagged": bool(tags)}
    if cluster_meta:
        meta["clustering"] = cluster_meta
        meta["clustering"]["mapped_signs"] = sum("cluster_id" in s for s in data)
    json.dump({"meta": meta, "signs": data}, open(out / "data.json", "w"), ensure_ascii=False, separators=(",", ":"))
    shutil.copy(ROOT / "site" / "index.html", out / "index.html")
    print(len(data), "signs ->", out, meta)


if __name__ == "__main__":
    main()
