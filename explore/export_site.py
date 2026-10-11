"""Export a small, deduped, OCR'd selection of sign crops for the web frontend.

Writes <out>/data.json (one entry per sign, mask inlined as a tiny base64 PNG) and <out>/img/<id>.jpg.
Usage: uv run explore/export_site.py coruna store_sign --out /path/site --n 240
"""
import argparse
import base64
import json
import random
from io import BytesIO
from pathlib import Path

from PIL import Image
from streetlevel import lookaround


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("slug", nargs="?", default="store_sign")
    ap.add_argument("--source", default="apple", help="apple or gsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--labels", default="shop,plate")
    ap.add_argument("--min-score", type=float, default=0.6)
    ap.add_argument("--min-side", type=int, default=56)
    ap.add_argument("--size", type=int, default=360)
    a = ap.parse_args()
    base = Path("data") / a.city / a.source / a.slug
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)

    rows = {r["id"]: r for r in map(json.loads, open(base / "instances.jsonl"))}
    labels = {o["id"]: o["label"] for o in map(json.loads, open(base / "labels.jsonl"))}
    ocr = {o["id"]: o["text"] for o in map(json.loads, open(base / "ocr.jsonl"))} if (base / "ocr.jsonl").exists() else {}
    groups = [json.loads(l) for l in open(base / "groups.jsonl")]
    n_groups = len(groups)

    picks = []
    for g in groups:
        r = rows[g["rep"]]
        if labels.get(r["id"]) not in a.labels.split(",") or r["score"] < a.min_score:
            continue
        w, h = r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]
        if min(w, h) < a.min_side:
            continue
        picks.append((r, len(g["members"])))
    picks.sort(key=lambda p: -p[0]["score"])
    picks = picks[: a.n]
    random.Random(7).shuffle(picks)

    data = []
    for r, seen in picks:
        im = Image.open(base / r["crop"]).convert("RGB")
        im.thumbnail((a.size, a.size))
        im.save(out / "img" / f"{r['id']}.jpg", quality=82, optimize=True)
        m = Image.open(base / r["mask"]).convert("L")
        m.thumbnail((96, 96))
        m = m.point(lambda v: 255 if v > 127 else 0).convert("1")
        buf = BytesIO()
        m.save(buf, "PNG", optimize=True)
        yaw = {0: 0, 1: 90, 2: 180, 3: -90}.get(r["face"], 0)
        data.append({
            "id": r["id"], "t": ocr.get(r["id"], ""), "s": r["score"], "d": r["date"], "seen": seen,
            "la": round(r["lat"], 6), "lo": round(r["lon"], 6), "w": im.width, "h": im.height,
            "u": lookaround.build_permalink(lat=r["lat"], lon=r["lon"], heading=r["heading"] + yaw, radians=False),
            "m": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        })
    meta = {"city": "A Coruña", "prompt": a.slug.replace("_", " "), "n": len(data), "groups": n_groups,
            "instances": len(rows), "dates": sorted({d["d"][:7] for d in data})}
    json.dump({"meta": meta, "signs": data}, open(out / "data.json", "w"), ensure_ascii=False)
    print(len(data), "signs ->", out, meta)


if __name__ == "__main__":
    main()
