"""Package the sign histories for the viewer: pick the signs with the richest timelines, re-encode their
dated windows and sign crops at web size, and write data.json. A "lives" count per sign comes from DINOv2:
consecutive dated crops that no longer look alike mark a change of sign (new business, new fascia).

Usage: uv run explore/export_history.py coruna --out site_history --n 150 [--size 560]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

DINO = "facebook/dinov2-base"


def embed(paths, batch=48):
    proc = AutoImageProcessor.from_pretrained(DINO)
    model = AutoModel.from_pretrained(DINO, dtype=torch.float16).cuda().eval()
    out = []
    for i in range(0, len(paths), batch):
        ims = [Image.open(p).convert("RGB") for p in paths[i : i + batch]]
        inp = proc(images=ims, return_tensors="pt").to("cuda", torch.float16)
        with torch.no_grad():
            h = model(**inp).last_hidden_state
        e = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1).float()
        out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--size", type=int, default=560)
    ap.add_argument("--change", type=float, default=0.55, help="similarity below this between consecutive crops = a new sign")
    a = ap.parse_args()
    data = Path("data") / a.city
    strips = [json.loads(l) for l in open(data / "history" / "strips.jsonl")]
    loc = {}
    lp = data / "apple" / "store_sign" / "location.jsonl"
    if lp.exists():
        rep = {g["sign"]: g["rep"] for g in map(json.loads, open(data / "apple" / "store_sign" / "groups.jsonl"))}
        by_id = {r["id"]: r for r in map(json.loads, open(lp))}
        loc = {sg: by_id[r] for sg, r in rep.items() if r in by_id}

    # change points from the detected crops
    crops = [(s["sign"], f) for s in strips for f in s["frames"] if f["detected"]]
    E = embed([data / f["crop"] for _, f in crops])
    k = 0
    for s in strips:
        det = [f for f in s["frames"] if f["detected"]]
        es = E[k : k + len(det)]
        k += len(det)
        lives, prev = 1, None
        for f, e in zip(det, es):
            sim = None if prev is None else float(prev @ e)
            f["sim_prev"] = None if sim is None else round(sim, 2)
            f["new"] = sim is not None and sim < a.change
            lives += f["new"]
            prev = e
        s["lives"] = lives
    strips.sort(key=lambda s: (-(s["lives"] >= 2), -s["n_dates"], -s["lives"]))
    strips = strips[: a.n]

    out = Path(a.out)
    (out / "win").mkdir(parents=True, exist_ok=True)
    (out / "sign").mkdir(exist_ok=True)
    signs = []
    for s in strips:
        frames = []
        for f in s["frames"]:
            name = Path(f["file"]).stem
            im = Image.open(data / f["file"]).convert("RGB")
            im.thumbnail((a.size, a.size))
            im.save(out / "win" / f"{name}.jpg", quality=78)
            fr = {"date": f["date"], "src": f["source"], "det": f["detected"], "win": f"win/{name}.jpg", "new": f.get("new", False)}
            if f["detected"]:
                c = Image.open(data / f["crop"]).convert("RGB")
                c.thumbnail((240, 160))
                c.save(out / "sign" / f"{name}.jpg", quality=80)
                fr["sign"] = f"sign/{name}.jpg"
            frames.append(fr)
        l = loc.get(s["sign"], {})
        signs.append({"id": s["sign"], "lat": s["lat"], "lon": s["lon"], "facing": s["facing"], "dates": s["n_dates"], "lives": s["lives"],
                      "street": l.get("street"), "barrio": l.get("barrio"), "frames": frames})
    json.dump({"signs": signs, "city": a.city}, open(out / "data.json", "w"))
    tot = sum(p.stat().st_size for p in out.rglob("*.jpg")) / 1e6
    print(f"{len(signs)} signs, {sum(len(s['frames']) for s in signs)} frames, {tot:.0f} MB -> {out}")


if __name__ == "__main__":
    main()
