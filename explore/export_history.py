"""Package the sign histories for the viewer: pick the signs with the richest timelines and pack each one's
dated windows into one vertical sprite (480x360 per frame) and its sign crops into one horizontal sprite
(240x160 cells), so a sign is two files whatever its length. Writes data.json.

A "lives" count per sign comes from DINOv2: a dated crop that looks unlike the three before it marks a change
of sign. Noisy on busy corners where several signs share a ray; read it as a hint, not a count.

Usage: uv run explore/export_history.py coruna --out site_history --n 160 [--size 480]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

DINO = "facebook/dinov2-base"
CELL = (240, 160)


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


def fit(im, size, bg=(31, 29, 22)):
    im = im.convert("RGB")
    im.thumbnail(size)
    cell = Image.new("RGB", size, bg)
    cell.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--change", type=float, default=0.35, help="max similarity to the previous three crops below which a crop is a new sign")
    a = ap.parse_args()
    data = Path("data") / a.city
    strips = [json.loads(l) for l in open(data / "history" / "strips.jsonl")]
    loc = {}
    lp = data / "apple" / "store_sign" / "location.jsonl"
    if lp.exists():
        rep = {g["sign"]: g["rep"] for g in map(json.loads, open(data / "apple" / "store_sign" / "groups.jsonl"))}
        by_id = {r["id"]: r for r in map(json.loads, open(lp))}
        loc = {sg: by_id[r] for sg, r in rep.items() if r in by_id}

    crops = [f for s in strips for f in s["frames"] if f["detected"]]
    E = embed([data / f["crop"] for f in crops])
    k = 0
    for s in strips:
        det = [f for f in s["frames"] if f["detected"]]
        es = E[k : k + len(det)]
        k += len(det)
        lives = 1
        for i, f in enumerate(det):
            m = max((float(es[i] @ es[j]) for j in range(max(0, i - 3), i)), default=None)
            f["new"] = m is not None and m < a.change
            lives += f["new"]
        s["lives"] = lives
    strips.sort(key=lambda s: (-s["n_dates"], -s["lives"]))
    strips = strips[: a.n]

    out = Path(a.out)
    (out / "win").mkdir(parents=True, exist_ok=True)
    (out / "sign").mkdir(exist_ok=True)
    W, H = a.size, a.size * 3 // 4
    signs = []
    for s in strips:
        fr = s["frames"]
        sprite = Image.new("RGB", (W, H * len(fr)))
        det = [f for f in fr if f["detected"]]
        cs = Image.new("RGB", (CELL[0] * max(1, len(det)), CELL[1]))
        frames, ci = [], 0
        for i, f in enumerate(fr):
            sprite.paste(Image.open(data / f["file"]).convert("RGB").resize((W, H), Image.LANCZOS), (0, i * H))
            o = {"date": f["date"], "src": f["source"][0], "det": f["detected"], "new": f.get("new", False), "ci": None}
            if f["detected"]:
                cs.paste(fit(Image.open(data / f["crop"]), CELL), (ci * CELL[0], 0))
                o["ci"] = ci
                ci += 1
            frames.append(o)
        sprite.save(out / "win" / f"{s['sign']}.jpg", quality=76)
        cs.save(out / "sign" / f"{s['sign']}.jpg", quality=80)
        l = loc.get(s["sign"], {})
        signs.append({"id": s["sign"], "lat": s["lat"], "lon": s["lon"], "facing": s["facing"], "dates": s["n_dates"], "lives": s["lives"],
                      "street": l.get("street"), "barrio": l.get("barrio"), "frames": frames})
    json.dump({"signs": signs, "city": a.city, "win": [W, H], "cell": CELL}, open(out / "data.json", "w"))
    tot = sum(p.stat().st_size for p in out.rglob("*.jpg")) / 1e6
    print(f"{len(signs)} signs, {sum(len(s['frames']) for s in signs)} frames, {tot:.0f} MB -> {out}")


if __name__ == "__main__":
    main()
