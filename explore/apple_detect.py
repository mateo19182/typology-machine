"""Stream Apple Look Around faces through SAM 3 and save crops of every instance of one text prompt.

Pipeline: pick panos from data/<city>/lookaround.csv (year filter, bbox, spatial thinning) -> threads fetch
and decode the requested faces (HEIC, zoom 0) -> the GPU runs SAM 3 in small batches -> crops (full-res,
padded) and quarter-res masks are written under data/<city>/apple/<slug>/, one jsonl row per instance.
Faces are not kept (re-downloading is cheaper than storing them). Resumable: finished (pano, face) pairs
are skipped.

Usage: uv run explore/apple_detect.py coruna --prompt "store sign" [--faces 1,3] [--year 2026]
       [--bbox W,S,E,N] [--thin 6] [--limit 500] [--threshold 0.5] [--batch 4]
Faces: 0 front (124° wide), 1 right, 3 left (67°), 2 back, 4 up, 5 down.
"""
import argparse
import csv
import json
import math
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import pillow_heif
import torch
from PIL import Image
from streetlevel import lookaround

pillow_heif.register_heif_opener()
PAD = 0.12
DOWN = 4  # masks are post-processed at 1/DOWN of the face size, which keeps GPU memory sane
MODEL = "JobHarvest/sam3"  # ungated mirror of facebook/sam3


def thin(rows, d):
    """Keep one pano per d-metre grid cell (latest first)."""
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: r["date"], reverse=True):
        k = (round(float(r["lat"]) * 111320 / d), round(float(r["lon"]) * 111320 * math.cos(math.radians(float(r["lat"]))) / d))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def select_panos(csv_path, a):
    rows = list(csv.DictReader(open(csv_path)))
    if a.year:
        rows = [r for r in rows if r["date"].startswith(a.year)]
    if a.bbox:
        W, S, E, N = map(float, a.bbox.split(","))
        rows = [r for r in rows if W <= float(r["lon"]) <= E and S <= float(r["lat"]) <= N]
    if a.thin:
        rows = thin(rows, a.thin)
    rows.sort(key=lambda r: (r["build_id"], r["id"]))  # keep drives together so tiles cache well
    return rows[: a.limit] if a.limit else rows


class Faces:
    """Download and decode faces. get_panorama_face takes (pano id, build id), both in the csv, so no tile fetch."""

    def __init__(self, workers):
        self.auth = lookaround.Authenticator()
        self.ex = ThreadPoolExecutor(workers)

    def fetch(self, row, face):
        for _ in range(3):
            try:
                data = lookaround.get_panorama_face((int(row["id"]), int(row["build_id"])), face, 0, self.auth)
                im = Image.open(BytesIO(data))
                im.load()
                return im.convert("RGB")
            except Exception:
                time.sleep(1)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--faces", default="1,3")
    ap.add_argument("--year")
    ap.add_argument("--bbox")
    ap.add_argument("--thin", type=float, default=6)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--min-side", type=int, default=32)
    a = ap.parse_args()

    base = Path("data") / a.city
    slug = re.sub(r"\W+", "_", a.prompt.strip())
    out = base / "apple" / slug
    (out / "crops").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    log_path = out / "instances.jsonl"
    done_path = out / "done.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()

    faces = [int(f) for f in a.faces.split(",")]
    rows = select_panos(base / "lookaround.csv", a)
    jobs = [(r, f) for r in rows for f in faces if f"{r['id']}_{f}" not in done]
    print(f"{len(rows)} panos, {len(jobs)} faces to do ({len(done)} already done)", flush=True)
    if not jobs:
        return

    from transformers import Sam3Model, Sam3Processor

    proc = Sam3Processor.from_pretrained(MODEL)
    model = Sam3Model.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()

    fetcher = Faces(a.workers)
    q = queue.Queue(maxsize=a.workers * 2)
    stop = object()

    def producer():
        def one(job):
            row, face = job
            res = fetcher.fetch(row, face)
            q.put((row, face, res))

        list(fetcher.ex.map(one, jobs))
        q.put(stop)

    threading.Thread(target=producer, daemon=True).start()

    log = open(log_path, "a")
    donef = open(done_path, "a")
    n_faces = n_inst = 0
    t0 = time.time()
    batch = []

    def flush():
        nonlocal n_faces, n_inst
        ims = [b[2] for b in batch]
        inp = proc(images=ims, text=[a.prompt] * len(ims), return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(**inp)
        sizes = [(im.height // DOWN, im.width // DOWN) for im in ims]
        res = proc.post_process_instance_segmentation(o, threshold=a.threshold, mask_threshold=0.5, target_sizes=sizes)
        for (row, face, im), r in zip(batch, res):
            W, H = im.size
            for k, (score, box, mask) in enumerate(zip(r["scores"], r["boxes"], r["masks"])):
                x0, y0, x1, y1 = [float(v) * DOWN for v in box]
                if x1 - x0 < a.min_side or y1 - y0 < a.min_side:
                    continue
                pw, ph = (x1 - x0) * PAD, (y1 - y0) * PAD
                cb = (max(0, x0 - pw), max(0, y0 - ph), min(W, x1 + pw), min(H, y1 + ph))
                name = f"{row['id']}_{face}_{k}"
                im.crop(tuple(int(v) for v in cb)).save(out / "crops" / f"{name}.jpg", quality=90)
                m = mask.cpu().numpy().astype(np.uint8) * 255
                mb = tuple(int(v / DOWN) for v in cb)
                Image.fromarray(m).crop(mb).save(out / "masks" / f"{name}.png")
                log.write(json.dumps({
                    "id": name, "pano": row["id"], "face": face, "date": row["date"][:10], "lat": float(row["lat"]),
                    "lon": float(row["lon"]), "heading": float(row["heading"]), "face_size": [W, H],
                    "score": round(float(score), 3), "box": [round(v) for v in (x0, y0, x1, y1)],
                    "crop_box": [round(v) for v in cb], "crop": f"crops/{name}.jpg", "mask": f"masks/{name}.png",
                }) + "\n")
                n_inst += 1
            donef.write(f"{row['id']}_{face}\n")
            n_faces += 1
        log.flush()
        donef.flush()
        batch.clear()
        del o, res, inp

    while True:
        item = q.get()
        if item is stop:
            break
        row, face, res = item
        if res is None:
            print("fetch failed", row["id"], face, flush=True)
            continue
        batch.append((row, face, res))
        if len(batch) >= a.batch:
            flush()
            if n_faces % 100 < a.batch:
                dt = time.time() - t0
                print(f"{n_faces}/{len(jobs)} faces, {n_inst} instances, {dt / n_faces:.2f}s/face, eta {(len(jobs) - n_faces) * dt / n_faces / 60:.0f} min", flush=True)
    if batch:
        flush()
    print(f"done: {n_faces} faces, {n_inst} instances in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
