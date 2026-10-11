"""SAM 3 over Google Street View panoramas: each equirectangular pano is cut into perspective views, SAM 3 finds
every instance of the prompt in each view, crops and quarter-res masks are saved like apple_detect does.

Writes under data/<city>/gsv/<slug>/: crops/, masks/, instances.jsonl, done.txt. Each instance records the
pano, the view yaw/pitch, the box in view pixels, and the absolute bearing to the box centre (for triangulation).

Usage: uv run explore/gsv_detect.py coruna --prompt "store sign" [--current-only] [--batch 4] [--views 8]
"""
import argparse
import json
import math
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import py360convert
import torch
from PIL import Image

PAD = 0.12
DOWN = 4
MODEL = "JobHarvest/sam3"
FOV = (70.0, 55.0)  # horizontal, vertical degrees
VIEW_HW = (1600, 2048)
PITCH = 8.0


def cut_views(im, n):
    e = np.asarray(im)
    out = []
    for k in range(n):
        yaw = k * 360.0 / n
        v = py360convert.e2p(e, fov_deg=FOV, u_deg=yaw, v_deg=PITCH, out_hw=VIEW_HW, mode="bilinear")
        out.append((yaw, Image.fromarray(v)))
    return out


def bearing(heading, yaw, x_norm):
    """absolute compass bearing of a point at normalised x (0..1) in a view at relative yaw."""
    f = 0.5 / math.tan(math.radians(FOV[0] / 2))
    rel = math.degrees(math.atan((x_norm - 0.5) / f))
    return (heading + yaw + rel) % 360


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--current-only", action="store_true")
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-side", type=int, default=32)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--loop", action="store_true", help="keep polling panos.jsonl for new panos until none arrive for 10 min")
    a = ap.parse_args()

    base = Path("data") / a.city / "gsv"
    slug = re.sub(r"\W+", "_", a.prompt.strip())
    out = base / slug
    (out / "crops").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    done_path = out / "done.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()

    from transformers import Sam3Model, Sam3Processor

    proc = Sam3Processor.from_pretrained(MODEL)
    model = Sam3Model.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    log = open(out / "instances.jsonl", "a")
    donef = open(done_path, "a")

    def pending():
        rows = [json.loads(l) for l in open(base / "panos.jsonl")] if (base / "panos.jsonl").exists() else []
        rows = [r for r in rows if r["id"] not in done and (base / "panos" / f"{r['id']}.jpg").exists()]
        if a.current_only:
            rows = [r for r in rows if r["is_current"]]
        return rows[: a.limit] if a.limit else rows

    idle_since = time.time()
    n_panos = n_inst = 0
    t0 = time.time()
    while True:
        rows = pending()
        if not rows:
            if not a.loop or time.time() - idle_since > 600:
                break
            time.sleep(30)
            continue
        idle_since = time.time()
        q = queue.Queue(maxsize=a.workers * 2)
        stop = object()

        def producer():
            def one(r):
                try:
                    im = Image.open(base / "panos" / f"{r['id']}.jpg").convert("RGB")
                    q.put((r, cut_views(im, a.views)))
                except Exception as e:
                    print("view error", r["id"], e, flush=True)
                    q.put((r, None))

            with ThreadPoolExecutor(a.workers) as ex:
                list(ex.map(one, rows))
            q.put(stop)

        threading.Thread(target=producer, daemon=True).start()
        while True:
            item = q.get()
            if item is stop:
                break
            r, views = item
            if views is None:
                continue
            for i in range(0, len(views), a.batch):
                chunk = views[i : i + a.batch]
                ims = [v for _, v in chunk]
                inp = proc(images=ims, text=[a.prompt] * len(ims), return_tensors="pt").to("cuda")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    o = model(**inp)
                sizes = [(im.height // DOWN, im.width // DOWN) for im in ims]
                res = proc.post_process_instance_segmentation(o, threshold=a.threshold, mask_threshold=0.5, target_sizes=sizes)
                for (yaw, im), rr in zip(chunk, res):
                    W, H = im.size
                    for k, (score, box, mask) in enumerate(zip(rr["scores"], rr["boxes"], rr["masks"])):
                        x0, y0, x1, y1 = [float(v) * DOWN for v in box]
                        if x1 - x0 < a.min_side or y1 - y0 < a.min_side:
                            continue
                        pw, ph = (x1 - x0) * PAD, (y1 - y0) * PAD
                        cb = (max(0, x0 - pw), max(0, y0 - ph), min(W, x1 + pw), min(H, y1 + ph))
                        name = f"{r['id']}_{int(yaw)}_{k}"
                        im.crop(tuple(int(v) for v in cb)).save(out / "crops" / f"{name}.jpg", quality=90)
                        m = mask.cpu().numpy().astype(np.uint8) * 255
                        Image.fromarray(m).crop(tuple(int(v / DOWN) for v in cb)).save(out / "masks" / f"{name}.png")
                        log.write(json.dumps({
                            "id": name, "pano": r["id"], "spot": r["spot"], "is_current": r["is_current"], "date": r["date"],
                            "lat": r["lat"], "lon": r["lon"], "heading": r["heading"], "view_yaw": yaw, "view_pitch": PITCH,
                            "fov": list(FOV), "view_size": [W, H], "score": round(float(score), 3),
                            "box": [round(v) for v in (x0, y0, x1, y1)], "crop_box": [round(v) for v in cb],
                            "bearing": round(bearing(r["heading"] or 0, yaw, (x0 + x1) / 2 / W), 2),
                            "crop": f"crops/{name}.jpg", "mask": f"masks/{name}.png",
                        }) + "\n")
                        n_inst += 1
                del o, res, inp
            done.add(r["id"])
            donef.write(r["id"] + "\n")
            donef.flush()
            log.flush()
            n_panos += 1
            if n_panos % 50 == 0:
                dt = time.time() - t0
                print(f"{n_panos} panos, {n_inst} instances, {dt / n_panos:.1f}s/pano", flush=True)
    print(f"done: {n_panos} panos, {n_inst} instances in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
