"""Side-by-side: Mapillary's own detections vs SAM 3 text-prompted masks, on the same cached images.

Picks images from data/<city>/cache that have Mapillary detections of the classes we care about, runs SAM 3
with a plain-English prompt per class, and writes data/<city>/sam3_vs_mapillary.html with overlays and counts.
Any extra image paths given with --extra (e.g. an Apple Look Around face) get the SAM 3 column only.

Usage: uv run explore/sam3_vs_mapillary.py [city] [--n 24] [--model JobHarvest/sam3] [--extra img.jpg ...]
"""
import argparse
import base64
import json
import random
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from crop_features import decode_polygon

# Mapillary class -> SAM 3 noun phrase
PROMPTS = {
    "object--sign--store": "store sign",
    "object--sign--advertisement": "advertisement poster",
    "object--trash-can": "trash can",
    "object--street-light": "street light",
    "object--support--pole": "pole",
    "object--bench": "bench",
    "object--manhole": "manhole cover",
    "object--traffic-sign--front": "traffic sign",
    "object--junction-box": "junction box",
    "object--banner": "banner",
}
COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4"]
COLOR = dict(zip(PROMPTS, COLORS))
MAX_W = 1400  # display width


def to_b64(im):
    buf = BytesIO()
    im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def draw_mapillary(im, dets):
    out = im.copy()
    d = ImageDraw.Draw(out, "RGBA")
    W, H = out.size
    counts = Counter()
    for det in dets:
        if det["value"] not in PROMPTS:
            continue
        pts = [(x * W, y * H) for x, y in decode_polygon(det["geometry"])]
        c = COLOR[det["value"]]
        d.polygon(pts, outline=c, width=3)
        counts[det["value"]] += 1
    return out, counts


def run_sam3(model, processor, im, device):
    """Returns {class: [(score, box, mask)]}"""
    res = {}
    for value, prompt in PROMPTS.items():
        inputs = processor(images=im, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        r = processor.post_process_instance_segmentation(
            outputs, threshold=0.5, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        res[value] = [(float(s), b.tolist(), m) for s, b, m in zip(r["scores"], r["boxes"], r["masks"])]
    return res


def draw_sam3(im, res):
    out = im.copy().convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    counts = Counter()
    for value, items in res.items():
        c = COLOR[value]
        rgb = tuple(int(c[i : i + 2], 16) for i in (1, 3, 5))
        for score, box, mask in items:
            m = Image.fromarray((mask.cpu().numpy() * 110).astype("uint8"), "L")
            overlay.paste(Image.new("RGBA", out.size, rgb + (0,)), mask=m)
            solid = Image.new("RGBA", out.size, rgb + (110,))
            overlay.paste(solid, mask=m)
            d.rectangle(box, outline=rgb + (255,), width=2)
            d.text((box[0] + 3, box[1] + 2), f"{score:.2f}", fill=(255, 255, 255, 255))
            counts[value] += 1
    return Image.alpha_composite(out, overlay).convert("RGB"), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--model", default="JobHarvest/sam3")  # ungated mirror of facebook/sam3 (transformers format)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    base = Path("data") / a.city
    cache = base / "cache"

    # candidate images: have a cached jpg + detections of >=2 classes of interest; prefer non-pano
    cands = []
    for jpg in (cache / "jpg").glob("*.jpg"):
        iid = jpg.stem
        dpath, ipath = cache / "detections" / f"{iid}.json", cache / "images" / f"{iid}.json"
        if not (dpath.exists() and ipath.exists()):
            continue
        try:
            dets = json.loads(dpath.read_text())
            meta = json.loads(ipath.read_text())
        except json.JSONDecodeError:
            continue
        classes = {d["value"] for d in dets if d["value"] in PROMPTS}
        if len(classes) >= 2 and not meta.get("is_pano"):
            cands.append((jpg, dets, meta))
    random.Random(a.seed).shuffle(cands)
    cands = cands[: a.n]
    print(f"{len(cands)} images selected")

    from transformers import Sam3Model, Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Sam3Processor.from_pretrained(a.model)
    model = Sam3Model.from_pretrained(a.model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()

    rows, tot_m, tot_s = [], Counter(), Counter()
    for jpg, dets, meta in cands:
        im = Image.open(jpg).convert("RGB")
        if im.width > MAX_W:
            im = im.resize((MAX_W, int(im.height * MAX_W / im.width)))
        left, cm = draw_mapillary(im, dets)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            res = run_sam3(model, processor, im, device)
        right, cs = draw_sam3(im, res)
        tot_m += cm
        tot_s += cs
        rows.append((f"{jpg.stem} · {str(meta.get('captured_at'))[:10]}", to_b64(left), to_b64(right), cm, cs))
        print(jpg.stem, dict(cm), dict(cs), flush=True)

    for path in a.extra:
        im = Image.open(path).convert("RGB")
        if im.width > 2000:
            im = im.resize((2000, int(im.height * 2000 / im.width)))
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            res = run_sam3(model, processor, im, device)
        right, cs = draw_sam3(im, res)
        rows.append((f"extra: {Path(path).name}", None, to_b64(right), Counter(), cs))
        print(path, dict(cs), flush=True)

    legend = " ".join(f'<span style="color:{COLOR[v]}">■ {p}</span>' for v, p in PROMPTS.items())
    table = "".join(
        f"<tr><th>{v}</th><td>{tot_m[v]}</td><td>{tot_s[v]}</td></tr>" for v in PROMPTS
    )
    body = []
    for title, l, r, cm, cs in rows:
        lc = " ".join(f"{k.split('--')[-1]}:{n}" for k, n in cm.items())
        rc = " ".join(f"{k.split('--')[-1]}:{n}" for k, n in cs.items())
        left = f'<figure><img src="{l}"><figcaption>Mapillary · {lc}</figcaption></figure>' if l else "<figure></figure>"
        body.append(f"<h3>{title}</h3><div class=pair>{left}<figure><img src=\"{r}\"><figcaption>SAM 3 · {rc}</figcaption></figure></div>")
    out = base / "sam3_vs_mapillary.html"
    out.write_text(
        f"<!doctype html><meta charset=utf-8><title>SAM 3 vs Mapillary</title><style>"
        "body{background:#111;color:#ddd;font:13px system-ui;margin:16px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}"
        "img{width:100%}figcaption{color:#999}table{border-collapse:collapse}td,th{padding:2px 10px;text-align:left}</style>"
        f"<p>{legend}</p><table><tr><th>class</th><th>Mapillary</th><th>SAM 3</th></tr>{table}</table>{''.join(body)}"
    )
    print(out.resolve())


if __name__ == "__main__":
    main()
