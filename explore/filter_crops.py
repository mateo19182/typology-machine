"""Zero-shot SigLIP filter over the crops of one apple_detect run: what kind of sign is this really?

SAM 3's "store sign" also returns traffic signs, notices, tunnel exit signs and the odd window. This labels
each crop with the closest of a few text descriptions and writes labels.jsonl next to instances.jsonl.
Also makes a contact sheet per label under data/<city>/apple/<slug>/sheets/ so the labels can be eyeballed.

Usage: uv run explore/filter_crops.py coruna store_sign
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL = "google/siglip-so400m-patch14-384"
LABELS = {
    "shop": "the name sign of a shop, bar, restaurant or business on a storefront",
    "traffic": "a road traffic sign",
    "exit": "an emergency exit sign with a running man pictogram",
    "notice": "a printed notice, poster or informational panel",
    "plate": "a street name plate on a wall",
    "other": "a photo of a building wall, window or street with no sign",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("slug", nargs="?", default="store_sign")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    base = Path("data") / a.city / "apple" / a.slug
    rows = [json.loads(l) for l in open(base / "instances.jsonl")]

    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
    keys = list(LABELS)
    tin = proc(text=list(LABELS.values()), padding="max_length", return_tensors="pt").to("cuda")
    with torch.no_grad():
        temb = model.get_text_features(**tin).pooler_output
    temb = temb / temb.norm(dim=-1, keepdim=True)

    out = []
    for i in range(0, len(rows), a.batch):
        b = rows[i : i + a.batch]
        ims = [Image.open(base / r["crop"]).convert("RGB") for r in b]
        iin = proc(images=ims, return_tensors="pt").to("cuda", torch.float16)
        with torch.no_grad():
            iemb = model.get_image_features(**iin).pooler_output
        iemb = iemb / iemb.norm(dim=-1, keepdim=True)
        p = (iemb @ temb.T * model.logit_scale.exp() + model.logit_bias).sigmoid()
        for r, pr in zip(b, p):
            out.append({"id": r["id"], "label": keys[int(pr.argmax())], "p": round(float(pr.max()), 3)})
    with open(base / "labels.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(Counter(o["label"] for o in out))

    sheets = base / "sheets"
    sheets.mkdir(exist_ok=True)
    byid = {r["id"]: r for r in rows}
    random.seed(5)
    for name in keys:
        s = [o for o in out if o["label"] == name]
        s = random.sample(s, min(24, len(s)))
        sheet = Image.new("RGB", (8 * 200, 3 * 150), (230, 228, 222))
        for i, o in enumerate(s):
            im = Image.open(base / byid[o["id"]]["crop"])
            im.thumbnail((190, 140))
            sheet.paste(im, ((i % 8) * 200 + 5, (i // 8) * 150 + 5))
        sheet.save(sheets / f"{name}.jpg", quality=85)
    print("sheets ->", sheets)


if __name__ == "__main__":
    main()
