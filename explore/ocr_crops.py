"""Run Florence-2 OCR over the crops of one apple_detect run. Writes ocr.jsonl next to instances.jsonl.

Usage: uv run explore/ocr_crops.py coruna store_sign [--min-score 0.5] [--batch 8]
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "florence-community/Florence-2-large"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("slug", nargs="?", default="store_sign")
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--only-label", help="use labels.jsonl from filter_crops and keep only this label")
    ap.add_argument("--ids-file", help="only these crop ids (one per line)")
    a = ap.parse_args()
    base = Path("data") / a.city / "apple" / a.slug
    out = base / "ocr.jsonl"
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    rows = [r for r in map(json.loads, open(base / "instances.jsonl")) if r["score"] >= a.min_score and r["id"] not in done]
    if a.ids_file:
        ids = set(Path(a.ids_file).read_text().split())
        rows = [r for r in rows if r["id"] in ids]
    if a.only_label:
        keep = {o["id"] for o in map(json.loads, open(base / "labels.jsonl")) if o["label"] in a.only_label.split(",")}
        rows = [r for r in rows if r["id"] in keep]
    print(f"{len(rows)} crops to OCR ({len(done)} done)")

    proc = AutoProcessor.from_pretrained(MODEL)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    with open(out, "a") as f:
        for i in range(0, len(rows), a.batch):
            batch = rows[i : i + a.batch]
            ims = [Image.open(base / r["crop"]).convert("RGB") for r in batch]
            inp = proc(text=["<OCR>"] * len(ims), images=ims, return_tensors="pt").to(dev, torch.float16 if dev == "cuda" else torch.float32)
            with torch.no_grad():
                ids = model.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"], max_new_tokens=48, num_beams=3)
            for r, txt in zip(batch, proc.batch_decode(ids, skip_special_tokens=True)):
                f.write(json.dumps({"id": r["id"], "text": txt.strip()}) + "\n")
            if (i // a.batch) % 50 == 0:
                print(f"{i + len(batch)}/{len(rows)}", flush=True)
    print("->", out)


if __name__ == "__main__":
    main()
