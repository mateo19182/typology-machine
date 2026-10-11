"""Ask a vision-language model to name each cluster from a contact sheet of its members.

Reads clusters_<emb>.json from cluster_signs.py, builds a 4x4 sheet per cluster, asks Qwen2.5-VL for a short
name and a one-line description, writes cluster_names_<emb>.json and prints a table.

Usage: uv run explore/name_clusters.py coruna store_sign --emb dinov2
"""
import argparse
import json
import random
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPT = (
    "These are {n} shop signs from the streets of A Coruña, Spain, grouped together by an algorithm because they look alike. "
    "Ignore what the words say. Describe what the signs have in common as physical objects: shape, material, how they are mounted, "
    "lighting, typography, colour. Answer with JSON: {{\"name\": <2-4 word name for this type of sign>, \"description\": <one sentence>}}"
)


def sheet(paths, k=16, cell=256):
    rnd = random.Random(0)
    s = paths if len(paths) <= k else rnd.sample(paths, k)
    cols = 4
    rows = (len(s) + cols - 1) // cols
    im = Image.new("RGB", (cols * cell, rows * cell), (235, 233, 228))
    for i, p in enumerate(s):
        c = Image.open(p).convert("RGB")
        c.thumbnail((cell - 12, cell - 12))
        im.paste(c, ((i % cols) * cell + (cell - c.width) // 2, (i // cols) * cell + (cell - c.height) // 2))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("slug", nargs="?", default="store_sign")
    ap.add_argument("--emb", default="dinov2")
    a = ap.parse_args()
    base = Path("data") / a.city / "apple" / a.slug
    rows = {r["id"]: r for r in map(json.loads, open(base / "instances.jsonl"))}
    clusters = json.load(open(base / f"clusters_{a.emb}.json"))

    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

    out = {}
    for c, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        im = sheet([base / rows[m]["crop"] for m in members])
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT.format(n=min(16, len(members)))}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
        with torch.no_grad():
            ids = model.generate(**inp, max_new_tokens=120, do_sample=False)
        ans = proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        m = re.search(r"\{.*\}", ans, re.S)
        try:
            j = json.loads(m.group(0)) if m else {"name": ans.strip()[:40], "description": ""}
        except json.JSONDecodeError:
            j = {"name": ans.strip()[:40], "description": ""}
        j["size"] = len(members)
        out[c] = j
        print(f"{c:>3} {len(members):>4}  {j.get('name', '')!s:32} {j.get('description', '')}", flush=True)
    json.dump(out, open(base / f"cluster_names_{a.emb}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
