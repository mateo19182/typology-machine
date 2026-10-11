"""Ask a vision-language model to name each cluster from a contact sheet of its members.

Reads clusters_<emb>.json from cluster_signs.py, builds a 4x4 sheet per cluster, asks a Qwen3.5 VLM for a short
name and a one-line description, writes cluster_names_<emb>.json and prints a table.

Usage: uv run explore/name_clusters.py coruna store_sign --emb dinov2 [--model Qwen/Qwen3.5-9B]
"""
import argparse
import json
import random
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "Qwen/Qwen3.5-4B"  # 9B is better but needs the GPU to itself
PROMPT = (
    "This contact sheet shows {n} shop signs from A Coruña, Spain. An algorithm grouped them because they look alike. "
    "You are a typologist like Bernd and Hilla Becher: ignore the words, look at the objects. "
    "First decide the dominant value of each attribute across the sheet:\n"
    "shape: one of [long horizontal fascia band, hanging blade sign, square or small plaque, individual letters on wall, "
    "window vinyl or glass, poster or printed panel, light box, awning text, other]\n"
    "material: one of [painted metal, acrylic light box, wood, stone or engraved, glass, fabric, paper or vinyl, mixed]\n"
    "lighting: one of [backlit, unlit, neon or LED letters, mixed]\n"
    "palette: one of [dark ground light type, light ground dark type, bright colours, metallic or gold, mixed]\n"
    "typography: one of [all caps sans, script or cursive, serif, logo or wordmark, mixed]\n"
    "Then give a specific 2-5 word name for this kind of sign that a curator would use as a grid title, and one sentence on what "
    "makes the group recognisable. Do not use the words 'store sign', 'street sign' or 'signage' in the name. "
    "Answer only with JSON: {{\"shape\": ..., \"material\": ..., \"lighting\": ..., \"palette\": ..., \"typography\": ..., "
    "\"name\": ..., \"description\": ...}}"
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
    ap.add_argument("--source", default="apple", help="apple or gsv")
    ap.add_argument("--emb", default="dinov2")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    base = Path("data") / a.city / a.source / a.slug
    rows = {r["id"]: r for r in map(json.loads, open(base / "instances.jsonl"))}
    clusters = json.load(open(base / f"clusters_{a.emb}.json"))

    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()

    out = {}
    for c, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        im = sheet([base / rows[m]["crop"] for m in members])
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT.format(n=min(16, len(members)))}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
        inp = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
        with torch.no_grad():
            ids = model.generate(**inp, max_new_tokens=160, do_sample=False)
        ans = proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        m = re.search(r"\{.*\}", ans, re.S)
        try:
            j = json.loads(m.group(0)) if m else {"name": ans.strip()[:40], "description": ""}
        except json.JSONDecodeError:
            j = {"name": ans.strip()[:40], "description": ""}
        j["size"] = len(members)
        out[c] = j
        attrs = " · ".join(str(j.get(k, "?")) for k in ("shape", "material", "lighting", "palette", "typography"))
        print(f"{c:>3} {len(members):>4}  {j.get('name', '')!s:34} [{attrs}]  {j.get('description', '')}", flush=True)
    json.dump(out, open(base / f"cluster_names_{a.emb}.json", "w"), ensure_ascii=False, indent=1)
    print("model:", a.model)


if __name__ == "__main__":
    main()
