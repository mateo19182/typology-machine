"""Tag every deduped sign crop with a Qwen3.5 VLM: is it a sign, what kind, what business, what it says.

Closed vocabularies; anything outside them is mapped to "unknown".
Writes tags.jsonl next to instances.jsonl (resumable).

Usage: uv run explore/tag_signs.py coruna store_sign [--model Qwen/Qwen3.5-9B] [--batch 8] [--limit N]
"""
import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "Qwen/Qwen3.5-9B"

# The schema. Small on purpose: is it a sign, what kind, what business. Everything else comes from elsewhere
# (position, street, barrio, date from the pano; form from the mask later if we want it).
SCHEMA = {
    "is_sign": ["yes", "no"],
    "kind": ["shop_sign", "street_name_plate", "traffic_or_official", "poster_or_notice", "for_rent_or_sale", "other"],
    "business": ["bar_cafe", "restaurant", "fashion", "food_shop", "pharmacy_health", "bank_insurance", "real_estate",
                 "professional_office", "hair_beauty", "hotel", "services", "retail_other", "public_institution", "not_applicable", "unknown"],
}
FREE = ["text"]  # the main words on the sign, as written


def build_prompt():
    lines = ["This is a crop from a street photo in A Coruña, Spain. A detector thought it was a shop sign.",
             "Answer with one JSON object and nothing else, using exactly one value from each list:",
             f"- is_sign: {', '.join(SCHEMA['is_sign'])}  (a sign is lettering or a logo made to be read from the street)",
             f"- kind: {', '.join(SCHEMA['kind'])}",
             f"- business: {', '.join(SCHEMA['business'])}  (what the shop sells or does, read from the sign; not_applicable if it is not a shop sign)",
             "- text: the main words on the sign as written (string, empty if none)"]
    return "\n".join(lines)


def clean(j):
    out = {}
    for k, v in SCHEMA.items():
        val = str(j.get(k, "unknown")).strip().lower().replace(" ", "_").replace("-", "_")
        out[k] = val if val in v else ("unknown" if "unknown" in v else v[-1])
    for k in FREE:
        out[k] = str(j.get(k, "") or "").strip()[:200]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("slug", nargs="?", default="store_sign")
    ap.add_argument("--source", default="apple", help="apple or gsv")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--min-score", type=float, default=0.6)
    ap.add_argument("--max-side", type=int, default=768, help="downscale crops before the VLM")
    ap.add_argument("--all", action="store_true", help="tag every instance, not just dedupe representatives")
    a = ap.parse_args()
    base = Path("data") / a.city / a.source / a.slug
    rows = {r["id"]: r for r in map(json.loads, open(base / "instances.jsonl"))}
    if a.all or not (base / "groups.jsonl").exists():
        ids = list(rows)
    else:
        ids = [g["rep"] for g in map(json.loads, open(base / "groups.jsonl"))]
    ids = [i for i in ids if rows[i]["score"] >= a.min_score]
    out = base / "tags.jsonl"
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    ids = [i for i in ids if i not in done]
    if a.limit:
        ids = ids[: a.limit]
    print(f"{len(ids)} signs to tag ({len(done)} done) with {a.model}", flush=True)
    if not ids:
        return

    proc = AutoProcessor.from_pretrained(a.model)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    prompt = build_prompt()
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)

    import time

    t0 = time.time()
    n = 0
    with open(out, "a") as f:
        for i in range(0, len(ids), a.batch):
            b = ids[i : i + a.batch]
            ims = []
            for sid in b:
                im = Image.open(base / rows[sid]["crop"]).convert("RGB")
                im.thumbnail((a.max_side, a.max_side))
                ims.append(im)
            inp = proc(text=[text] * len(ims), images=ims, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=120, do_sample=False)
            answers = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            for sid, ans in zip(b, answers):
                m = re.search(r"\{.*\}", ans, re.S)
                try:
                    j = json.loads(m.group(0)) if m else {}
                except json.JSONDecodeError:
                    j = {}
                rec = {"id": sid, **clean(j), "raw_ok": bool(j)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            f.flush()
            if (i // a.batch) % 10 == 0:
                dt = time.time() - t0
                print(f"{n}/{len(ids)} {dt / n:.2f}s/sign eta {(len(ids) - n) * dt / n / 60:.0f} min", flush=True)
    print("->", out)


if __name__ == "__main__":
    main()
