"""Tag every deduped sign crop with a fixed attribute schema using a Qwen3.5 VLM. One sign per call, batched.

The schema is the typology: grids are built from attribute values, not from unsupervised clusters.
Closed vocabularies, "unknown" allowed; anything outside the vocabulary is mapped to "unknown".
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

# The schema. Order matters a little: the model answers in this order, and "kind" first lets it bail out early.
SCHEMA = {
    # what the object is. Only shop_sign feeds the shop typology; the others are their own series.
    "kind": ["shop_sign", "street_name_plate", "traffic_or_official", "poster_or_notice", "for_rent_or_sale", "not_a_sign", "unclear"],
    # where it sits on the building: the Becher axis, the sign as an object
    "placement": ["fascia", "projecting", "flat_on_wall", "on_glass", "on_awning", "freestanding", "other"],
    # how it is made
    "construction": ["flat_panel", "light_box", "dimensional_letters", "neon_tube", "led_letters", "vinyl_on_glass", "painted_directly", "engraved_plate", "fabric", "other"],
    "material": ["metal", "acrylic_plastic", "wood", "stone", "ceramic_tile", "glass", "fabric", "paper_cardboard", "brass_gold", "unknown"],
    "illumination": ["none", "backlit", "front_lit", "self_lit_letters", "unknown"],
    # colour, two values so grids can be sorted by ground and by type
    "ground_colour": ["black", "white", "grey", "red", "blue", "green", "yellow", "orange", "brown", "gold", "pink_purple", "multicolour", "transparent", "unknown"],
    "type_colour": ["black", "white", "grey", "red", "blue", "green", "yellow", "orange", "brown", "gold", "pink_purple", "multicolour", "unknown"],
    # lettering
    "case": ["all_caps", "mixed_case", "lowercase", "logo_only", "unknown"],
    "type_style": ["sans", "serif", "script", "display_decorative", "handwritten", "mixed", "none", "unknown"],
    # the words: what they say, what language. Galician vs Spanish is a real axis in this city.
    "language": ["galician", "spanish", "english", "other", "ambiguous", "none"],
    # what the business is, read from the sign only
    "business": ["bar_cafe", "restaurant", "fashion", "food_shop", "pharmacy_health", "bank_insurance", "real_estate", "professional_office", "hair_beauty", "hotel", "services", "retail_other", "public_institution", "unknown"],
    "chain": ["chain_or_franchise", "independent", "unknown"],
    # state of the object
    "condition": ["new", "good", "worn", "damaged_or_faded", "unknown"],
    "era_impression": ["traditional_pre_1980", "late_20th_century", "contemporary", "unknown"],
    # is the crop usable for a grid
    "complete": ["yes", "partly_cut", "no"],
    "occluded": ["no", "partly", "mostly"],
}
FREE = ["text", "secondary_text"]  # strings


def build_prompt():
    lines = ["You are cataloguing shop signs photographed from the street in A Coruña, Galicia, Spain, for a typology in the manner of Bernd and Hilla Becher.",
             "Look at this one crop and fill in every field with exactly one value from its list. Use 'unknown' when the image does not show it. Do not guess from the words what the object is made of.",
             "Fields and allowed values:"]
    for k, v in SCHEMA.items():
        lines.append(f"- {k}: {', '.join(v)}")
    lines.append("- text: the main name on the sign as written (string, empty if none)")
    lines.append("- secondary_text: any other words on the sign (string, empty if none)")
    lines.append("Answer with one JSON object and nothing else.")
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
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--min-score", type=float, default=0.6)
    ap.add_argument("--max-side", type=int, default=768, help="downscale crops before the VLM")
    ap.add_argument("--all", action="store_true", help="tag every instance, not just dedupe representatives")
    a = ap.parse_args()
    base = Path("data") / a.city / a.slug if (Path("data") / a.city / a.slug).exists() else Path("data") / a.city / "apple" / a.slug
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
                gen = model.generate(**inp, max_new_tokens=260, do_sample=False)
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
