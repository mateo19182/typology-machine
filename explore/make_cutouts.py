"""Segmented versions of the crops: the crop with the SAM mask as alpha, written as PNG under
data/<city>/apple/<slug>/cutouts/<id>.png. By default only group representatives; --all for every view.

Usage: uv run explore/make_cutouts.py coruna store_sign [--all]
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageFilter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("slug", nargs="?", default="store_sign")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    base = Path("data") / a.city / "apple" / a.slug
    out = base / "cutouts"
    out.mkdir(exist_ok=True)
    rows = {r["id"]: r for r in map(json.loads, open(base / "instances.jsonl"))}
    if a.all or not (base / "groups.jsonl").exists():
        ids = list(rows)
    else:
        ids = [g["rep"] for g in map(json.loads, open(base / "groups.jsonl"))]
    n = 0
    for sid in ids:
        dst = out / f"{sid}.png"
        if dst.exists():
            continue
        r = rows[sid]
        im = Image.open(base / r["crop"]).convert("RGBA")
        m = Image.open(base / r["mask"]).convert("L").resize(im.size, Image.BILINEAR).filter(ImageFilter.GaussianBlur(1))
        im.putalpha(m)
        im.save(dst, optimize=True)
        n += 1
    print(f"{n} cutouts written -> {out}")


if __name__ == "__main__":
    main()
