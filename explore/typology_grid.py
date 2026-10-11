"""Local HTML viewer: one Becher-style grid per Mapillary class, from data/<city>/crops.jsonl.

Usage: uv run explore/typology_grid.py [city] [--min-px 48]
"""
import argparse
import html
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("--min-px", type=int, default=48, help="drop crops whose short side is smaller")
    a = ap.parse_args()

    base = Path("data") / a.city
    by_class = defaultdict(list)
    for line in open(base / "crops.jsonl"):
        r = json.loads(line)
        if min(r["crop_px"]) >= a.min_px:
            by_class[r["value"]].append(r)
    classes = sorted(by_class, key=lambda v: -len(by_class[v]))

    nav = "".join(f'<a href="#{v}">{html.escape(v.split("--", 1)[1])} <b>{len(by_class[v])}</b></a>' for v in classes)
    sections = []
    for v in classes:
        rows = sorted(by_class[v], key=lambda r: -min(r["crop_px"]))  # sharpest first
        cells = "".join(
            f'<a href="https://www.mapillary.com/app/?pKey={r["image"]}&focus=photo" target="_blank" title="{r["crop_px"][0]}×{r["crop_px"][1]}px">'
            f'<img loading="lazy" src="{r["file"]}"></a>'
            for r in rows
        )
        sections.append(f'<section id="{v}"><h2>{html.escape(v)} · {len(rows)}</h2><div class="g">{cells}</div></section>')

    page = f"""<!doctype html><meta charset="utf-8"><title>{a.city} typologies</title>
<style>
body{{background:#f2f0eb;color:#222;font:13px/1.4 system-ui;margin:0}}
nav{{position:sticky;top:0;background:#f2f0ebee;padding:10px 16px;display:flex;flex-wrap:wrap;gap:6px 14px;border-bottom:1px solid #ccc;z-index:1}}
nav a{{color:#444;text-decoration:none}} section{{padding:8px 16px 32px}} h2{{font-weight:500;letter-spacing:.02em}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:10px}}
.g a{{aspect-ratio:1;display:flex;align-items:center;justify-content:center;background:#e4e1da}}
.g img{{max-width:100%;max-height:100%;filter:grayscale(var(--gs,0))}}
body.bw{{--gs:1}}
</style>
<nav><label><input type="checkbox" onchange="document.body.classList.toggle('bw',this.checked)"> b/w</label>{nav}</nav>
{''.join(sections)}"""
    out = base / "typologies.html"
    out.write_text(page)
    print(out.resolve(), {v: len(by_class[v]) for v in classes[:10]})


if __name__ == "__main__":
    main()
