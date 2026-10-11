"""Local HTML grid for one apple_detect run: crops sorted by score, OCR text if present, link to Look Around.

Usage: uv run explore/apple_detect.py ... ; uv run explore/apple_grid.py coruna store_sign [--min-score 0.5]
"""
import argparse
import html
import json
from pathlib import Path

from streetlevel import lookaround


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("slug", nargs="?", default="store_sign")
    ap.add_argument("--min-score", type=float, default=0.5)
    a = ap.parse_args()
    base = Path("data") / a.city / "apple" / a.slug
    rows = [r for r in map(json.loads, open(base / "instances.jsonl")) if r["score"] >= a.min_score]
    ocr = {}
    if (base / "ocr.jsonl").exists():
        ocr = {o["id"]: o["text"] for o in map(json.loads, open(base / "ocr.jsonl"))}
    rows.sort(key=lambda r: -r["score"])

    cells = []
    for r in rows:
        # look toward the face: right face = heading+90°, left = heading-90°, front = heading
        yaw = {0: 0, 1: 90, 2: 180, 3: -90}.get(r["face"], 0)
        link = lookaround.build_permalink(lat=r["lat"], lon=r["lon"], heading=r["heading"] + yaw, radians=False)
        txt = html.escape(ocr.get(r["id"], ""))
        cells.append(
            f'<figure><a href="{link}" target="_blank"><img loading="lazy" src="{r["crop"]}" title="{r["id"]} score {r["score"]}"></a>'
            f'<figcaption>{txt}<span>{r["score"]:.2f}</span></figcaption></figure>'
        )
    page = f"""<!doctype html><meta charset="utf-8"><title>{a.slug} · {a.city} · Apple</title>
<style>
body{{background:#f2f0eb;color:#222;font:12px/1.3 system-ui;margin:0;padding:12px}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}}
figure{{margin:0;background:#e4e1da;display:flex;flex-direction:column}}
figure a{{aspect-ratio:1;display:flex;align-items:center;justify-content:center}}
img{{max-width:100%;max-height:100%;filter:grayscale(var(--gs,0))}}
figcaption{{padding:4px 6px;display:flex;justify-content:space-between;gap:6px;color:#444}} figcaption span{{color:#999}}
body.bw{{--gs:1}}
</style>
<p>{len(rows)} instances ≥ {a.min_score} · <label><input type="checkbox" onchange="document.body.classList.toggle('bw',this.checked)"> b/w</label></p>
<div class="g">{''.join(cells)}</div>"""
    out = base / "grid.html"
    out.write_text(page)
    print(out.resolve(), len(rows))


if __name__ == "__main__":
    main()
