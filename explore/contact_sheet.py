"""Random sample of a city's images as a local HTML contact sheet (thumbs hotlinked from Mapillary).

Stratified so recent captures and panos are not drowned out by the 2019 bulk.
Usage: uv run explore/contact_sheet.py [city] [n]
"""
import csv
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.environ["MAPILLARY_TOKEN"]
FIELDS = "id,thumb_1024_url,captured_at,is_pano,make,model,camera_type,computed_compass_angle,height,width"


def graph(ids):
    r = requests.get(
        "https://graph.mapillary.com/",
        params={"ids": ",".join(ids), "fields": FIELDS, "access_token": TOKEN},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def main(city, n):
    rows = list(csv.DictReader(open(Path("data") / city / "images.csv")))
    year = lambda r: datetime.fromtimestamp(int(r["captured_at"]) / 1000, timezone.utc).year
    buckets = defaultdict(list)
    for r in rows:
        buckets[(year(r), r["is_pano"] == "True")].append(r)
    random.seed(0)
    per = max(1, n // len(buckets))
    sample = [r for b in buckets.values() for r in random.sample(b, min(per, len(b)))]

    meta = {}
    ids = [r["id"] for r in sample]
    for i in range(0, len(ids), 50):
        meta.update(graph(ids[i : i + 50]))

    cards = []
    for r in sorted(sample, key=lambda r: (year(r), r["is_pano"])):
        m = meta.get(r["id"], {})
        if not m.get("thumb_1024_url"):
            continue
        cam = f'{m.get("make", "")} {m.get("model", "")}'.strip() or "?"
        label = f'{year(r)} · {"pano" if m.get("is_pano") else m.get("camera_type", "")} · {cam} · q{float(r["quality_score"]):.2f}'
        link = f"https://www.mapillary.com/app/?pKey={r['id']}"
        cards.append(
            f'<a href="{link}" target="_blank"><img loading="lazy" src="{m["thumb_1024_url"]}"><span>{label}</span></a>'
        )

    html = f"""<!doctype html><meta charset="utf-8"><title>{city} contact sheet</title>
<style>body{{background:#111;color:#ddd;font:12px system-ui;margin:16px}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:8px}}
a{{color:#aaa;text-decoration:none}} img{{width:100%;aspect-ratio:4/3;object-fit:cover;display:block;background:#222}}</style>
<h1>{city}: {len(cards)} samples of {len(rows)} images</h1><div class="g">{''.join(cards)}</div>"""
    out = Path("data") / city / "contact_sheet.html"
    out.write_text(html)
    print(out.resolve())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "coruna", int(sys.argv[2]) if len(sys.argv) > 2 else 200)
