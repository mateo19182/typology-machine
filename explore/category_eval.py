"""Small category review set and Tailscale review server.

uv run explore/category_eval.py prepare
uv run explore/category_eval.py serve --host 100.83.166.6 --port 8792
"""
import argparse
import hashlib
import json
import random
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data/coruna/category_eval"
CATEGORIES = {
    "store_sign": "A fixed sign identifying a shop, business, or institution; includes fascia, blade signs and business nameplates.",
    "street_name_plate": "An official plate naming a street or square. Excludes shop names and road-direction signs.",
    "graffiti": "Painted or drawn tags, lettering, stencils, or murals on a surface, rather than a business sign.",
    "poster_notice": "An advertisement, sale/rental notice, informational poster, or temporary printed announcement.",
    "banner": "A hanging or stretched fabric/vinyl banner, rather than a rigid sign or paper poster.",
    "street_light": "An outdoor street-light fixture or lamp post.",
    "bin": "A public litter bin, waste container, or wheelie bin.",
    "bench": "An outdoor bench or public seat.",
    "pole": "A plain utility/support pole; use street_light when it visibly includes a street lamp.",
    "manhole_cover": "A cover or access hatch in the ground; excludes drainage grates.",
    "traffic_sign": "A road-safety, parking, directional, or other official regulatory sign; excludes street-name plates.",
    "junction_box": "A street-side electrical, utility, or telecom cabinet/box.",
    "other": "The main subject is visible but belongs to none of the listed categories.",
    "unclear": "Too small, blurred, occluded, or mixed to assign one primary category reliably.",
}
INSTRUCTION = ("Classify the primary object in this street-photo crop from A Coruña. "
               "Use the visible appearance and the category definitions. Do not assume it is a shop sign. "
               "If graffiti is incidental on another clear primary object, choose that object's category. "
               "Choose other for an identifiable object outside the list; unclear when the image does not "
               "support one category. Return only the category; do not transcribe text or infer attributes.")


def read_jsonl(path):
    if not path.exists():
        return []
    # Active detectors append complete lines; ignore an unfinished final write.
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith("\n")]


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def prepare():
    BASE.mkdir(parents=True, exist_ok=True)
    manifest = BASE / "sample.json"
    if manifest.exists():
        print("Frozen sample already exists:", manifest)
        return
    rng, chosen = random.Random(42), []
    for source in ("apple", "gsv"):
        base = ROOT / "data/coruna" / source / "store_sign"
        groups = read_jsonl(base / "groups.jsonl")
        reps = {g["rep"]: g["sign"] for g in groups}
        old_labels = {r["id"]: r["label"] for r in read_jsonl(base / "labels.jsonl")}
        buckets = defaultdict(list)
        for r in read_jsonl(base / "instances.jsonl"):
            if r["id"] not in reps or r["score"] < .6:
                continue
            buckets[old_labels.get(r["id"], "unlabeled")].append(r)
        for rows in buckets.values():
            rng.shuffle(rows)
        selected, panos = [], set()
        # Old labels only diversify selection. They are never reference labels.
        while len(selected) < 80 and any(buckets.values()):
            for key in sorted(buckets):
                if len(selected) == 80:
                    break
                while buckets[key]:
                    r = buckets[key].pop()
                    if r["pano"] not in panos and (base / r["crop"]).exists():
                        selected.append(r)
                        panos.add(r["pano"])
                        break
        for r in selected:
            chosen.append({"id": f"{source}:{r['id']}", "source": source, "date": r.get("date"),
                           "group": reps[r["id"]], "pano": r["pano"], "path": str(base / r["crop"]),
                           "sampling_hint": old_labels.get(r["id"], "unlabeled"), "detector_prompt": "store sign"})
    mapillary = defaultdict(dict)
    values = ["street-light", "trash-can", "bench", "support--pole", "manhole", "traffic-sign--front", "junction-box", "banner"]
    for r in read_jsonl(ROOT / "data/coruna/crops.jsonl"):
        key = r["value"].removeprefix("object--")
        if key in values and min(r["crop_px"]) >= 48:
            mapillary[key][r["feature"]] = r
    for key in values:
        candidates = [r for r in mapillary[key].values() if (ROOT / "data/coruna" / r["file"]).exists()]
        for r in rng.sample(candidates, min(10, len(candidates))):
            chosen.append({"id": f"mapillary:{r['feature']}", "source": "mapillary", "date": str(r["captured_at"]),
                           "group": r["feature"], "pano": r["image"], "path": str(ROOT / "data/coruna" / r["file"]),
                           "sampling_hint": key, "detector_prompt": None})
    rng.shuffle(chosen)
    (BASE / "images").mkdir(exist_ok=True)
    for i, item in enumerate(chosen):
        original = Path(item["path"])
        item["sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
        item["image"] = f"images/{i}.jpg"
        with Image.open(original) as im:
            im = im.convert("RGB")
            im.thumbnail((768, 768))
            im.save(BASE / item["image"], quality=95)
        item["evaluation_sha256"] = hashlib.sha256((BASE / item["image"]).read_bytes()).hexdigest()
    sample_id = hashlib.sha256(json.dumps(chosen, sort_keys=True).encode()).hexdigest()[:16]
    write_json(manifest, {"id": sample_id, "categories": CATEGORIES, "instruction": INSTRUCTION,
                          "items": chosen, "note": "Exploratory, deliberately diverse sample. Human labels start empty. Known within-source groups sampled once; cross-source duplicates remain possible."})
    print(f"Prepared {len(chosen)} crops -> {manifest}")


def connect():
    db = sqlite3.connect(BASE / "reviews.sqlite3", timeout=10)
    db.execute("CREATE TABLE IF NOT EXISTS labels (sample_id TEXT, item_id TEXT, category TEXT, updated_at TEXT, PRIMARY KEY(sample_id,item_id))")
    return db


def snapshot(sample_base=None):
    sample_base = sample_base or BASE
    sample = json.loads((sample_base / "sample.json").read_text())
    with connect() as db:
        labels = {r[0]: {"category": r[1], "updated_at": r[2]} for r in db.execute("SELECT item_id, category, updated_at FROM labels WHERE sample_id=?", (sample["id"],))}
    predictions, runs = {}, {}
    for file in sorted((sample_base / "runs").glob("*/run.json")):
        run = json.loads(file.read_text())
        if run["sample_id"] != sample["id"]:
            continue
        key = file.parent.name
        runs[key] = run
        predictions[key] = {r["id"]: r for r in read_jsonl(file.parent / "predictions.jsonl")}
    return {"sample": sample, "labels": labels, "runs": runs, "predictions": predictions}


def make_server(host, port):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, data, content_type="application/json"):
            body = json.dumps(data, ensure_ascii=False).encode() if content_type == "application/json" else data
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            sample_base = BASE
            if path == "/multicategory" or path.startswith("/multicategory/"):
                sample_base = BASE / "multicategory"
                path = path.removeprefix("/multicategory") or "/"
            if path in ("/api/data", "/api/export"):
                self.respond(200, snapshot(sample_base))
            elif path == "/":
                self.respond(200, Path(__file__).with_suffix(".html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/detection":
                self.respond(200, Path(__file__).with_name("detection_eval.html").read_bytes(), "text/html; charset=utf-8")
            elif path.startswith("/detector/"):
                file = (BASE / path.lstrip("/")).resolve()
                if file.is_relative_to((BASE / "detector").resolve()) and file.is_file() and file.suffix in (".json", ".jpg"):
                    content_type = "image/jpeg" if file.suffix == ".jpg" else "application/json; charset=utf-8"
                    self.respond(200, file.read_bytes(), content_type)
                else:
                    self.respond(404, {"error": "Not ready or not found"})
            elif path.startswith("/images/"):
                file = sample_base / path.lstrip("/")
                if file.resolve().parent == (sample_base / "images").resolve() and file.is_file():
                    self.respond(200, file.read_bytes(), "image/jpeg")
                else:
                    self.respond(404, {"error": "Not found"})
            else:
                self.respond(404, {"error": "Not found"})

        def do_POST(self):
            sample_base = BASE / "multicategory" if self.path == "/multicategory/api/label" else BASE
            if self.path not in ("/api/label", "/multicategory/api/label"):
                return self.respond(404, {"error": "Not found"})
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self.respond(403, {"error": "Cross-origin writes are not allowed"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                sample = json.loads((sample_base / "sample.json").read_text())
                if request["sample_id"] != sample["id"] or request["id"] not in {r["id"] for r in sample["items"]}:
                    raise ValueError("Unknown sample or crop")
                category = request["category"]
                if category is not None and category not in sample["categories"]:
                    raise ValueError("Unknown category")
                with connect() as db:
                    if category is None:
                        db.execute("DELETE FROM labels WHERE sample_id=? AND item_id=?", (sample["id"], request["id"]))
                    else:
                        db.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?,?)", (sample["id"], request["id"], category, datetime.now(timezone.utc).isoformat()))
                self.respond(200, {"saved": True})
            except (ValueError, KeyError, TypeError) as e:
                self.respond(400, {"error": str(e)})

    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["prepare", "serve"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8792)
    args = ap.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        print(f"Review page: http://{args.host}:{args.port}", flush=True)
        make_server(args.host, args.port).serve_forever()
