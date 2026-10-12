"""Group crops that are the same physical sign seen from several panos: DINOv2 embedding similarity plus
proximity. Nothing is dropped. Writes groups.jsonl (one row per sign with all its views and a representative)
and signs.jsonl (one row per crop: which sign it belongs to, how many views, is it the representative),
plus embeddings.npy for later clustering.

Usage: uv run explore/dedupe_crops.py coruna store_sign [--sim 0.9] [--radius 40]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree
from transformers import AutoImageProcessor, AutoModel

MODEL = "facebook/dinov2-base"  # visual similarity only; no text model needed anywhere in the pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("slug", nargs="?", default="store_sign")
    ap.add_argument("--source", default="apple", help="apple or gsv")
    ap.add_argument("--labels", default="", help="optional: only crops with these labels.jsonl labels")
    ap.add_argument("--sim", type=float, default=0.8)
    ap.add_argument("--radius", type=float, default=30, help="metres between pano positions")
    ap.add_argument("--max-extent", type=float, default=50, help="split groups whose pano positions spread wider than this (m)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--current-only", action="store_true", help="gsv: only crops from the current pass (no history)")
    a = ap.parse_args()
    base = Path("data") / a.city / a.source / a.slug
    rows = [r for r in map(json.loads, open(base / "instances.jsonl"))]
    if a.current_only:
        rows = [r for r in rows if r.get("is_current", True)]
    if a.labels:
        keep = {o["id"] for o in map(json.loads, open(base / "labels.jsonl")) if o["label"] in a.labels.split(",")}
        rows = [r for r in rows if r["id"] in keep]
    print(len(rows), "crops")

    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    embs = []
    for i in range(0, len(rows), a.batch):
        ims = [Image.open(base / r["crop"]).convert("RGB") for r in rows[i : i + a.batch]]
        iin = proc(images=ims, return_tensors="pt").to("cuda", torch.float16)
        with torch.no_grad():
            h = model(**iin).last_hidden_state
        e = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1).float()  # CLS + mean patch
        embs.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
    E = np.concatenate(embs)
    np.save(base / "embeddings.npy", E)
    json.dump([r["id"] for r in rows], open(base / "embeddings_ids.json", "w"))

    lat0 = rows[0]["lat"]
    xy = np.array([(r["lon"] * 111320 * math.cos(math.radians(lat0)), r["lat"] * 111320) for r in rows])
    tree = cKDTree(xy)
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in tree.query_pairs(a.radius):
        if rows[i]["pano"] != rows[j]["pano"] and float(E[i] @ E[j]) >= a.sim:
            parent[find(i)] = find(j)

    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    # repeated identical designs (chain signage, tunnel exit signs) chain-merge along a street; split those by
    # position so a group is one physical spot
    from scipy.cluster.hierarchy import fcluster, linkage

    split = {}
    for k, members in groups.items():
        if len(members) > 1:
            pts = xy[members]
            if np.ptp(pts, axis=0).max() > a.max_extent:
                lab = fcluster(linkage(pts, method="complete"), a.max_extent, criterion="distance")
                for m, l in zip(members, lab):
                    split.setdefault((k, int(l)), []).append(m)
                continue
        split[(k, 0)] = members
    groups = split
    with open(base / "groups.jsonl", "w") as f, open(base / "signs.jsonl", "w") as g:
        for k, members in enumerate(groups.values()):
            best = max(members, key=lambda i: rows[i]["score"] * min(rows[i]["box"][2] - rows[i]["box"][0], rows[i]["box"][3] - rows[i]["box"][1]))
            sign = f"s{k:05d}"
            f.write(json.dumps({"sign": sign, "rep": rows[best]["id"], "members": [rows[i]["id"] for i in members]}) + "\n")
            for i in members:
                g.write(json.dumps({"id": rows[i]["id"], "sign": sign, "views": len(members), "is_rep": i == best}) + "\n")
    sizes = [len(m) for m in groups.values()]
    print(f"{len(rows)} crops -> {len(groups)} groups; singletons {sum(s == 1 for s in sizes)}, largest {max(sizes)}")


if __name__ == "__main__":
    main()
