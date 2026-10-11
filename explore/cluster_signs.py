"""Cluster deduped sign crops into sub-types and make one contact sheet per cluster.

Embeddings: 'siglip' reuses embeddings.npy from dedupe_crops.py (semantic), 'dinov2' computes DINOv2 CLS
embeddings (visual form). PCA to 50 dims, then HDBSCAN; k-means as an alternative with --k.
Writes clusters/<emb>/<cluster>.jpg sheets and clusters_<emb>.json next to instances.jsonl.

Usage: uv run explore/cluster_signs.py coruna store_sign --emb dinov2 [--min-cluster 12] [--k 0]
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel

DINO = "facebook/dinov2-base"


def dinov2_embed(paths, batch=64):
    proc = AutoImageProcessor.from_pretrained(DINO)
    model = AutoModel.from_pretrained(DINO, dtype=torch.float16).cuda().eval()
    out = []
    for i in range(0, len(paths), batch):
        ims = [Image.open(p).convert("RGB") for p in paths[i : i + batch]]
        inp = proc(images=ims, return_tensors="pt").to("cuda", torch.float16)
        with torch.no_grad():
            h = model(**inp).last_hidden_state
        e = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1).float()  # CLS + mean patch
        out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("slug", nargs="?", default="store_sign")
    ap.add_argument("--emb", choices=["siglip", "dinov2"], default="dinov2")
    ap.add_argument("--min-cluster", type=int, default=12)
    ap.add_argument("--k", type=int, default=0, help="use k-means with this k instead of HDBSCAN")
    ap.add_argument("--min-score", type=float, default=0.6)
    a = ap.parse_args()
    base = Path("data") / a.city / "apple" / a.slug
    rows = {r["id"]: r for r in map(json.loads, open(base / "instances.jsonl"))}
    reps = [g["rep"] for g in map(json.loads, open(base / "groups.jsonl"))]
    reps = [r for r in reps if rows[r]["score"] >= a.min_score]
    print(len(reps), "deduped signs")

    if a.emb == "siglip":
        E = np.load(base / "embeddings.npy")
        ids = json.load(open(base / "embeddings_ids.json"))
        idx = {i: k for k, i in enumerate(ids)}
        X = E[[idx[r] for r in reps]]
    else:
        cache = base / "embeddings_dinov2.npy"
        if cache.exists() and json.load(open(base / "embeddings_dinov2_ids.json")) == reps:
            X = np.load(cache)
        else:
            X = dinov2_embed([base / rows[r]["crop"] for r in reps])
            np.save(cache, X)
            json.dump(reps, open(base / "embeddings_dinov2_ids.json", "w"))

    Z = PCA(50, random_state=0).fit_transform(X)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    if a.k:
        labels = KMeans(a.k, n_init=4, random_state=0).fit_predict(Z)
    else:
        labels = HDBSCAN(min_cluster_size=a.min_cluster, min_samples=5, metric="euclidean").fit_predict(Z)
    sizes = Counter(labels)
    print(f"{len(sizes) - (1 if -1 in sizes else 0)} clusters, noise {sizes.get(-1, 0)}, sizes {sorted((v for k, v in sizes.items() if k != -1), reverse=True)[:20]}")

    out = base / "clusters" / a.emb
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.jpg"):
        f.unlink()
    clusters = {}
    rnd = random.Random(0)
    for c in sorted(sizes, key=lambda c: -sizes[c]):
        members = [r for r, l in zip(reps, labels) if l == c]
        clusters[int(c)] = members
        s = members if len(members) <= 24 else rnd.sample(members, 24)
        sheet = Image.new("RGB", (6 * 200, ((len(s) + 5) // 6) * 150), (230, 228, 222))
        for i, r in enumerate(s):
            im = Image.open(base / rows[r]["crop"])
            im.thumbnail((190, 140))
            sheet.paste(im, ((i % 6) * 200 + 5, (i // 6) * 150 + 5))
        sheet.save(out / f"{'noise' if c == -1 else c:0>2}.jpg" if c != -1 else out / "noise.jpg", quality=85)
    json.dump(clusters, open(base / f"clusters_{a.emb}.json", "w"))
    print("sheets ->", out)


if __name__ == "__main__":
    main()
