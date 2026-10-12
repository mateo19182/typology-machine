"""Build a standalone, offline-capable audit of saved deduplication groups. No inference or mutations."""
import base64
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parents[1]


def build():
    cases, images, summaries = [], {}, []
    rng = np.random.default_rng(42)
    fingerprints = {}
    for source in ("apple", "gsv"):
        folder = ROOT / "data/coruna" / source / "store_sign"
        group_bytes = (folder / "groups.jsonl").read_bytes()
        fingerprints[source] = hashlib.sha256(group_bytes).hexdigest()
        groups = [json.loads(l) for l in group_bytes.splitlines()]
        ids = json.loads((folder / "embeddings_ids.json").read_text())
        index = {sid: i for i, sid in enumerate(ids)}
        embeddings = np.load(folder / "embeddings.npy", mmap_mode="r")
        assert len(ids) == len(embeddings) and len(index) == len(ids)
        rows = {}
        for line in (folder / "instances.jsonl").open():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row["id"] in index:
                rows[row["id"]] = row
        assert all(sid in rows for sid in ids)
        group_of = {sid: g["sign"] for g in groups for sid in g["members"]}
        lat0 = rows[ids[0]]["lat"]
        xy = np.array([(rows[sid]["lon"]*111320*np.cos(np.radians(lat0)), rows[sid]["lat"]*111320) for sid in ids])
        tree = cKDTree(xy)
        multi = [g for g in groups if len(g["members"]) > 1]
        summaries.append({"source": source, "crops": len(ids), "groups": len(groups),
                          "singletons": len(groups)-len(multi), "largest": max(len(g["members"]) for g in groups),
                          "groups_sha256": fingerprints[source]})

        def picture(sid, rep=False):
            key = source + ":" + sid
            row = rows[sid]
            if key not in images:
                with Image.open(folder / row["crop"]) as im:
                    im = im.convert("RGB")
                    im.thumbnail((420, 300))
                    buf = BytesIO()
                    im.save(buf, format="JPEG", quality=80)
                images[key] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            return {"image": key, "id": sid, "group": group_of[sid], "pano": row["pano"],
                    "date": row.get("date"), "rep": rep, "lat": row["lat"], "lon": row["lon"],
                    "view": row.get("face", row.get("view_yaw")), "crop": str(folder / row["crop"])}

        def group_case(g, reason):
            members = g["members"]
            idx = [index[sid] for sid in members]
            e = np.asarray(embeddings[idx])
            sims = e @ e.T
            np.fill_diagonal(sims, 2)
            a, b = np.unravel_index(np.argmin(sims), sims.shape)
            extent = float(pdist(xy[idx]).max())
            # Always include the weakest pair and representative; bound file size for huge groups.
            order = list(dict.fromkeys([g["rep"], members[a], members[b]] +
                        [members[int(i)] for i in np.linspace(0, len(members)-1, min(33, len(members))) ]))
            if len(members) <= 36:
                order = list(dict.fromkeys([g["rep"], members[a], members[b]] + members))
            cases.append({"id": source+":"+g["sign"], "source": source, "kind": "merged",
                          "reason": reason, "group": g["sign"], "total": len(members),
                          "min_similarity": float(sims[a,b]), "extent": extent,
                          "same_pano": len(members)-len({rows[s]["pano"] for s in members}),
                          "weak_pair": [members[a], members[b]],
                          "members": [picture(sid, sid == g["rep"]) for sid in order]})

        selected = set()
        for g in sorted(multi, key=lambda g: len(g["members"]), reverse=True)[:8]:
            selected.add(g["sign"])
            group_case(g, "Large group: check for chains joining repeated designs or neighboring objects.")
        # Screen a bounded random subset for transitive merges with weak pairwise similarity.
        pool = [g for g in multi if g["sign"] not in selected]
        candidates = []
        for n in rng.choice(len(pool), min(500, len(pool)), replace=False):
            g = pool[int(n)]
            e = np.asarray(embeddings[[index[s] for s in g["members"]]])
            candidates.append((float((e @ e.T).min()), g))
        for score, g in sorted(candidates, key=lambda x: x[0])[:8]:
            selected.add(g["sign"])
            group_case(g, "Weak visual match within one group: a possible chain merge.")
        remaining = [g for g in multi if g["sign"] not in selected]
        for n in rng.choice(len(remaining), min(8, len(remaining)), replace=False):
            group_case(remaining[int(n)], "Random multi-view group: a less targeted comparison sample.")

        # Find visually similar nearby crops currently assigned to different groups.
        pairs = {}
        for i in rng.choice(len(ids), min(2500, len(ids)), replace=False):
            sid = ids[int(i)]
            neighbors = [j for j in tree.query_ball_point(xy[i], 30)
                         if group_of[ids[j]] != group_of[sid]]
            if not neighbors:
                continue
            scores = np.asarray(embeddings[neighbors]) @ embeddings[i]
            for n in np.argsort(scores)[-2:]:
                j, score = neighbors[int(n)], float(scores[n])
                if score < .7:
                    continue
                key = tuple(sorted((sid, ids[j])))
                pairs[key] = {"similarity": score, "distance": float(np.linalg.norm(xy[i]-xy[j])),
                              "same_pano": rows[sid]["pano"] == rows[ids[j]]["pano"]}
        used = set()
        for same_pano in (True, False):
            ranked = [(key, val) for key, val in pairs.items() if val["same_pano"] == same_pano]
            count = 0
            for key, val in sorted(ranked, key=lambda x: x[1]["similarity"], reverse=True):
                # Avoid showing the same crop repeatedly in this small review.
                if any(s in used for s in key):
                    continue
                used.update(key)
                cases.append({"id": source+":pair:"+":".join(key), "source": source, "kind": "separate",
                    "reason": "Similar nearby crops in separate groups. " +
                        ("Same panorama: direct matches are excluded by the current algorithm." if same_pano else
                         "Different panoramas: check for a missed match or a legitimate distinct object."),
                    **val, "members": [picture(s) for s in key]})
                count += 1
                if count == 8:
                    break
    snapshot = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()[:16]
    data = {"snapshot": snapshot, "summaries": summaries, "cases": cases, "images": images}
    template = (ROOT / "explore/dedupe_review.html").read_text()
    dest = ROOT / "data/coruna/comparison_share/deduplication-review.html"
    dest.write_text(template.replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    print(json.dumps({"file": str(dest), "bytes": dest.stat().st_size, "cases": len(cases),
                      "images": len(images), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    build()
