"""Precision-first alternative: split legacy groups using all-pairs visual checks.

Does not overwrite production groups. Missing duplicates is intentional. Thresholds are
conservative heuristics, not a measured precision guarantee. Uses CPU and saved embeddings.
"""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

POLICY = {"version": 2, "min_similarity": .95, "competitor_margin": .03,
          "same_capture_date": True, "max_group_size": None,
          "max_baseline_m": 20, "camera_angle_filter": False, "triangulation_filter": False,
          "group_rule": "every pair must pass all checks", "min_sift_inliers": 20,
          "min_inlier_ratio": .85, "descriptor_ratio": .65, "min_coverage": .35,
          "max_reprojection_error_px": 2.5, "matching_region": "unpadded detection box",
          "candidate_scope": "all pairs within existing groups; no new cross-group merges"}


def complete_groups(members, edges, initial_groups=None):
    """Deterministic greedy clique partition; an A-B-C chain cannot create a group."""
    sets = {s: {s} for s in members}
    owner = {s: s for s in members}
    allowed = {frozenset((a,b)) for a,b,score in edges}
    if initial_groups:
        seeded=set()
        for group in initial_groups:
            if not group:
                continue
            if any(s not in owner or s in seeded for s in group):
                raise ValueError('Initial groups must be disjoint subsets of members')
            if any(frozenset((s,t)) not in allowed for i,s in enumerate(group) for t in group[i+1:]):
                raise ValueError('Every pair in an initial group must pass')
            first=group[0]
            for s in group[1:]:
                sets[first].update(sets.pop(s));owner[s]=first
            seeded.update(group)
    for a,b,score in sorted(edges, key=lambda e: (-e[2], min(e[:2]), max(e[:2]))):
        x,y=owner[a],owner[b]
        if x==y:
            continue
        if all(frozenset((i,j)) in allowed for i in sets[x] for j in sets[y]):
            for s in sets[y]: owner[s]=x
            sets[x].update(sets.pop(y))
    return [sorted(group) for group in sets.values()]


def feature_verifier(folder, rows, *, min_inlier_ratio=.85, min_coverage=.35):
    sift = cv2.SIFT_create(nfeatures=1600)

    @lru_cache(maxsize=512)
    def features(sid):
        r = rows[sid]
        image = cv2.imread(str(folder / r["crop"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        x0,y0,x1,y1 = r["box"]
        cx,cy,*_ = r["crop_box"]
        # Background padding must not supply the evidence for a sign match.
        image = image[max(0,int(y0-cy)):int(y1-cy), max(0,int(x0-cx)):int(x1-cx)]
        if not image.size or min(image.shape) < 32:
            return None
        scale = min(1., 768/max(image.shape))
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale)
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return np.float32([k.pt for k in keypoints]), descriptors, image.shape

    def verify(a, b):
        fa, fb = features(a), features(b)
        if fa is None or fb is None or fa[1] is None or fb[1] is None or min(len(fa[0]),len(fb[0])) < 20:
            return False, {"reason": "insufficient_local_features"}
        matcher = cv2.BFMatcher()
        def good(x,y):
            return {m.queryIdx:m.trainIdx for pair in matcher.knnMatch(x,y,k=2) if len(pair)==2
                    for m,n in [pair] if m.distance < .65*n.distance}
        ab, ba = good(fa[1],fb[1]), good(fb[1],fa[1])
        matches = [(i,j) for i,j in ab.items() if ba.get(j)==i]
        if len(matches)<20:
            return False, {"reason": "few_mutual_matches", "matches": len(matches)}
        x = np.float32([fa[0][i] for i,j in matches])
        y = np.float32([fb[0][j] for i,j in matches])
        H, mask = cv2.findHomography(x,y,cv2.RANSAC,2.5)
        if H is None or mask is None:
            return False, {"reason": "no_consistent_image_transform"}
        inliers = mask.ravel().astype(bool)
        def coverage(points, shape):
            return float(cv2.contourArea(cv2.convexHull(points))) / (shape[0]*shape[1])
        ca, cb = coverage(x[inliers],fa[2]), coverage(y[inliers],fb[2])
        # A match must cover the sign, not only a common logo or one word.
        ok = int(inliers.sum())>=20 and float(inliers.mean())>=min_inlier_ratio and min(ca,cb)>=min_coverage
        return bool(ok), {"reason": "local_match" if ok else "weak_or_partial_local_match",
                         "inliers": int(inliers.sum()), "inlier_ratio": float(inliers.mean()),
                         "coverage": [ca,cb]}
    return verify


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",choices=["apple","gsv"],default="apple")
    a=ap.parse_args()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    folder=Path("data/coruna")/a.source/"store_sign"
    original=(folder/"groups.jsonl").read_bytes()
    groups=[json.loads(l) for l in original.splitlines()]
    ids=json.loads((folder/"embeddings_ids.json").read_text())
    index={sid:i for i,sid in enumerate(ids)}
    E=np.load(folder/"embeddings.npy",mmap_mode="r")
    rows={}
    for line in (folder/"instances.jsonl").open():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        if r["id"] in index:rows[r["id"]]=r
    assert len(ids)==len(E) and all(s in rows for s in ids)
    by_pano={}
    for sid,r in rows.items():by_pano.setdefault(r["pano"],[]).append(sid)
    verify=feature_verifier(folder,rows)
    out=folder/"strict_dedupe_v2"
    out.mkdir(exist_ok=True)
    manifest={"policy":POLICY,"source":a.source,"original_groups_sha256":hashlib.sha256(original).hexdigest(),
              "opencv":cv2.__version__,"note":"Proposed groups only; original groups remain unchanged. No calibrated false-positive guarantee."}
    (out/"config.json").write_text(json.dumps(manifest,indent=2))
    evidence=[];result=[];counts=Counter()

    def output(members):
        rep=max(members,key=lambda sid:rows[sid]["score"]*min(rows[sid]["box"][2]-rows[sid]["box"][0],rows[sid]["box"][3]-rows[sid]["box"][1]))
        stable=hashlib.sha256((a.source+":"+":".join(sorted(members))).encode()).hexdigest()[:16]
        result.append({"sign":"p"+stable,"rep":rep,"members":members})

    for gi,g in enumerate(groups):
        members=g["members"]
        edges=[]
        if len(members)>1:
            e=np.asarray(E[[index[s] for s in members]])
            sims=e@e.T
            np.fill_diagonal(sims,-1)
            # Same-panorama and cross-date matches remain excluded.
            for i,s in enumerate(members):
                for j,t in enumerate(members):
                    if rows[s]["pano"]==rows[t]["pano"] or not rows[s].get("date") or rows[s].get("date")!=rows[t].get("date"):
                        sims[i,j]=-1
            for i,j in zip(*np.where(np.triu(sims >= .95, k=1))):
                s,t=members[i],members[j]
                score=float(sims[i,j]);record={"a":s,"b":t,"old_group":g["sign"],"similarity":score}
                # A competing object in either target panorama makes this pair ambiguous.
                margins=[]
                for x,y in [(s,t),(t,s)]:
                    rivals=[z for z in by_pano[rows[y]["pano"]] if z!=y]
                    rival=max((float(E[index[x]]@E[index[z]]) for z in rivals),default=-1)
                    margins.append(score-rival)
                record["competitor_margins"]=margins
                if min(margins)<.03:
                    record.update(accepted=False,reason="ambiguous_other_object")
                else:
                    ra,rb=rows[s],rows[t]
                    distance=math.hypot((ra['lon']-rb['lon'])*111320*math.cos(math.radians(ra['lat'])),
                                        (ra['lat']-rb['lat'])*111320)
                    record['camera_distance_m']=distance
                    if distance>20:record.update(accepted=False,reason="camera_distance")
                    else:
                        ok,local=verify(s,t)
                        record.update(accepted=ok,reason=local["reason"],local_features=local)
                evidence.append(record);counts[record["reason"]]+=1
                if record["accepted"]:
                    edges.append((s,t,score))
        for group in complete_groups(members,edges):output(group)
        if gi%10000==0:print(a.source,gi,"groups checked;",counts['local_match'],"passing pair checks",flush=True)
    assert sum(len(g["members"]) for g in result)==sum(len(g["members"]) for g in groups)
    assert len({s for g in result for s in g["members"]})==sum(len(g["members"]) for g in result)
    membership={s:g['sign'] for g in result for s in g['members']}
    for r in evidence:r['same_output_group']=membership[r['a']]==membership[r['b']]
    for name,values in [("groups.jsonl",result),("evidence.jsonl",evidence)]:
        path=out/name;temp=path.with_suffix(".tmp")
        temp.write_text("".join(json.dumps(r)+"\n" for r in values));temp.replace(path)
    summary={"source":a.source,"crops":sum(len(g["members"]) for g in result),"old_groups":len(groups),
             "new_groups":len(result),"multi_groups":sum(len(g['members'])>1 for g in result),
             "largest_group":max(len(g['members']) for g in result),
             "grouped_crops":sum(len(g['members']) for g in result if len(g['members'])>1),
             "passing_pairs":counts['local_match'],"reasons":dict(counts)}
    (out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
