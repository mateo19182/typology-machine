"""Package all proposed strict pairs, plus rejected examples, into the offline review UI."""
import base64
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def main():
    cases=[];images={};summaries=[];configs=[]
    for source in ("apple","gsv"):
        folder=ROOT/"data/coruna"/source/"store_sign"
        output=folder/"strict_dedupe_v1"
        configs.append(json.loads((output/"config.json").read_text()))
        summary=json.loads((output/"summary.json").read_text())
        summaries.append({"source":source,"crops":summary["crops"],"groups":summary["new_groups"],
                          "singletons":summary["crops"]-2*summary["accepted_pairs"],"largest":2})
        evidence=[json.loads(l) for l in (output/"evidence.jsonl").read_text().splitlines()]
        selected=[r for r in evidence if r["accepted"]]
        for reason in sorted({r["reason"] for r in evidence if not r["accepted"]}):
            selected.extend(sorted((r for r in evidence if r["reason"]==reason),key=lambda r:r["similarity"],reverse=True)[:2])
        wanted={r[k] for r in selected for k in ("a","b")}
        rows={}
        for line in (folder/"instances.jsonl").open():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if r["id"] in wanted:rows[r["id"]]=r
        for e in selected:
            members=[]
            for sid in (e["a"],e["b"]):
                row=rows[sid];key=source+":"+sid
                if key not in images:
                    with Image.open(folder/row["crop"]) as im:
                        im=im.convert("RGB");im.thumbnail((640,420));buf=BytesIO();im.save(buf,format="JPEG",quality=88)
                    images[key]="data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode()
                members.append({"image":key,"id":sid,"group":"previously "+e["old_group"],
                    "pano":row["pano"],"date":row.get("date"),"rep":False,"lat":row["lat"],"lon":row["lon"],
                    "view":row.get("face",row.get("view_yaw")),"crop":str(folder/row["crop"])})
            a,b=members
            distance=math.hypot((a["lon"]-b["lon"])*111320*math.cos(math.radians(a["lat"])),(a["lat"]-b["lat"])*111320)
            if e["accepted"]:
                f=e["local_features"]
                reason=f"PROPOSED PAIR, not human-verified. {f['inliers']} local-feature inliers, {f['inlier_ratio']:.0%} agreement, object coverage {f['coverage'][0]:.0%}/{f['coverage'][1]:.0%}. Previously in {e['old_group']}."
            else:reason=f"KEPT SEPARATE by strict policy: {e['reason'].replace('_',' ')}. Previously merged in {e['old_group']}."
            cases.append({"id":source+":"+e["a"]+":"+e["b"],"source":source,
                          "kind":"merged" if e["accepted"] else "separate","reason":reason,
                          "group":"from "+e["old_group"],"total":2,"min_similarity":e["similarity"],"extent":distance,
                          "similarity":e["similarity"],"distance":distance,"same_pano":0,"members":members,
                          "evidence":e})
    snapshot="strict-"+hashlib.sha256(json.dumps(configs,sort_keys=True).encode()).hexdigest()[:16]
    data={"snapshot":snapshot,"cases":cases,"images":images,"summaries":summaries}
    template=(ROOT/"explore/dedupe_review.html").read_text()
    start=template.index('<details><summary>How the current')
    end=template.index('<p class="notice">',start)
    explanation='''<details open><summary>Strict policy: false merges matter more than missed duplicates</summary>
<p>This is a proposed replacement, not the production map's current grouping. Every proposed automatic pair is included below. The old groups remain saved. The large Apple group s18500 becomes 250 separate observations.</p>
<p>Pairs must be mutual best visual matches within an old group, with DINO cosine similarity ≥ 0.95, and a ≥ 0.03 margin over other detections in the target panorama. They must come from different panoramas on the same recorded capture date, have compatible forward viewing rays, and agree in at least 20 local image-feature matches. Those matches must cover ≥ 35% of each detected object and have ≥ 85% geometric agreement. Background padding is excluded from feature matching.</p>
<p>Groups have at most two members. There are no transitive merges, no cross-date or cross-source merges, and no new merges between old groups. All other crops remain separate. Geometry is a rejection filter: two intersecting rays alone do not establish identity. Repeated identical signs can still fool image matching. These thresholds have not been calibrated on human-reviewed pairs, so no numerical precision claim is justified.</p>
<p>Start with “Proposed pairs.” Mark any false pair as different and note why. Rejected examples are included to show the deliberate tradeoff: some real duplicates will stay separate. If the accepted pairs still contain errors, require human approval before suppressing any observation. All decisions remain reversible.</p>
</details><details><summary>Proposed grouping totals</summary><div id="stats"></div><p>All proposed pairs from both saved datasets are included, plus up to two rejected candidates per rejection reason. This is not a random accuracy sample; review every proposed pair to assess this frozen run.</p></details>
'''
    template=template[:start]+explanation+template[end:]
    template=template.replace('Are these really the same object?','Strict deduplication: inspect every proposed pair')
    template=template.replace('Review the saved Coruña sign groups.','Review the proposed conservative replacement for the Coruña sign groups.')
    template=template.replace('Currently merged','Proposed pairs').replace('Currently separate','Rejected pairs')
    template=template.replace('Merged group ','Proposed pair ').replace("'Separate groups'","'Kept separate'")
    path=ROOT/'data/coruna/comparison_share/strict-deduplication-review.html'
    path.write_text(template.replace('__DATA__',json.dumps(data).replace('</','<\\/')))
    print(json.dumps({"file":str(path),"cases":len(cases),"proposed_pairs":sum(c['kind']=='merged' for c in cases),"bytes":path.stat().st_size}))


if __name__=='__main__':main()
