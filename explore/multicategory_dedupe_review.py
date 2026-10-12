"""Build offline crop review for the multicategory CPU duplicate pilot."""
import argparse
import base64
import hashlib
from io import BytesIO
import itertools
import json
from pathlib import Path
import random
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def build(folder, destination):
    config=json.loads((folder/'config.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    rows={r['id']:r for r in map(json.loads,(folder/'observations.jsonl').read_text().splitlines())}
    groups=[g for g in map(json.loads,(folder/'groups.jsonl').read_text().splitlines())if len(g['members'])>1]
    edges={frozenset((e['a'],e['b'])):e for e in map(json.loads,(folder/'evidence.jsonl').read_text().splitlines())if e['accepted']and e['same_output_group']}
    cases=[];images={};summaries=[]
    def checks(g):return [edges[frozenset(pair)]for pair in itertools.combinations(g['members'],2)]
    for category,s in summary['categories'].items():
        available=[g for g in groups if g['category']==category];picked=[]
        rankings=[sorted(available,key=lambda g:-len(g['members'])),sorted(available,key=lambda g:min(min(e['coverage'])for e in checks(g))),sorted(available,key=lambda g:min(e['inlier_ratio']for e in checks(g))),sorted(available,key=lambda g:-max(e['distance_m']for e in checks(g)))]
        for ranked in rankings:
            ids={g['id']for g in picked};picked.extend([g for g in ranked if g['id']not in ids][:20])
        remaining=[g for g in available if g not in picked];picked+=random.Random(42).sample(remaining,min(20,len(remaining)))
        summaries.append(dict(source=category,crops=s['observations'],groups=s['groups'],singletons=s['groups']-s['multi_groups'],largest=s['largest_group']))
        for g in picked:
            evidence=checks(g);weak=min(evidence,key=lambda e:min(e['coverage']));members=[]
            for sid in [g['rep']]+[i for i in g['members']if i!=g['rep']]:
                r=rows[sid]
                with Image.open(ROOT/r['crop'])as im:
                    im=im.convert('RGB');im.thumbnail((560,360));buf=BytesIO();im.save(buf,format='JPEG',quality=85)
                images[sid]='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()
                members.append(dict(image=sid,id=sid,group=g['id'],pano=r['pano'],date=r['date'],rep=sid==g['rep'],lat=r['lat'],lon=r['lon'],view=r['view_id'],raw=r['raw']))
            cases.append(dict(id=g['id'],group=g['id'],source=category,kind='merged',total=len(members),members=members,same_pano=0,weak_pair=[weak['a'],weak['b']],min_similarity=min(e['inlier_ratio']for e in evidence),extent=max(e['distance_m']for e in evidence),pair_evidence=evidence,
                reason=f"{category}: all {len(evidence)} pairs independently verified. Minimum local coverage {min(min(e['coverage'])for e in evidence):.1%}. Same capture day; different panoramas; no semantic embedding filter. Every group member is shown."))
    data=dict(snapshot='multicategory-pilot-'+hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16],cases=cases,images=images,summaries=summaries)
    template=(ROOT/'explore/dedupe_review.html').read_text();start=template.index('<p>Review the saved');end=template.index('<p class="notice">',start)
    intro=f'''<p>Graffiti and street-name duplicate proposals from the new SAM detections. Judge physical identity, including whether separate graffiti pieces were accidentally joined. All group members are shown.</p>
<details open><summary>CPU pilot: what was checked</summary><p>{summary['sampled']:,} observations selected near the map center, {config['sample_limit_per_category']} per category. Every candidate pair within 100 m inside that sample was checked. Different panoramas, same recorded capture day, at least 20 reciprocal SIFT matches, {config['policy']['min_inlier_ratio']:.0%} geometric agreement and {config['policy']['min_coverage']:.0%} coverage of each unpadded detection. Every pair in a group must pass. No camera-angle filter.</p><p>No DINO or Qwen semantic embeddings were computed. Saved local SIFT descriptors and pair evidence can be reused to extend this pilot. Matching templates and separate tags can still produce false positives; these category proposals need their own review.</p><p>All {summary['observations']:,} observations remain saved. Unsampled observations and same-view duplicate flags remain separate. This review does not collapse observations on the frontend.</p></details><details><summary>Results and sample</summary><div id="stats"></div><p>Per category: largest groups, weakest local coverage and agreement, widest camera spans, then random remaining groups. This is a targeted review, not an accuracy estimate.</p></details>'''
    if config['policy'].get('context_geometry'):
        intro += '<p><b>Additional physical-location check:</b> matching features outside the detection must establish a shared-scene transform, and that transform must align the object boxes in both directions. At least 20 background inliers, 50% agreement, 20% projected overlap and a center inside the counterpart box. This rejects identical designs at different places. Context rejection can miss legitimate duplicates.</p>'
    template=template[:start]+intro+template[end:]
    template=template.replace('Source <select','Category <select').replace('<option value="all">Both sources</option><option>apple</option><option>gsv</option>','<option value="all">Both categories</option><option>graffiti</option><option>street_name_plate</option>')
    template=template.replace('weakest pair similarity ${c.min_similarity.toFixed(3)}','lowest feature agreement ${(100*c.min_similarity).toFixed(1)}%')
    template=template.replace("$('cards').append(card)","const link=document.createElement('a');link.href=m.raw;link.target='_blank';link.rel='noopener';link.textContent='Full street image ↗';card.append(link);$('cards').append(card)")
    template=template.replace('__DATA__',json.dumps(data,separators=(',',':')).replace('</','<\\/'))
    destination.write_text(template);print(f'{len(cases)} groups; {len(images)} crop images; {destination}')
    return data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.input,a.output)
