"""Build a review of the larger, all-pairs-checked v2 groups."""
import base64
import argparse
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import random

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,choices=[2,3,4,5,6,7,8,9,10,11,12,13],default=2)
    version=parser.parse_args().version
    cases=[];images={};summaries=[];configs=[]
    for source in ('apple','gsv'):
        folder=ROOT/'data/coruna'/source/'store_sign';out=folder/f'strict_dedupe_v{version}'
        config=json.loads((out/'config.json').read_text());configs.append(config)
        summary=json.loads((out/'summary.json').read_text())
        summaries.append({'source':source,'crops':summary['crops'],'groups':summary['new_groups'],
            'singletons':summary['new_groups']-summary['multi_groups'],'largest':summary['largest_group']})
        groups=[json.loads(l) for l in (out/'groups.jsonl').read_text().splitlines()]
        groups=[g for g in groups if len(g['members'])>1]
        with (out/'evidence.jsonl').open() as evidence_file:
            edge={frozenset((r['a'],r['b'])):r for r in map(json.loads,evidence_file) if r['accepted'] and r['same_output_group']}
        if version>=3:
            old={frozenset(g['members']) for g in map(json.loads,(folder/f'strict_dedupe_v{version-1}/groups.jsonl').read_text().splitlines())}
            groups=[g for g in groups if frozenset(g['members']) not in old]
        groups.sort(key=lambda g:(-len(g['members']),g['sign']))
        if version==2:
            selected=groups[:40]
            selected+=random.Random(42).sample(groups[40:],min(40,len(groups[40:])))
        else:
            def checks_for(g):
                return [edge[frozenset((s,t))] for i,s in enumerate(g['members']) for t in g['members'][i+1:]]
            selected=groups[:20]
            for ranking in [sorted(groups,key=lambda g:min(e['similarity'] for e in checks_for(g))),
                            [g for g in groups if any(e.get('cross_legacy_group') for e in checks_for(g))]] + (
                            [sorted(groups,key=lambda g:-max(e['camera_distance_m'] for e in checks_for(g)))] if version>=8 else []) + (
                            [sorted(groups,key=lambda g:min(min(e['local_features']['coverage'])/.25 for e in checks_for(g))),
                             sorted(groups,key=lambda g:min(e['local_features']['inlier_ratio'] for e in checks_for(g)))] if version>=9 else []):
                used={g['sign'] for g in selected}
                selected.extend([g for g in ranking if g['sign'] not in used][:20])
            used={g['sign'] for g in selected};remaining=[g for g in groups if g['sign'] not in used]
            selected+=random.Random(42).sample(remaining,min(20,len(remaining)))
        wanted={s for g in selected for s in g['members']};rows={}
        for line in (folder/'instances.jsonl').open():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if r['id'] in wanted:rows[r['id']]=r
        for g in selected:
            members=[]
            checks=[edge[frozenset((s,t))] for i,s in enumerate(g['members']) for t in g['members'][i+1:]]
            assert all(r['accepted'] and r['same_output_group'] for r in checks)
            weakest=min(checks,key=lambda e:e['similarity']);extent=0
            for e in checks:
                a,b=rows[e['a']],rows[e['b']]
                extent=max(extent,math.hypot((a['lat']-b['lat'])*111320,(a['lon']-b['lon'])*111320*math.cos(math.radians(a['lat']))))
            for sid in [g['rep']]+[s for s in g['members'] if s!=g['rep']]:
                row=rows[sid];key=source+':'+sid
                if key not in images:
                    with Image.open(folder/row['crop']) as im:
                        im=im.convert('RGB');im.thumbnail((560,360));buf=BytesIO();im.save(buf,format='JPEG',quality=85)
                    images[key]='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()
                members.append({'image':key,'id':sid,'group':g['sign'],'pano':row['pano'],'date':row.get('date'),
                    'rep':sid==g['rep'],'lat':row['lat'],'lon':row['lon'],'view':row.get('face',row.get('view_yaw')),
                    'crop':str(folder/row['crop'])})
            cases.append({'id':source+':'+g['sign'],'source':source,'kind':'merged','group':g['sign'],
                'reason':f"Proposed v{version} group. All {len(checks)} pairs pass independently; no camera-angle or triangulation filter. All members are shown."+
                         (f" New or expanded since v{version-1}; {sum(bool(e.get('cross_legacy_group')) for e in checks)} matches cross old group boundaries." if version>=3 else ''),
                'total':len(members),'min_similarity':weakest['similarity'],'extent':extent,'same_pano':0,
                'weak_pair':[weakest['a'],weakest['b']],'members':members,
                'pair_evidence':checks})
    data={'snapshot':f'strict-v{version}-'+hashlib.sha256(json.dumps(configs,sort_keys=True).encode()).hexdigest()[:16],
          'cases':cases,'images':images,'summaries':summaries}
    template=(ROOT/'explore/dedupe_review.html').read_text()
    start=template.index('<details><summary>How the current');end=template.index('<p class="notice">',start)
    explanation='''<p><a href="strict-deduplication-review.html">Previous version: pairs with camera-angle checks</a></p>
<details open><summary>Version 2: larger groups, no camera-angle filter</summary>
<p>There is no group-size cap. Every pair in a group must pass the visual checks independently, preventing chain merges. Camera angle, ray intersections and inferred object height are not used.</p>
<p>Retained checks: DINO similarity ≥ 0.95; ≥ 0.03 margin over competing detections in the target panorama; ≥ 20 matching local image features; ≥ 85% feature agreement; ≥ 35% coverage of both objects. Matches use the unpadded detection boxes. All observations must come from different panoramas, on the same recorded capture date, with camera positions at most 20 m apart. Missing duplicates remains preferable to merging different objects.</p>
<p>All eligible pairs within the old groups are considered, replacing the previous mutual-nearest-neighbor restriction. We do not add matches between separate old groups. Groups are built deterministically, strongest matches first, and every cross-pair must pass before two groups join. A pair can pass its checks yet remain in different groups if joining them would introduce a failing pair.</p>
<p>This remains a proposed grouping; the production map and earlier review files are unchanged. No numerical precision guarantee is claimed. Mark any group containing distinct signs as “different” and identify the crops in the notes.</p></details>
<details><summary>Snapshot and sampling</summary><div id="stats"></div><p>Per source: the 40 largest multi-observation groups, plus up to 40 randomly selected remaining groups. Every member of each selected group is shown. This targets larger groups for review and is not a random accuracy estimate. Singletons remain in the saved output but are omitted here.</p></details>
'''
    template=template[:start]+explanation+template[end:]
    if version>=3:
        template=template.replace('strict-deduplication-review.html','strict-deduplication-v2.html')
        template=template.replace('Previous version: pairs with camera-angle checks','Previous version: stricter visual thresholds')
        template=template.replace('Version 2: larger groups, no camera-angle filter','Version 3: wider search, new and expanded groups')
        template=template.replace('≥ 0.95','≥ 0.92').replace('≥ 0.03','≥ 0.02').replace('20 m apart','35 m apart')
        template=template.replace('All eligible pairs within the old groups are considered, replacing the previous mutual-nearest-neighbor restriction. We do not add matches between separate old groups.',
            'All nearby eligible crops are considered, including pairs from different old groups. Existing v2 groups are preserved. Additions and group joins still require every pair to pass. Local feature thresholds are unchanged.')
        template=template.replace('Per source: the 40 largest multi-observation groups, plus up to 40 randomly selected remaining groups.',
            'Only groups that are new or expanded since v2 are shown. Per source: up to 20 largest, 20 with the weakest passing visual similarity, 20 crossing old group boundaries, and 20 random remaining groups, with no repeats. Totals describe the complete v3 output, including unchanged groups.')
    template=template.replace('Are these really the same object?','Deduplication v2: larger groups')
    template=template.replace('Review the saved Coruña sign groups.','Review the proposed v2 Coruña sign groups.')
    if version>=3:template=template.replace('Deduplication v2: larger groups','Deduplication v3: new and expanded groups').replace('proposed v2 Coruña','proposed v3 Coruña')
    if version>=4:
        template=template.replace('strict-deduplication-v2.html','strict-deduplication-v3.html').replace('Version 3:','Version 4:').replace('Deduplication v3:','Deduplication v4:').replace('proposed v3 Coruña','proposed v4 Coruña')
        template=template.replace('≥ 0.92','≥ 0.88').replace('≥ 0.02','≥ 0.01').replace('35 m apart','50 m apart').replace('Existing v2 groups','Existing v3 groups').replace('since v2','since v3').replace('complete v3 output','complete v4 output')
    if version>=5:
        template=template.replace('strict-deduplication-v3.html','strict-deduplication-v4.html').replace('Version 4:','Version 5:').replace('Deduplication v4:','Deduplication v5:').replace('proposed v4 Coruña','proposed v5 Coruña')
        template=template.replace('≥ 0.88','≥ 0.84').replace('≥ 0.01','≥ 0.005').replace('50 m apart','65 m apart').replace('Existing v3 groups','Existing v4 groups').replace('since v3','since v4').replace('complete v4 output','complete v5 output')
    if version>=6:
        template=template.replace('strict-deduplication-v4.html','strict-deduplication-v5.html').replace('Version 5:','Version 6:').replace('Deduplication v5:','Deduplication v6:').replace('proposed v5 Coruña','proposed v6 Coruña')
        template=template.replace('≥ 0.84','≥ 0.78').replace('≥ 0.005','≥ 0.002').replace('Existing v4 groups','Existing v5 groups').replace('since v4','since v5').replace('complete v5 output','complete v6 output')
    if version>=7:
        template=template.replace('strict-deduplication-v5.html','strict-deduplication-v6.html').replace('Version 6:','Version 7:').replace('Deduplication v6:','Deduplication v7:').replace('proposed v6 Coruña','proposed v7 Coruña')
        template=template.replace('≥ 0.78','≥ 0.70').replace('≥ 0.002 margin','≥ 0 margin').replace('Existing v5 groups','Existing v6 groups').replace('since v5','since v6').replace('complete v6 output','complete v7 output')
    if version>=8:
        template=template.replace('strict-deduplication-v6.html','strict-deduplication-v7.html').replace('Version 7:','Version 8:').replace('Deduplication v7:','Deduplication v8:').replace('proposed v7 Coruña','proposed v8 Coruña')
        template=template.replace('≥ 0.70','≥ 0.60').replace('65 m apart','100 m apart').replace('Existing v6 groups','Existing v7 groups').replace('since v6','since v7').replace('complete v7 output','complete v8 output')
    if version>=8:
        template=template.replace('20 crossing old group boundaries, and 20 random remaining groups', '20 crossing old group boundaries, 20 spanning the greatest camera distances, and 20 random remaining groups')
    if version>=9:
        template=template.replace('strict-deduplication-v7.html','strict-deduplication-v8.html').replace('Version 8:','Version 9:').replace('Deduplication v8:','Deduplication v9:').replace('proposed v8 Coruña','proposed v9 Coruña')
        template=template.replace('≥ 85% feature agreement','≥ 80% feature agreement').replace('≥ 35% coverage','≥ 25% coverage').replace('Existing v7 groups','Existing v8 groups').replace('since v7','since v8').replace('complete v8 output','complete v9 output')
        template=template.replace('Local feature thresholds are unchanged.', 'Local matching now requires 80% feature agreement and 25% coverage of each crop; the minimum remains 20 inliers. Saved geometric evidence is reevaluated against these thresholds.')
        template=template.replace('20 spanning the greatest camera distances, and 20 random remaining groups', '20 spanning the greatest camera distances, 20 with lowest local-feature coverage, 20 with lowest feature agreement, and 20 random remaining groups')
    if version>=10:
        template=template.replace('strict-deduplication-v8.html','strict-deduplication-v9.html').replace('Version 9:','Version 10:').replace('Deduplication v9:','Deduplication v10:').replace('proposed v9 Coruña','proposed v10 Coruña')
        template=template.replace('80% feature agreement','75% feature agreement').replace('25% coverage','20% coverage').replace('Existing v8 groups','Existing v9 groups').replace('since v8','since v9').replace('complete v9 output','complete v10 output')
    if version>=11:
        template=template.replace('strict-deduplication-v9.html','strict-deduplication-v10.html').replace('Version 10:','Version 11:').replace('Deduplication v10:','Deduplication v11:').replace('proposed v10 Coruña','proposed v11 Coruña')
        template=template.replace('75% feature agreement','70% feature agreement').replace('20% coverage','15% coverage').replace('Existing v9 groups','Existing v10 groups').replace('since v9','since v10').replace('complete v10 output','complete v11 output')
    if version>=12:
        template=template.replace('strict-deduplication-v10.html','strict-deduplication-v11.html').replace('Version 11:','Version 12:').replace('Deduplication v11:','Deduplication v12:').replace('proposed v11 Coruña','proposed v12 Coruña')
        template=template.replace('70% feature agreement','65% feature agreement').replace('15% coverage','10% coverage').replace('Existing v10 groups','Existing v11 groups').replace('since v10','since v11').replace('complete v11 output','complete v12 output')
    if version>=13:
        template=template.replace('strict-deduplication-v11.html','strict-deduplication-v12.html').replace('Version 12:','Version 13:').replace('Deduplication v12:','Deduplication v13:').replace('proposed v12 Coruña','proposed v13 Coruña')
        template=template.replace('65% feature agreement','50% feature agreement').replace('Existing v11 groups','Existing v12 groups').replace('since v11','since v12').replace('complete v12 output','complete v13 output')
    template=template.replace('Currently merged','Proposed groups').replace('<option value="separate">Currently separate</option>','')
    template=template.replace('Merged group ','Proposed group ')
    target=ROOT/f'data/coruna/comparison_share/strict-deduplication-v{version}.html'
    target.write_text(template.replace('__DATA__',json.dumps(data).replace('</','<\\/')))
    print(json.dumps({'file':str(target),'cases':len(cases),'images':len(images),'bytes':target.stat().st_size}))


if __name__=='__main__':main()
