"""Expand nearby matches while preserving the preceding version and all-pairs consistency."""
import argparse
from collections import Counter
import hashlib
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from strict_dedupe import POLICY as V2_POLICY, complete_groups, feature_verifier

POLICY={**V2_POLICY,'version':3,'min_similarity':.92,'competitor_margin':.02,'max_baseline_m':35,
        'candidate_scope':'all nearby crops, including across previous groups','preserve_v2_groups':True}


def make_parallel_verifier(folder,rows,policy=None):
    policy = policy or V2_POLICY
    worker_state=threading.local()
    def verify_pair(s,t):
        if not hasattr(worker_state,'verify'):
            worker_state.verify=feature_verifier(folder,rows,min_inlier_ratio=policy['min_inlier_ratio'],min_coverage=policy['min_coverage'])
        seed=int.from_bytes(hashlib.sha256(':'.join(sorted((s,t))).encode()).digest()[:4],'big') & 0x7fffffff
        cv2.setRNGSeed(seed)
        return worker_state.verify(s,t)
    return verify_pair


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',choices=['apple','gsv'],default='apple');ap.add_argument('--version',type=int,choices=[3,4,5,6,7,8,9,10,11,12,13],default=3);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    policy=dict(POLICY)
    if a.version==4:
        policy.update(version=4,min_similarity=.88,competitor_margin=.01,max_baseline_m=50,preserve_baseline_version=3)
        policy.pop('preserve_v2_groups',None)
    if a.version==5:
        policy.update(version=5,min_similarity=.84,competitor_margin=.005,max_baseline_m=65,preserve_baseline_version=4)
        policy.pop('preserve_v2_groups',None)
    if a.version==6:
        policy.update(version=6,min_similarity=.78,competitor_margin=.002,max_baseline_m=65,preserve_baseline_version=5)
        policy.pop('preserve_v2_groups',None)
    if a.version==7:
        policy.update(version=7,min_similarity=.70,competitor_margin=0.,max_baseline_m=65,preserve_baseline_version=6)
        policy.pop('preserve_v2_groups',None)
    if a.version==8:
        policy.update(version=8,min_similarity=.60,competitor_margin=0.,max_baseline_m=100,preserve_baseline_version=7)
        policy.pop('preserve_v2_groups',None)
    if a.version==9:
        policy.update(version=9,min_similarity=.60,competitor_margin=0.,max_baseline_m=100,preserve_baseline_version=8,min_inlier_ratio=.80,min_coverage=.25)
        policy.pop('preserve_v2_groups',None)
    if a.version==10:
        policy.update(version=10,min_similarity=.60,competitor_margin=0.,max_baseline_m=100,preserve_baseline_version=9,min_inlier_ratio=.75,min_coverage=.20)
        policy.pop('preserve_v2_groups',None)
    if a.version==11:
        policy.update(version=11,min_similarity=.60,competitor_margin=0.,max_baseline_m=100,preserve_baseline_version=10,min_inlier_ratio=.70,min_coverage=.15)
        policy.pop('preserve_v2_groups',None)
    if a.version==12:
        policy.update(version=12,min_similarity=.60,competitor_margin=0.,max_baseline_m=100,preserve_baseline_version=11,min_inlier_ratio=.65,min_coverage=.10)
        policy.pop('preserve_v2_groups',None)
    if a.version==13:
        policy.update(version=13,min_similarity=.60,competitor_margin=0.,max_baseline_m=100,preserve_baseline_version=12,min_inlier_ratio=.50,min_coverage=.10)
        policy.pop('preserve_v2_groups',None)
    baseline_version=a.version-1
    cv2.setNumThreads(1);cv2.setRNGSeed(42)
    folder=Path('data/coruna')/a.source/'store_sign';baseline=folder/f'strict_dedupe_v{baseline_version}';out=folder/f'strict_dedupe_v{a.version}';out.mkdir(exist_ok=True)
    original=(folder/'groups.jsonl').read_bytes();v2_bytes=(baseline/'groups.jsonl').read_bytes()
    groups=[json.loads(l) for l in original.splitlines()];v2=[json.loads(l) for l in v2_bytes.splitlines()]
    original_group={s:g['sign'] for g in groups for s in g['members']}
    ids=json.loads((folder/'embeddings_ids.json').read_text());index={sid:i for i,sid in enumerate(ids)}
    E=np.load(folder/'embeddings.npy',mmap_mode='r');rows={}
    with (folder/'instances.jsonl').open() as f:
        for line in f:
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if r['id'] in index:rows[r['id']]=r
    assert set(ids)==set(original_group)=={s for g in v2 for s in g['members']}==set(rows)
    xy=np.array([(rows[s]['lon']*111320*math.cos(math.radians(rows[ids[0]]['lat'])),rows[s]['lat']*111320) for s in ids])
    tree=cKDTree(xy);panos=np.array([rows[s]['pano'] for s in ids]);dates=np.array([rows[s].get('date','') for s in ids]);by_pano={}
    for i,p in enumerate(panos):by_pano.setdefault(p,[]).append(i)
    previous={frozenset((r['a'],r['b'])):r for r in map(json.loads,(baseline/'evidence.jsonl').read_text().splitlines())}
    evidence=[];edges=[];counts=Counter();seen=set()
    verify_pair=make_parallel_verifier(folder,rows,policy)
    pool=ThreadPoolExecutor(max_workers=max(1,a.workers));pending=deque()
    def record_result(record):
        evidence.append(record);counts[record['reason']]+=1
        if record['accepted']:edges.append((record['a'],record['b'],record['similarity']))
    def finish_one():
        record,future=pending.popleft();ok,local=future.result()
        record.update(accepted=ok,reason=local['reason'],local_features=local)
        record_result(record)
    # Preserve actual baseline groups; additions must satisfy every cross-pair.
    for g in v2:
        for i,s in enumerate(g['members']):
            for t in g['members'][i+1:]:
                key=frozenset((s,t));r=dict(previous[key]);assert r['accepted']
                r[f'preserved_v{baseline_version}']=True;r.setdefault('cross_legacy_group',False)
                evidence.append(r);edges.append((s,t,r['similarity']));seen.add(key);counts[f'preserved_v{baseline_version}']+=1
    manifest={'policy':policy,'source':a.source,'original_groups_sha256':hashlib.sha256(original).hexdigest(),
              f'v{baseline_version}_groups_sha256':hashlib.sha256(v2_bytes).hexdigest(),'opencv':cv2.__version__,'workers':a.workers,'new_local_match_rng':'sha256(sorted crop IDs), 31-bit',
              'note':f'Proposed output; production groups and prior versions remain intact. User approved reviewed v{baseline_version} examples, not a measured precision estimate.'}
    (out/'config.json').write_text(json.dumps(manifest,indent=2))
    for i,s in enumerate(ids):
        neighbors=np.array(tree.query_ball_point(xy[i],policy['max_baseline_m']),dtype=int)
        neighbors=neighbors[(neighbors>i)&(panos[neighbors]!=panos[i])&(dates[neighbors]==dates[i])]
        if dates[i] and len(neighbors):
            scores=np.asarray(E[neighbors])@E[i]
            for j,score in zip(neighbors[scores>=policy['min_similarity']],scores[scores>=policy['min_similarity']]):
                j=int(j);t=ids[j];key=frozenset((s,t))
                if key in seen:continue
                seen.add(key);score=float(score)
                record={'a':s,'b':t,'old_group':original_group[s], 'old_groups':[original_group[s],original_group[t]],
                        'cross_legacy_group':original_group[s]!=original_group[t], 'similarity':score,
                        'camera_distance_m':float(np.linalg.norm(xy[i]-xy[j]))}
                margins=[]
                for x,y in [(i,j),(j,i)]:
                    rivals=[z for z in by_pano[panos[y]] if z!=y]
                    rival=float(np.max(np.asarray(E[rivals])@E[x])) if rivals else -1
                    margins.append(score-rival)
                record['competitor_margins']=margins
                if min(margins)<policy['competitor_margin']:record.update(accepted=False,reason='ambiguous_other_object')
                else:
                    prior=previous.get(key,{})
                    if 'local_features' in prior:
                        local=dict(prior['local_features']);ok=prior['accepted']
                        if a.version>=9 and 'inliers' in local:
                            ok=local['inliers']>=policy['min_sift_inliers'] and local['inlier_ratio']>=policy['min_inlier_ratio'] and min(local['coverage'])>=policy['min_coverage']
                            local['reason']='local_match' if ok else 'weak_or_partial_local_match'
                        record['local_evidence_reused']=True
                    else:
                        pending.append((record,pool.submit(verify_pair,s,t)))
                        if len(pending)>=128:finish_one()
                        continue
                    record.update(accepted=ok,reason=local['reason'],local_features=local)
                record_result(record)
        if i%5000==0:print(a.source,i,'/',len(ids),'crops;',len(edges),'passing pairs',flush=True)
    while pending:finish_one()
    pool.shutdown()
    grouped=complete_groups(ids,edges,initial_groups=[g['members'] for g in v2]);result=[]
    for members in grouped:
        rep=max(members,key=lambda sid:rows[sid]['score']*min(rows[sid]['box'][2]-rows[sid]['box'][0],rows[sid]['box'][3]-rows[sid]['box'][1]))
        stable=hashlib.sha256((a.source+':'+':'.join(sorted(members))).encode()).hexdigest()[:16]
        result.append({'sign':'p'+stable,'rep':rep,'members':members})
    membership={s:g['sign'] for g in result for s in g['members']}
    assert len(membership)==sum(len(g['members']) for g in result)==len(ids)
    assert all(len({membership[s] for s in g['members']})==1 for g in v2)
    for r in evidence:r['same_output_group']=membership[r['a']]==membership[r['b']]
    for name,values in [('groups.jsonl',result),('evidence.jsonl',evidence)]:
        p=out/name;temp=p.with_suffix('.tmp');temp.write_text(''.join(json.dumps(r)+'\n' for r in values));temp.replace(p)
    summary={'source':a.source,'crops':len(ids),'old_groups':len(groups),'new_groups':len(result),
        'multi_groups':sum(len(g['members'])>1 for g in result),'largest_group':max(map(len,grouped)),
        'grouped_crops':sum(len(g['members']) for g in result if len(g['members'])>1),
        'passing_pairs':len(edges),'cross_legacy_group_matches':sum(r['accepted'] and r.get('cross_legacy_group',False) and r['same_output_group'] for r in evidence),'reasons':dict(counts)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
