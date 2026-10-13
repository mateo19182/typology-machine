"""CPU dedupe pilot for SAM categories; saves proposals and reusable SIFT descriptors.

No semantic embeddings or production merges. Samples nearby observations around a fixed
map center, then checks every eligible spatial pair within that sample. Extend --limit
using the same --out to reuse features/evidence and preserve previous proposal groups.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import itertools
import json
import os
import math
from pathlib import Path
import threading
import urllib.request

import cv2
import numpy as np
from scipy.spatial import cKDTree
from strict_dedupe import complete_groups

ROOT = Path(__file__).resolve().parents[1]
POLICY = dict(max_distance_m=100, same_capture_day=True, different_panorama=True,
              min_inliers=20, descriptor_ratio=.65, min_inlier_ratio=.5,
              min_coverage=.1, ransac_px=2.5, max_side=768, sift_features=1600,
              semantic_embeddings=False, camera_angle_filter=False,
              group_rule='every pair passes independently', version=1)


def atomic(path, text):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text);tmp.replace(path)


def rows_from_run(run, categories):
    rows=[]
    for path in sorted((run/'views').glob('*/result.json')):
        r=json.loads(path.read_text());m=r['metadata']
        for p in r['predictions']:
            if p['category_proposal'] not in categories:continue
            rows.append(dict(p,source=r['source'],view_id=r['view_id'],pano=str(m['id']),
                             lat=float(m['lat']),lon=float(m['lon']),date=str(m['date'])[:10],
                             raw=os.environ.get("TYPOLOGY_PUBLIC_URL", "http://127.0.0.1:8790").rstrip("/")+f"/proposal-context/{r['source']}/{run.name}/{r['view_id']}.jpg"))
    return rows


def select_sample(rows, category, limit, center):
    eligible=[r for r in rows if r['category_proposal']==category and not r.get('duplicate_of')]
    def distance(r):return (r['lat']-center[0])**2+((r['lon']-center[1])*math.cos(math.radians(center[0])))**2
    return sorted(eligible,key=lambda r:(distance(r),r['id']))[:limit]


def candidates(rows):
    if not rows:return []
    ref=rows[0]['lat'];xy=np.array([[r['lon']*111320*math.cos(math.radians(ref)),r['lat']*111320] for r in rows])
    pairs=[]
    for a,b in sorted(cKDTree(xy).query_pairs(POLICY['max_distance_m'])):
        x,y=rows[a],rows[b]
        if x['category_proposal']!=y['category_proposal'] or x['pano']==y['pano'] or not x['date'] or x['date']!=y['date']:continue
        pairs.append((x['id'],y['id'],float(np.linalg.norm(xy[a]-xy[b]))))
    return pairs


class Verifier:
    def __init__(self, rows, cache):
        self.rows=rows;self.cache=cache;cache.mkdir(parents=True,exist_ok=True);self.local=threading.local()

    @lru_cache(maxsize=1024)
    def features(self, sid):
        r=self.rows[sid];crop=Path(r['crop']);crop=crop if crop.is_absolute() else ROOT/crop
        stat=crop.stat();fingerprint=json.dumps([sid,r['box'],r['crop_box'],str(crop),stat.st_size,stat.st_mtime_ns,POLICY],sort_keys=True)
        key=hashlib.sha256(fingerprint.encode()).hexdigest();path=self.cache/(key+'.npz')
        if path.exists():
            with np.load(path) as d:return d['points'],d['descriptors'],d['shape']
        im=cv2.imread(str(crop),cv2.IMREAD_GRAYSCALE)
        if im is None:raise ValueError(f'Unreadable crop: {sid}')
        x0,y0,x1,y1=r['box'];cx,cy,*_=r['crop_box'];im=im[max(0,int(y0-cy)):int(y1-cy),max(0,int(x0-cx)):int(x1-cx)]
        if not im.size or min(im.shape)<32:points,desc,shape=np.empty((0,2),np.float32),np.empty((0,128),np.float32),np.array(im.shape)
        else:
            scale=min(1,768/max(im.shape))
            if scale<1:im=cv2.resize(im,None,fx=scale,fy=scale)
            if not hasattr(self.local,'sift'):self.local.sift=cv2.SIFT_create(nfeatures=1600)
            kp,desc=self.local.sift.detectAndCompute(im,None);points=np.float32([k.pt for k in kp]).reshape(-1,2);shape=np.array(im.shape)
            if desc is None:desc=np.empty((0,128),np.float32)
        # Atomic, thread-specific temporary names tolerate shared crops in concurrent pairs.
        temp=path.with_name(path.name+f'.{threading.get_ident()}.tmp')
        with temp.open('wb') as f:np.savez_compressed(f,points=points,descriptors=desc,shape=shape)
        temp.replace(path)
        return points,desc,shape

    def __call__(self, pair):
        a,b,distance=pair;fa,fb=self.features(a),self.features(b)
        record=dict(a=a,b=b,distance_m=distance,accepted=False)
        if min(len(fa[0]),len(fb[0]))<20:return dict(record,reason='insufficient_local_features')
        matcher=cv2.BFMatcher()
        def good(x,y):return {m.queryIdx:m.trainIdx for pair in matcher.knnMatch(x,y,k=2) if len(pair)==2 for m,n in [pair] if m.distance<.65*n.distance}
        ab,ba=good(fa[1],fb[1]),good(fb[1],fa[1]);matches=[(i,j)for i,j in ab.items()if ba.get(j)==i]
        if len(matches)<20:return dict(record,reason='few_mutual_matches',matches=len(matches))
        x=np.float32([fa[0][i]for i,j in matches]);y=np.float32([fb[0][j]for i,j in matches])
        cv2.setRNGSeed(int(hashlib.sha256(('\0'.join(sorted((a,b)))).encode()).hexdigest()[:8],16)&0x7fffffff)
        H,mask=cv2.findHomography(x,y,cv2.RANSAC,2.5)
        if H is None or mask is None:return dict(record,reason='no_transform')
        inliers=mask.ravel().astype(bool)
        coverage=[float(cv2.contourArea(cv2.convexHull(points[inliers])))/float(shape[0]*shape[1])for points,shape in [(x,fa[2]),(y,fb[2])]]
        ok=int(inliers.sum())>=20 and float(inliers.mean())>=POLICY['min_inlier_ratio'] and min(coverage)>=POLICY['min_coverage']
        return dict(record,accepted=ok,reason='local_match'if ok else'weak_or_partial_local_match',inliers=int(inliers.sum()),inlier_ratio=float(inliers.mean()),coverage=coverage)



def verify_context_geometry(fa, fb, boxa, boxb):
    """Require background correspondences to map the object to its counterpart.

    Object pixels are excluded from correspondence fitting, so identical sign designs
    cannot establish physical identity by themselves.
    """
    def background(features, box):
        points,desc=features[:2];x0,y0,x1,y1=box;dx=(x1-x0)*.1;dy=(y1-y0)*.1
        outside=(points[:,0]<x0-dx)|(points[:,0]>x1+dx)|(points[:,1]<y0-dy)|(points[:,1]>y1+dy)
        return points[outside],desc[outside]
    pa,da=background(fa,boxa);pb,db=background(fb,boxb)
    if min(len(pa),len(pb))<20:return dict(accepted=False,reason='insufficient_background_features')
    matcher=cv2.BFMatcher()
    def good(x,y):return {m.queryIdx:m.trainIdx for pair in matcher.knnMatch(x,y,k=2)if len(pair)==2 for m,n in[pair]if m.distance<.7*n.distance}
    ab,ba=good(da,db),good(db,da);matches=[(i,j)for i,j in ab.items()if ba.get(j)==i]
    if len(matches)<20:return dict(accepted=False,reason='few_background_matches',matches=len(matches))
    x=np.float32([pa[i]for i,j in matches]);y=np.float32([pb[j]for i,j in matches]);H,mask=cv2.findHomography(x,y,cv2.RANSAC,3)
    if H is None or mask is None:return dict(accepted=False,reason='no_background_transform')
    count=int(mask.sum());agreement=float(mask.mean());base=dict(background_inliers=count,background_agreement=agreement)
    if count<20 or agreement<.5:return dict(base,accepted=False,reason='weak_background_transform')
    def corners(box):
        x0,y0,x1,y1=box;return np.float32([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    try:
        mapped=cv2.perspectiveTransform(corners(boxa)[None],H)[0]
        reverse=cv2.perspectiveTransform(corners(boxb)[None],np.linalg.inv(H))[0]
    except (cv2.error,np.linalg.LinAlgError):return dict(base,accepted=False,reason='invalid_background_transform')
    if not np.isfinite(mapped).all()or not np.isfinite(reverse).all()or not cv2.isContourConvex(mapped):return dict(base,accepted=False,reason='invalid_projected_box')
    def alignment(poly,box):
        target=corners(box);intersection,_=cv2.intersectConvexConvex(poly,target);union=cv2.contourArea(poly)+cv2.contourArea(target)-intersection
        center=np.mean(poly,axis=0);targetcenter=np.mean(target,axis=0);dims=np.array([box[2]-box[0],box[3]-box[1]])
        return float(intersection/union)if union>0 else 0,float(np.max(np.abs(center-targetcenter)/dims))
    overlap,error=alignment(mapped,boxb);backoverlap,backerror=alignment(reverse,boxa)
    ok=min(overlap,backoverlap)>=.2 and max(error,backerror)<=.5
    return dict(base,accepted=ok,reason='context_aligned'if ok else'object_not_at_same_scene_position',projected_iou=[overlap,backoverlap],normalized_center_error=[error,backerror])


class ContextVerifier:
    def __init__(self, rows, run, cache):
        self.rows=rows;self.run=run;self.cache=cache;cache.mkdir(parents=True,exist_ok=True);self.local=threading.local()

    @lru_cache(maxsize=160)
    def features(self, view):
        key=hashlib.sha256((str(self.run.resolve())+'|'+view+'|1024|4000').encode()).hexdigest();path=self.cache/(key+'.npz')
        if path.exists():
            with np.load(path)as f:return f['points'],f['descriptors'],f['shape'],f['source_size']
        result=json.loads((self.run/'views'/view/'result.json').read_text())
        url=f"http://127.0.0.1:8793/proposal-context/{result['source']}/{self.run.name}/{view}.jpg"
        with urllib.request.urlopen(url,timeout=90)as response:content=response.read()
        image=cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_GRAYSCALE)
        if image is None:raise ValueError('Invalid context image')
        scale=min(1,1024/max(image.shape));image=cv2.resize(image,None,fx=scale,fy=scale)
        if not hasattr(self.local,'sift'):self.local.sift=cv2.SIFT_create(nfeatures=4000)
        kp,desc=self.local.sift.detectAndCompute(image,None);points=np.float32([k.pt for k in kp]).reshape(-1,2)
        if desc is None:desc=np.empty((0,128),np.float32)
        shape=np.array(image.shape);source_size=np.array(result['image_size']);tmp=path.with_name(path.name+f'.{threading.get_ident()}.tmp')
        with tmp.open('wb')as f:np.savez_compressed(f,points=points,descriptors=desc,shape=shape,source_size=source_size)
        tmp.replace(path);return points,desc,shape,source_size

    def __call__(self,e):
        a,b=self.rows[e['a']],self.rows[e['b']]
        try:
            fa,fb=self.features(a['view_id']),self.features(b['view_id'])
            def scaled(row,features):
                h,w=features[2];sw,sh=features[3];return np.array(row['box'])*np.array([w/sw,h/sh,w/sw,h/sh])
            cv2.setRNGSeed(int(hashlib.sha256(('context'+e['a']+e['b']).encode()).hexdigest()[:8],16)&0x7fffffff)
            evidence=verify_context_geometry(fa,fb,scaled(a,fa),scaled(b,fb))
        except (OSError,ValueError,cv2.error)as error:
            evidence=dict(accepted=False,reason='context_unavailable',error=type(error).__name__)
        return dict(e,crop_accepted=True,accepted=evidence['accepted'],reason=evidence['reason'],context=evidence)

def run(args):
    cv2.setNumThreads(1);out=args.out;out.mkdir(parents=True,exist_ok=True)
    allrows=([json.loads(line) for line in (args.reuse_evidence_from/'observations.jsonl').read_text().splitlines()] if args.reuse_evidence_from else rows_from_run(args.run,args.categories.split(',')));rows={r['id']:r for r in allrows}
    assert len(rows)==len(allrows)
    config=dict(policy=POLICY,run_id=args.run.name,categories=args.categories.split(','),sample_limit_per_category=args.limit,center=args.center,
                note='CPU geometric proposals only. No DINO embeddings, semantic competitor margin, human ground truth, or frontend merges. All observations retained, including excluded same-view duplicate flags. Sample is spatially concentrated, not a representative accuracy estimate.',opencv=cv2.__version__)
    if args.context_check:
        config['context_policy']=dict(max_side=1024,sift_features=4000,descriptor_ratio=.7,exclude_box_padding=.1,ransac_px=3,min_inliers=20,min_inlier_ratio=.5,min_projected_iou=.2,max_normalized_center_error=.5,require_both_directions=True)
    oldconfig=json.loads((out/'config.json').read_text())if(out/'config.json').exists()else None
    if oldconfig and (oldconfig['policy']!=POLICY or oldconfig['run_id']!=args.run.name):raise ValueError('Use a new output for a changed policy/run')
    oldgroups=[json.loads(l)for l in(out/'groups.jsonl').read_text().splitlines()]if(out/'groups.jsonl').exists()else []
    if any(s not in rows for g in oldgroups for s in g['members']):raise ValueError('Input lost previous observations')
    prior={frozenset((r['a'],r['b'])):r for r in map(json.loads,(out/'evidence.jsonl').read_text().splitlines())}if(out/'evidence.jsonl').exists()else {}
    if args.reuse_evidence_from:
        prior_config=json.loads((args.reuse_evidence_from/'config.json').read_text())
        changed={k for k in POLICY if prior_config['policy'].get(k)!=POLICY[k]}
        if changed-{'min_inlier_ratio','min_coverage','context_geometry'} or prior_config['run_id']!=args.run.name:
            raise ValueError('Cannot reuse geometry with changed feature extraction or candidate policy')
        prior={frozenset((r['a'],r['b'])):r for r in map(json.loads,(args.reuse_evidence_from/'evidence.jsonl').read_text().splitlines())}
        for e in prior.values():
            if 'inliers' in e:
                e['accepted']=e['inliers']>=20 and e['inlier_ratio']>=POLICY['min_inlier_ratio'] and min(e['coverage'])>=POLICY['min_coverage']
                e['reason']='local_match' if e['accepted'] else 'weak_or_partial_local_match'
            e['geometry_reused']=True
        config['geometry_reused_from']=str(args.reuse_evidence_from)
    selected=[];pairs=[]
    for category in config['categories']:
        sample=select_sample(allrows,category,args.limit,args.center);selected.extend(r['id']for r in sample);pairs.extend(candidates(sample))
    previous_selected=set(json.loads((out/'selected_ids.json').read_text()))if(out/'selected_ids.json').exists()else set()
    if not previous_selected.issubset(selected):raise ValueError('Sample changed: use a new output or increase limit without changing center/input')
    config['input_sha256']=hashlib.sha256(json.dumps(allrows,sort_keys=True).encode()).hexdigest()
    fresh=[p for p in pairs if frozenset(p[:2])not in prior];print(f'{len(allrows)} observations retained; {len(selected)} sampled; {len(pairs)} pairs; {len(fresh)} fresh',flush=True)
    verify=Verifier(rows,out/'sift_features');evidence=[prior[frozenset(p[:2])]for p in pairs if frozenset(p[:2])in prior]
    # Bounded batches prevent queued work or retained feature arrays growing with the run.
    with ThreadPoolExecutor(max_workers=args.workers)as pool:
        for start in range(0,len(fresh),256):
            evidence.extend(pool.map(verify,fresh[start:start+256]));print(f'{min(start+256,len(fresh))}/{len(fresh)} fresh pairs; {sum(e["accepted"]for e in evidence)} passing',flush=True)
    if args.context_check:
        verifier=ContextVerifier(rows,args.run,out/'context_features');passing=[e for e in evidence if e['accepted']]
        retained=[e for e in evidence if not e['accepted']]
        with ThreadPoolExecutor(max_workers=args.workers)as pool:
            for start in range(0,len(passing),16):
                retained.extend(pool.map(verifier,passing[start:start+16]));print(f'Context {min(start+16,len(passing))}/{len(passing)}; {sum(e["accepted"]for e in retained)} passing',flush=True)
        evidence=retained
    edges=[(e['a'],e['b'],e['inliers']*e['inlier_ratio']*min(e['coverage']))for e in evidence if e['accepted']]
    groups=complete_groups(sorted(rows),edges,[g['members']for g in oldgroups if len(g['members'])>1]);output=[]
    for members in groups:
        rep=max(members,key=lambda sid:rows[sid]['score']*min(rows[sid]['box'][2]-rows[sid]['box'][0],rows[sid]['box'][3]-rows[sid]['box'][1]))
        output.append(dict(id='m'+hashlib.sha256('\0'.join(members).encode()).hexdigest()[:16],category=rows[rep]['category_proposal'],rep=rep,members=members))
    owner={sid:g['id']for g in output for sid in g['members']}
    for e in evidence:e['same_output_group']=owner[e['a']]==owner[e['b']]
    summary=dict(observations=len(rows),sampled=len(selected),candidate_pairs=len(pairs),passing_pairs=len(edges),categories={})
    for category in config['categories']:
        gs=[g for g in output if g['category']==category];multi=[g for g in gs if len(g['members'])>1]
        summary['categories'][category]=dict(observations=sum(len(g['members'])for g in gs),sampled=sum(rows[s]['category_proposal']==category for s in selected),groups=len(gs),multi_groups=len(multi),grouped_observations=sum(len(g['members'])for g in multi),largest_group=max(map(lambda g:len(g['members']),gs),default=0))
    for name,value in [('config.json',config),('summary.json',summary),('selected_ids.json',selected)]:atomic(out/name,json.dumps(value,indent=2))
    for name,value in [('observations.jsonl',allrows),('groups.jsonl',output),('evidence.jsonl',evidence)]:atomic(out/name,''.join(json.dumps(r)+'\n'for r in value))
    print(json.dumps(summary,indent=2),flush=True);return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--categories',default='graffiti,street_name_plate');p.add_argument('--limit',type=int,default=800);p.add_argument('--center',nargs=2,type=float,default=[43.3665,-8.4065]);p.add_argument('--workers',type=int,default=4);p.add_argument('--min-inlier-ratio',type=float,default=.5);p.add_argument('--min-coverage',type=float,default=.1);p.add_argument('--reuse-evidence-from',type=Path);p.add_argument('--context-check',action='store_true');args=p.parse_args();POLICY.update(min_inlier_ratio=args.min_inlier_ratio,min_coverage=args.min_coverage,context_geometry=args.context_check);run(args)

if __name__=='__main__':main()
