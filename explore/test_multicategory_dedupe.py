import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from multicategory_dedupe import Verifier, candidates, select_sample, verify_context_geometry

class DedupeTests(unittest.TestCase):
    def test_candidate_limits_and_same_panorama(self):
        def row(i,lat=43,date='2026-07-16',pano=None):return dict(id=i,pano=pano or i,lat=lat,lon=-8,date=date,category_proposal='graffiti')
        rows=[row('a'),row('b',lat=43.0002),row('c',lat=43.002),row('d',pano='a'),row('e',date='2026-07-17')]
        pairs={frozenset((a,b))for a,b,_ in candidates(rows)}
        self.assertIn(frozenset(('a','b')),pairs);self.assertNotIn(frozenset(('a','c')),pairs);self.assertNotIn(frozenset(('a','d')),pairs);self.assertFalse(any('e'in p for p in pairs))

    def test_local_evidence_and_persistent_features(self):
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp);rng=np.random.default_rng(41);im=rng.integers(0,256,(320,400),dtype=np.uint8)
            im=cv2.GaussianBlur(im,(3,3),0)
            for name,image in [('a',im),('b',im),('c',rng.integers(0,256,(320,400),dtype=np.uint8))]:cv2.imwrite(str(path/(name+'.jpg')),image)
            rows={n:dict(crop=str(path/(n+'.jpg')),box=[0,0,400,320],crop_box=[0,0,400,320])for n in 'abc'}
            v=Verifier(rows,path/'features');self.assertTrue(v(('a','b',10))['accepted']);self.assertFalse(v(('a','c',10))['accepted'])
            files=list((path/'features').glob('*.npz'));self.assertEqual(len(files),3);mtimes=[p.stat().st_mtime_ns for p in files]
            self.assertTrue(Verifier(rows,path/'features')(('a','b',10))['accepted']);self.assertEqual(mtimes,[p.stat().st_mtime_ns for p in files])

    def test_shared_padding_cannot_supply_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);rng=np.random.default_rng(6)
            background=rng.integers(0,256,(500,600),dtype=np.uint8)
            rows={}
            for name in ('a','b'):
                image=background.copy();image[200:300,250:350]=rng.integers(0,256,(100,100),dtype=np.uint8)
                cv2.imwrite(str(path/(name+'.jpg')),image)
                rows[name]=dict(crop=str(path/(name+'.jpg')),box=[250,200,350,300],crop_box=[0,0,600,500])
            self.assertFalse(Verifier(rows,path/'cache')(('a','b',10))['accepted'])

    def test_context_requires_same_physical_scene_position(self):
        rng=np.random.default_rng(20)
        image=cv2.GaussianBlur(rng.integers(0,256,(512,640),dtype=np.uint8),(3,3),0)
        other=cv2.GaussianBlur(rng.integers(0,256,(512,640),dtype=np.uint8),(3,3),0)
        box=[220,180,400,280]
        sign=rng.integers(0,256,(100,180),dtype=np.uint8)
        image[180:280,220:400]=sign;other[180:280,220:400]=sign
        moved=cv2.warpAffine(image,np.float32([[1,0,20],[0,1,10]]),(640,512))
        def features(im):
            points,desc=cv2.SIFT_create(nfeatures=4000).detectAndCompute(im,None)
            return np.float32([p.pt for p in points]),desc
        a,b,c=features(image),features(moved),features(other)
        self.assertTrue(verify_context_geometry(a,b,box,[240,190,420,290])['accepted'])
        self.assertFalse(verify_context_geometry(a,c,box,box)['accepted'])
        self.assertFalse(verify_context_geometry(a,a,box,[20,20,200,120])['accepted'])

    def test_sample_excludes_only_flagged_and_other_categories(self):
        rows=[dict(id='a',lat=43,lon=-8,category_proposal='graffiti'),dict(id='b',lat=43,lon=-8,category_proposal='graffiti',duplicate_of='a'),dict(id='c',lat=43,lon=-8,category_proposal='street_name_plate')]
        self.assertEqual([r['id']for r in select_sample(rows,'graffiti',10,[43,-8])],['a'])

if __name__=='__main__':unittest.main()
