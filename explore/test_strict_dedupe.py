import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from strict_dedupe import feature_verifier, complete_groups
from wider_dedupe import make_parallel_verifier


class StrictDedupeTests(unittest.TestCase):
    def test_preserves_baseline_group_despite_stronger_competing_edge(self):
        edges=[('a','b',.96),('b','c',.99)]
        self.assertEqual(complete_groups(['a','b','c'],edges,[['a','b'],['c']]),[['a','b'],['c']])
        with self.assertRaises(ValueError):
            complete_groups(['a','b','c'],edges,[['a','c']])

    def test_chain_is_not_a_group(self):
        self.assertEqual(complete_groups(['a','b','c'], [('a','b',.99),('b','c',.98)]), [['a','b'],['c']])

    def test_every_pair_can_form_a_larger_group(self):
        self.assertEqual(complete_groups(['a','b','c'], [('a','b',.99),('b','c',.98),('a','c',.97)]), [['a','b','c']])

    def test_competing_groups_preserve_every_pair_requirement(self):
        edges=[('a','b',.99),('b','c',.98),('a','c',.97),('c','d',.99)]
        result=complete_groups(['a','b','c','d'],edges)
        allowed={frozenset((a,b)) for a,b,_ in edges}
        self.assertEqual(sorted(s for group in result for s in group),['a','b','c','d'])
        self.assertTrue(all(frozenset((a,b)) in allowed for group in result
                            for i,a in enumerate(group) for b in group[i+1:]))
        self.assertEqual(result,complete_groups(['a','b','c','d'],list(reversed(edges))))

    def test_local_match_requires_shared_detail_across_object(self):
        cv2.setNumThreads(1)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);rng=np.random.default_rng(8)
            image=rng.integers(0,256,(400,400),dtype=np.uint8)
            different=rng.integers(0,256,(400,400),dtype=np.uint8)
            rows={}
            for sid,im in [("a",image),("b",image),("c",different)]:
                cv2.imwrite(str(folder/f"{sid}.png"),im)
                rows[sid]={"crop":f"{sid}.png","box":[0,0,400,400],"crop_box":[0,0,400,400]}
            verify=feature_verifier(folder,rows)
            self.assertTrue(verify("a","b")[0])
            self.assertFalse(verify("a","c")[0])
            parallel_verify=make_parallel_verifier(folder,rows)
            pairs=[('a','b'),('a','c'),('b','a'),('c','a')]*3
            expected=[parallel_verify(*pair) for pair in pairs]
            with ThreadPoolExecutor(max_workers=4) as pool:
                actual=list(pool.map(lambda pair:parallel_verify(*pair),pairs))
            self.assertEqual(expected,actual)


if __name__=="__main__":unittest.main()
