import unittest
import numpy as np
from build_wander import neighbors

class WanderTest(unittest.TestCase):
    def test_neighbors_exclude_same_panorama_and_saved_group(self):
        rows=[dict(source='apple',view_id=pano,object_id=group) for pano,group in [('a_1','a'),('a_3','b'),('b_1','a'),('c_1','c')]]
        links=neighbors(np.eye(4,dtype=np.float32),rows)
        self.assertEqual(links[0],[3])
        self.assertNotIn(3,links[3])
        self.assertEqual(len(links),4)

if __name__=='__main__':unittest.main()
