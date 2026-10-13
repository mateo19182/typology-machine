import unittest
import numpy as np
from PIL import Image
from build_wander import neighbors, palette


def rows(count):
    return [dict(source='apple',view_id=f'{i}_1',object_id=str(i),aspect=1.) for i in range(count)]

class WanderTest(unittest.TestCase):
    def test_neighbors_exclude_same_panorama_and_saved_group(self):
        r=rows(4);r[1]['view_id']='0_3';r[2]['object_id']='0'
        links,_,_=neighbors(np.ones((4,1),dtype=np.float32),r)
        self.assertEqual(links[0],[3])
        self.assertNotIn(3,links[3])

    def test_weak_and_differently_proportioned_matches_are_not_forced(self):
        r=rows(4);r[2]['aspect']=2.
        v=np.array([[1,0],[.5,np.sqrt(.75)],[.95,np.sqrt(1-.95**2)],[.8,.6]])
        links,_,_=neighbors(v,r)
        self.assertEqual(links[0],[3])
        self.assertEqual(neighbors(np.eye(3),rows(3))[0],[[],[],[]])

    def test_colour_reranks_only_visually_eligible_candidates(self):
        v=np.array([[1,0],[.9,np.sqrt(.19)],[.88,np.sqrt(1-.88**2)],[.5,np.sqrt(.75)]])
        p=np.array([[1,0],[0,1],[1,0],[1,0]],dtype=float)
        shape,color,_=neighbors(v,rows(4),palettes=p)
        self.assertEqual(shape[0],[1,2]);self.assertEqual(color[0],[2,1])
        self.assertNotIn(3,color[0])

    def test_palette_distinguishes_colour_and_brightness(self):
        palettes=[palette(Image.new('RGB',(50,40),color)) for color in ['red','blue','black','white']]
        for i,a in enumerate(palettes):
            self.assertAlmostEqual(float(a@a),1.)
            for b in palettes[i+1:]:self.assertLess(float(a@b),.5)

if __name__=='__main__':unittest.main()
