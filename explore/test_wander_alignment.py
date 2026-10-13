import math
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from wander_alignment import FeatureData, detection_features, extract_features, fit_alignment


def synthetic_features(scale=1.06, angle=7, translation=(.025,-.018), spread=.4):
    rng = np.random.default_rng(19182)
    points = rng.uniform(-spread,spread,(64,2)).astype(np.float32)
    radians = math.radians(angle)
    linear = scale*np.asarray([[math.cos(radians),-math.sin(radians)], [math.sin(radians),math.cos(radians)]])
    transformed = (points@linear.T+translation).astype(np.float32)
    desc = rng.uniform(0,255,(64,128)).astype(np.float32)
    return FeatureData(transformed,desc.copy(),512,512), FeatureData(points,desc.copy(),512,512)


class AlignmentTests(unittest.TestCase):
    def test_detection_coordinates_remove_padding_and_edge_clipping(self):
        rng = np.random.default_rng(72)
        box = np.array([20., 30., 220., 130.])
        pixels = rng.uniform(box[:2]+2, box[2:]-2, (64,2))
        # Both paddings contain these background features, which must be discarded.
        pixels = np.vstack([pixels, [[10, 25], [225, 35], [100, 135]]])
        descriptors = rng.uniform(0, 255, (len(pixels),128)).astype(np.float32)
        def crop_features(bounds):
            bounds=np.asarray(bounds);size=bounds[2:]-bounds[:2]
            points=((pixels-(bounds[:2]+bounds[2:])/2)/max(size)).astype(np.float32)
            return FeatureData(points,descriptors.copy(),float(size[0]),float(size[1]))
        padded = [-4,18,244,142]
        clipped = [0,18,244,142]
        source=detection_features(crop_features(padded),box,padded)
        dest=detection_features(crop_features(clipped),box,clipped)
        self.assertEqual((source.width,source.height),(200,100))
        self.assertEqual(len(source.points),64)
        np.testing.assert_allclose(source.points,(pixels[:64]-[120,80])/200,atol=1e-7)
        np.testing.assert_allclose(dest.points,source.points,atol=1e-7)
        fit=fit_alignment(source,dest)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit['scale'],1,places=5)
        self.assertAlmostEqual(fit['x'],0,places=5)
        self.assertAlmostEqual(fit['y'],0,places=5)

    def test_detection_coordinate_validation(self):
        source,_=synthetic_features()
        for box,crop in [([0,0,512,512],[0,0,300,300]),
                         ([10,10,10,30],[0,0,512,512]),
                         ([-20,0,512,512],[0,0,512,512]),
                         ([0,0,float('nan'),512],[0,0,512,512]),
                         ([0,0,512],[0,0,512,512]),
                         ([250,250,251,251],[0,0,512,512])]:
            with self.subTest(box=box,crop=crop):
                self.assertEqual(len(detection_features(source,box,crop).points),0)

    def test_known_destination_to_source_contract(self):
        source,dest=synthetic_features()
        fit=fit_alignment(source,dest)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit['scale'],1.06,places=5)
        self.assertAlmostEqual(fit['angle'],7,places=4)
        self.assertAlmostEqual(fit['x'],.025,places=5)
        self.assertAlmostEqual(fit['y'],-.018,places=5)
        self.assertGreaterEqual(fit['coverage'],.15)
        self.assertEqual(fit['inliers'],64)

    def test_actual_sift_recovers_warped_image(self):
        rng=np.random.default_rng(19182)
        image=np.full((384,512),170,np.uint8)
        for _ in range(240):
            center=tuple(int(x) for x in rng.integers([12,12],[500,372]))
            cv2.circle(image,center,int(rng.integers(3,15)),int(rng.integers(0,255)),-1)
        cv2.putText(image,'TYPOLOGY',(40,180),cv2.FONT_HERSHEY_SIMPLEX,1.3,20,3)
        angle=math.radians(7);scale=1.04
        linear=scale*np.asarray([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        center=np.array([256,192]);translation=np.array([.025,-.018])
        warp=np.column_stack([linear,center-linear@center+translation*512])
        dest_image=cv2.warpAffine(image,warp,(512,384),borderValue=170)
        with tempfile.TemporaryDirectory() as directory:
            source_path=Path(directory)/'source.png';dest_path=Path(directory)/'dest.png'
            Image.fromarray(image).save(source_path);Image.fromarray(dest_image).save(dest_path)
            source=extract_features(source_path);dest=extract_features(dest_path)
            self.assertLessEqual(len(source.points),500)
            self.assertIs(source,extract_features(source_path))
            fit=fit_alignment(source,dest)
            self.assertIsNotNone(fit)
            self.assertAlmostEqual(fit['angle'],-7,delta=.4)
            self.assertAlmostEqual(fit['scale'],1/scale,delta=.01)
            expected=-np.linalg.inv(linear)@translation
            self.assertAlmostEqual(fit['x'],expected[0],delta=.005)
            self.assertAlmostEqual(fit['y'],expected[1],delta=.005)
            Image.new('L',(512,384),170).save(source_path)
            self.assertEqual(len(extract_features(source_path).points),0)
            self.assertEqual(len(extract_features(Path(directory)/'missing.png').points),0)

    def test_rejects_large_transforms_and_localised_features(self):
        for kwargs in [{'scale':1.5},{'scale':.65},{'angle':30},{'translation':(.16,.0)},{'spread':.06}]:
            with self.subTest(kwargs=kwargs):
                self.assertIsNone(fit_alignment(*synthetic_features(**kwargs)))

    def test_rejects_reflection_collinear_and_bad_consensus(self):
        source,dest=synthetic_features()
        reflected=FeatureData(dest.points*np.array([-1,1],np.float32),source.descriptors,512,512)
        self.assertIsNone(fit_alignment(reflected,dest))
        line=np.column_stack([np.linspace(-.4,.4,64),np.zeros(64)]).astype(np.float32)
        self.assertIsNone(fit_alignment(FeatureData(line,source.descriptors,512,512),FeatureData(line,dest.descriptors,512,512)))
        outliers=source.points.copy();outliers[:45]=np.random.default_rng(2).uniform(-.4,.4,(45,2))
        self.assertIsNone(fit_alignment(FeatureData(outliers,source.descriptors,512,512),dest))

    def test_unrelated_features_and_invalid_data(self):
        source,dest=synthetic_features()
        unrelated=np.random.default_rng(999).uniform(0,255,dest.descriptors.shape).astype(np.float32)
        self.assertIsNone(fit_alignment(source,FeatureData(dest.points,unrelated,512,512)))
        invalid=source.points.copy();invalid[0,0]=np.nan
        self.assertIsNone(fit_alignment(FeatureData(invalid,source.descriptors,512,512),dest))


if __name__=='__main__': unittest.main()
