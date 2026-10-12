import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from export_layers import feature
from proposal_context import ContextStore


class ContextTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name) / 'city'
        self.run = 'abcdef123456'
        self.view = 'pano_1'
        self.root = self.base / 'apple/multicategory' / self.run / 'views' / self.view
        self.root.mkdir(parents=True)
        self.source = self.base / 'apple/faces/pano_1.png'
        self.source.parent.mkdir()
        Image.new('RGB', (600, 1000), 'red').save(self.source)
        self.result = {'source': 'apple', 'view_id': self.view, 'source_path': str(self.source), 'image_size': [600, 1000], 'metadata': {'lat': 43, 'lon': -8, 'face': 1}}
        self.write_result()
        self.store = ContextStore(self.base, max_side=160)
        self.route = f'/proposal-context/apple/{self.run}/{self.view}.jpg'

    def tearDown(self):
        self.temp.cleanup()

    def write_result(self):
        (self.root / 'result.json').write_text(json.dumps(self.result))

    def test_aspect_ratio_geometry_and_cache(self):
        target = self.store.get(self.route)
        with Image.open(target) as image:
            self.assertEqual(image.size, (96, 160))
        with patch('proposal_context.Image.open', side_effect=AssertionError('must use cache')):
            self.assertEqual(self.store.get(self.route), target)
        pred = {'id': 'p', 'category_proposal': 'graffiti', 'score': .8, 'prompt': 'graffiti', 'box': [60, 200, 300, 800]}
        props = feature(self.result, pred, '/crop.jpg', self.run)['properties']
        self.assertEqual(props['raw'], self.route)
        self.assertEqual(props['bx'], [.1, .2, .5, .8])
        self.assertEqual(list(target.parent.glob('.context-*')), [])

    def test_missing_and_traversal(self):
        for route in ['/proposal-context/apple/../../private.jpg', '/proposal-context/apple/'+self.run+'/%2e%2e.jpg', '/proposal-context/apple/'+self.run+'/missing.jpg']:
            with self.assertRaises(FileNotFoundError):
                self.store.get(route)
        outside = Path(self.temp.name) / 'private.png'
        Image.new('RGB', (600, 1000)).save(outside)
        self.source.unlink()
        self.source.symlink_to(outside)
        with self.assertRaises(FileNotFoundError):
            self.store.get(self.route)

    def test_wrong_dimensions_and_missing_source(self):
        self.result['image_size'] = [1000, 600]
        self.write_result()
        with self.assertRaises(ValueError):
            self.store.get(self.route)
        self.source.unlink()
        with self.assertRaises(FileNotFoundError):
            self.store.get(self.route)

    def test_gsv_recuts_perspective(self):
        root = self.base / 'gsv/multicategory' / self.run
        view = root / 'views/gsv_90'
        view.mkdir(parents=True)
        source = self.base / 'gsv/panos/pano.jpg'
        source.parent.mkdir()
        Image.new('RGB', (400, 200), 'blue').save(source)
        result = {'source': 'gsv', 'view_id': 'gsv_90', 'source_path': str(source), 'image_size': [1600, 2048], 'metadata': {'view_yaw': 90, 'view_pitch': 8}}
        (view / 'result.json').write_text(json.dumps(result))
        (root / 'config.json').write_text(json.dumps({'gsv_projection': {'fov': [70, 55]}}))
        import numpy as np
        with patch('py360convert.e2p', return_value=np.zeros((160, 125, 3), dtype=np.uint8)) as project:
            target = self.store.get(f'/proposal-context/gsv/{self.run}/gsv_90.jpg')
        self.assertEqual(project.call_args.kwargs, {'fov_deg': (70, 55), 'u_deg': 90, 'v_deg': 8, 'out_hw': (160, 125), 'mode': 'bilinear'})
        with Image.open(target) as image:
            self.assertEqual(image.size, (125, 160))


if __name__ == '__main__':
    unittest.main()
