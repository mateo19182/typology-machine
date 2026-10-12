import json
from pathlib import Path
import tempfile
import unittest

from export_layers import asset_url, export


class LayerExportTest(unittest.TestCase):
    def test_snapshot_keeps_multiple_objects_and_duplicate_flags_without_exposing_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'city'
            run = base / 'apple' / 'multicategory' / 'test-run'
            view = run / 'views' / 'p_1'
            view.mkdir(parents=True)
            (run / 'config.json').write_text(json.dumps({'prompts': {'graffiti': ['graffiti'], 'bench': ['bench']}, 'source': 'apple', 'model': 'sam3', 'threshold': .5}))
            predictions = []
            for index in range(3):
                crop = view / f'{index}.jpg'
                crop.write_bytes(b'jpg')
                predictions.append({'id': f'p-{index}', 'category_proposal': 'graffiti', 'score': .9, 'prompt': 'graffiti', 'box': [0, 0, 60, 60], 'crop': str(crop)})
            predictions[-1]['duplicate_of'] = 'p-0'
            result = {'metadata': {'lat': '43.3', 'lon': '-8.4', 'id': 'p', 'date': '2026-01-01', 'face': 1}, 'image_size': [100, 100], 'source': 'apple', 'view_id': 'p_1', 'predictions': predictions}
            (view / 'result.json').write_text(json.dumps(result))
            (view / 'features.pt').write_bytes(b'not public')
            (run / 'views' / 'unfinished').mkdir()
            out = base / 'site'
            out.mkdir()
            (out / 'data.json').write_text('legacy')
            group_folder = base / 'apple/multicategory_dedupe/pilot_v3'
            group_folder.mkdir(parents=True)
            (group_folder / 'config.json').write_text(json.dumps({'run_id': 'test-run'}))
            (group_folder / 'groups.jsonl').write_text(json.dumps({'id': 'group1', 'category': 'graffiti', 'rep': 'p-1', 'members': ['p-0', 'p-1']}) + '\n')

            manifest = export(base, run, out)
            self.assertEqual(manifest['completed_views'], 1)
            self.assertEqual(manifest['total_detections'], 3)
            self.assertFalse(manifest['deduplicated'])
            layer = manifest['categories'][0]
            self.assertEqual(layer['same_view_duplicate_flags'], 1)
            self.assertEqual(layer['object_count'], 1)
            features = json.loads((out / layer['url']).read_text())['features']
            self.assertEqual(len({f['properties']['id'] for f in features}), 3)
            self.assertEqual(features[0]['geometry']['coordinates'], [-8.4, 43.3])
            self.assertEqual(features[-1]['properties']['duplicate_of'], 'p-0')
            self.assertEqual(features[0]['properties']['duplicate_group'], features[1]['properties']['duplicate_group'])
            self.assertEqual(features[0]['properties']['group_rep'], 'p-1')
            self.assertEqual(len({f['properties']['raw'] for f in features}), 1)
            self.assertEqual(features[0]['properties']['bx'], [0, 0, .6, .6])
            asset = out / features[0]['properties']['crop'].lstrip('/')
            self.assertEqual(asset.read_bytes(), b'jpg')
            self.assertFalse((asset.parent / 'features.pt').exists())
            self.assertFalse(asset.parent.is_symlink())
            self.assertEqual((out / 'data.json').read_text(), 'legacy')
            self.assertEqual(manifest['categories'][1]['count'], 0)
            unchanged = export(base, run, out, if_changed=True)
            self.assertEqual(unchanged['snapshot_id'], manifest['snapshot_id'])
            (run / 'ocr.jsonl').write_text(json.dumps({'id': 'p-0', 'text': 'Rúa Real', 'model': 'test'}) + '\n' + '{"id":')
            refreshed = export(base, run, out, if_changed=True)
            self.assertNotEqual(refreshed['snapshot_id'], manifest['snapshot_id'])
            latest = json.loads((out / refreshed['categories'][0]['url']).read_text())['features']
            self.assertEqual(latest[0]['properties']['text'], 'Rúa Real')
            self.assertEqual(refreshed['ocr_records'], 1)
            self.assertTrue((out / layer['url']).exists())
            published = (out / 'layers/manifest.json').read_bytes()
            snapshots_before = set((out / 'layers').iterdir())
            (view / 'result.json').write_text('{broken')
            with self.assertRaises(json.JSONDecodeError):
                export(base, run, out, if_changed=True)
            self.assertEqual((out / 'layers/manifest.json').read_bytes(), published)
            self.assertEqual(set((out / 'layers').iterdir()), snapshots_before)

            with self.assertRaises(ValueError):
                asset_url(view / 'features.pt', base, out)
            with self.assertRaises(ValueError):
                asset_url(Path(tmp) / 'private.jpg', base, out)

    def test_supplemental_people_layer_survives_refresh_and_keeps_its_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'city'
            out = base / 'site'

            def make_run(name, category, count=1):
                run = base / 'apple' / 'multicategory' / name
                run.mkdir(parents=True, exist_ok=True)
                config = {'prompts': {category: [category]}, 'source': 'apple',
                          'model': 'sam3', 'threshold': .5}
                if category == 'person':
                    config['experiment'] = 'person-pilot-v1'
                (run / 'config.json').write_text(json.dumps(config))
                for index in range(count):
                    view = run / 'views' / f'view{index}'
                    view.mkdir(parents=True, exist_ok=True)
                    crop = view / 'crop.jpg'
                    crop.write_bytes(b'jpg')
                    result = {'source': 'apple', 'view_id': f'view{index}',
                              'image_size': [100, 100],
                              'metadata': {'lat': 43.3, 'lon': -8.4, 'date': '2023-05-10'},
                              'predictions': [{'id': f'{category}-{index}', 'category_proposal': category,
                                               'score': .9, 'prompt': category, 'box': [0, 0, 50, 90],
                                               'crop': str(crop)}]}
                    (view / 'result.json').write_text(json.dumps(result))
                return run

            main = make_run('main', 'graffiti')
            people = make_run('pilot', 'person')
            first = export(base, main, out, extra_runs=[people])
            self.assertEqual(first['total_detections'], 2)
            self.assertEqual(first['completed_views'], 1)
            self.assertEqual([run['completed_views'] for run in first['runs']], [1, 1])
            person = next(c for c in first['categories'] if c['id'] == 'person')
            self.assertEqual(person['label'], 'People')
            self.assertEqual(person['pilot'], 'person-pilot-v1')
            self.assertEqual(person['completed_views'], 1)
            item = json.loads((out / person['url']).read_text())['features'][0]['properties']
            self.assertEqual(item['raw'], '/proposal-context/apple/pilot/view0.jpg')
            self.assertEqual(item['date'], '2023-05-10')
            self.assertTrue((out / item['crop'].lstrip('/')).exists())
            unchanged = export(base, main, out, if_changed=True, extra_runs=[people])
            self.assertEqual(unchanged['snapshot_id'], first['snapshot_id'])
            make_run('main', 'graffiti', count=2)
            updated = export(base, main, out, if_changed=True, extra_runs=[people])
            self.assertNotEqual(updated['snapshot_id'], first['snapshot_id'])
            self.assertEqual({c['id']: c['count'] for c in updated['categories']},
                             {'graffiti': 2, 'person': 1})
            make_run('pilot', 'person', count=2)
            updated_pilot = export(base, main, out, if_changed=True, extra_runs=[people])
            self.assertNotEqual(updated_pilot['snapshot_id'], updated['snapshot_id'])
            self.assertEqual(updated_pilot['total_detections'], 4)
            published = (out / 'layers/manifest.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'Categories appear in multiple runs'):
                export(base, main, out, extra_runs=[main])
            self.assertEqual((out / 'layers/manifest.json').read_bytes(), published)


if __name__ == '__main__':
    unittest.main()
