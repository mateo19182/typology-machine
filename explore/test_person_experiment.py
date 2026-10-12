import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from person_experiment import prepare, pending_images


class PersonExperimentTests(unittest.TestCase):
    def test_pinned_config_and_distinct_sample_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'original'
            source.mkdir()
            (source/'config.json').write_text(json.dumps({'revision': 'pinned', 'prompts': {'store_sign': ['store sign']}, 'threshold': .5}))
            sample = root/'sample.json'
            payload = {'run': str(source), 'samples': [{'result': {'source': 'apple', 'view_id': 'a'}}]}
            sample.write_text(json.dumps(payload))
            _, config, out = prepare(sample, source)
            self.assertEqual(config['revision'], 'pinned')
            self.assertEqual(config['prompts'], {'person': ['person']})
            self.assertEqual(config['legacy_store_sign'], 'disabled')
            self.assertNotEqual(out, source)
            self.assertEqual(prepare(sample, source)[2], out)
            payload['samples'][0]['result']['view_id'] = 'b'
            sample.write_text(json.dumps(payload))
            self.assertNotEqual(prepare(sample, source)[2], out)

    def test_resume_skips_completed_view_even_if_source_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'views'/'done').mkdir(parents=True)
            (root/'views'/'done'/'result.json').write_text('{}')
            image = root/'source.jpg'
            Image.new('RGB', (64, 80)).save(image)
            samples = [{'result': {'view_id': 'done', 'source_path': 'missing'}},
                       {'result': {'view_id': 'next', 'source_path': str(image), 'image_size': [64, 80], 'metadata': {'date': '2020-01-01'}}}]
            rows = list(pending_images(samples, root, lambda: False))
            self.assertEqual([r[0] for r in rows], ['next'])
            self.assertEqual(rows[0][1]['date'], '2020-01-01')
            self.assertEqual(list(pending_images(samples, root, lambda: True)), [])
            samples[1]['result']['image_size'] = [80, 64]
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                list(pending_images(samples, root, lambda: False))


if __name__ == '__main__':
    unittest.main()
