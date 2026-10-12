import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from backfill_people_pets import prepare, pending_images


class BackfillTests(unittest.TestCase):
    def test_separate_identity_and_immutable_originals(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'original'
            source.mkdir()
            config = {'model': 'model', 'revision': 'pinned', 'prompts': {'store_sign': ['store sign']}}
            (source / 'config.json').write_text(json.dumps(config))
            (source / 'additional_prompts.json').write_text(json.dumps({
                'schema_version': 1, 'model': 'model', 'revision': 'pinned',
                'prompts': {'person': ['person'], 'pet': ['dog', 'cat']}}))
            (source / 'person_pet_backfill.json').write_text(json.dumps({'views': ['b', 'a']}))
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            views, result, out = prepare(source)
            self.assertEqual(views, ['a', 'b'])
            self.assertEqual(set(result['prompts']), {'person', 'pet'})
            self.assertEqual(result['revision'], 'pinned')
            self.assertEqual(result['legacy_store_sign'], 'disabled')
            self.assertNotEqual(out, source)
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
            self.assertEqual(prepare(source)[2], out)

    def test_resume_and_refuse_repeated_categories(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, out = root / 'source', root / 'out'
            (out / 'views/done').mkdir(parents=True)
            (out / 'views/done/result.json').write_text('{}')
            (source / 'views/new').mkdir(parents=True)
            image = root / 'source.jpg'
            Image.new('RGB', (64, 80)).save(image)
            result = {'source': 'apple', 'view_id': 'new', 'source_path': str(image),
                      'metadata': {}, 'image_size': [64, 80], 'attempted_categories': ['store_sign']}
            path = source / 'views/new/result.json'
            path.write_text(json.dumps(result))
            rows = list(pending_images(source, ['done', 'new'], out, lambda: False))
            self.assertEqual([r[0] for r in rows], ['new'])
            result['attempted_categories'].append('person')
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'already attempted'):
                list(pending_images(source, ['new'], out, lambda: False))


if __name__ == '__main__':
    unittest.main()
