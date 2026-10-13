import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from prepare_flux_training import prepare, train
from publish_huggingface import (archive_record, export_archive, export_graffiti, finish_package,
                                upload_package, validate_reviewed_dataset, verify_package, write_json)


def reviewed_fixture(root):
    root.mkdir()
    hashes = {}
    for split, lat, phash in [('train', 43.34, '0000000000000000'), ('validation', 43.36, '7fffffffffffffff')]:
        directory = root / split
        directory.mkdir()
        Image.new('RGB', (256, 256), 'blue').save(directory / 'crop.jpg')
        row = {'id': split, 'group_id': split, 'panorama_id': split, 'file_name': 'crop.jpg',
               'text': 'c0runa graffiti on a wall', 'source': 'apple', 'la': lat, 'lo': -8.4,
               'phash': phash}
        path = directory / 'metadata.jsonl'
        path.write_text(json.dumps(row) + '\n')
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(root / 'summary.json', {'reviewed': True, 'train': 1, 'validation': 1,
        'manifest_sha256': hashes, 'validation_buffer_metres': 100, 'minimum_confidence': .9,
        'snapshot': 'abc123', 'caption': 'c0runa graffiti on a wall'})
    return root


class PublishingTests(unittest.TestCase):
    def test_export_preserves_reviewed_bytes_and_split_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = reviewed_fixture(root / 'source')
            export_graffiti(source, root / 'package')
            for split in ('train', 'validation'):
                self.assertEqual((source / split / 'crop.jpg').read_bytes(),
                                 (root / 'package' / split / 'crop.jpg').read_bytes())
            (root / 'package/train/crop.jpg').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verify_package(root / 'package')

    def test_review_hash_inventory_and_spatial_gates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = reviewed_fixture(root / 'source')
            summary = json.loads((source / 'summary.json').read_text())
            summary['reviewed'] = False
            write_json(source / 'summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'Review'):
                validate_reviewed_dataset(source)
            summary['reviewed'] = True
            write_json(source / 'summary.json', summary)
            Image.new('RGB', (1, 1)).save(source / 'train/unlisted.jpg')
            with self.assertRaisesRegex(ValueError, 'Unlisted'):
                validate_reviewed_dataset(source)
            (source / 'train/unlisted.jpg').unlink()
            path = source / 'validation/metadata.jsonl'
            row = json.loads(path.read_text())
            row['la'] = 43.34
            path.write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'manifest hash'):
                validate_reviewed_dataset(source)
            summary['manifest_sha256']['validation'] = hashlib.sha256(path.read_bytes()).hexdigest()
            write_json(source / 'summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'buffer'):
                validate_reviewed_dataset(source)

    def test_unlisted_secret_is_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'README.md').write_text('fixture')
            finish_package(root, 'archive')
            (root / '.env').write_text('PRIVATE=fixture')
            with patch('huggingface_hub.HfApi') as api:
                with self.assertRaisesRegex(ValueError, 'unlisted'):
                    upload_package(root, 'fixture/test')
                api.assert_not_called()

    def test_archive_is_loadable_and_preserves_categories_ocr_geometry_and_crop_links(self):
        from datasets import load_dataset
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state = root / 'state'; state.mkdir()
            site = root / 'site'; site.mkdir()
            write_json(state / 'current.json', {'query_snapshot': 'abc123'})
            write_json(site / 'crop-storage.json', {'base_url': 'https://crops.example/crops/v1'})
            db = sqlite3.connect(state / 'abc123.sqlite3')
            db.executescript('CREATE TABLE records(id TEXT,payload TEXT); CREATE TABLE observations(id TEXT,group_key TEXT,category TEXT);')
            for category in ('graffiti', 'store_sign'):
                row = {'id': category, 'category': category, 'source': 'apple', 'score': .9,
                       'crop': '/proposal-assets/apple/run/view/crop.jpg', 'text': 'coruña' if category == 'store_sign' else None,
                       'bx': [0, .1, .5, .8]}
                db.execute('INSERT INTO records VALUES (?,?)', (category, json.dumps(row)))
                db.execute('INSERT INTO observations VALUES (?,?,?)', (category, category, category))
            db.commit(); db.close()
            export_archive(state, site, root / 'package', shard_rows=1)
            dataset = load_dataset('parquet', data_files=str(root / 'package/data/*/*.parquet'), split='train')
            self.assertEqual(len(dataset), 2)
            self.assertEqual(set(dataset['category']), {'graffiti', 'store_sign'})
            self.assertEqual(dataset[1]['ocr_text'], 'coruña')
            self.assertEqual(dataset[0]['bbox'], [0, .1, .5, .8])
            self.assertTrue(dataset[0]['crop_url'].startswith('https://crops.example/crops/v1/'))
            self.assertIsNone(archive_record({'id':'x', 'category':'x', 'source':'x', 'crop':'/../.env'}, 'x', 'https://example', None)['crop_url'])

    def test_flux_plan_uses_base_checkpoint_pinned_trainer_and_train_split(self):
        from datasets import load_dataset
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = reviewed_fixture(root / 'source')
            plan = prepare(source, root / 'flux')
            self.assertIn('klein-base-4B', plan['model'])
            self.assertEqual(len(plan['revision']), 40)
            self.assertIn('--bnb_quantization_config_path', plan['command'])
            self.assertNotIn('--push_to_hub', plan['command'])
            dataset = load_dataset(str(root / 'flux/dataset'))
            self.assertEqual(dataset['train'][0]['id'], 'train')
            self.assertEqual(dataset['validation'][0]['id'], 'validation')
            self.assertEqual(json.loads((root / 'flux/training.json').read_text())['stage'], 'prepared')

    def test_flux_refuses_changed_image_bytes_and_busy_gpu_before_starting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = reviewed_fixture(root / 'source')
            plan = prepare(source, root / 'flux')
            with patch('prepare_flux_training.subprocess.check_output',
                       side_effect=[plan['diffusers_revision'], '', '16000\n']), \
                 patch('prepare_flux_training.subprocess.run') as run:
                with self.assertRaisesRegex(ValueError, 'already in use'):
                    train(root / 'flux')
                run.assert_not_called()
            self.assertEqual(json.loads((root / 'flux/training.json').read_text())['stage'], 'prepared')
            Image.new('RGB', (256, 256), 'red').save(root / 'flux/dataset/train/crop.jpg')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                train(root / 'flux')


if __name__ == '__main__':
    unittest.main()
