import hashlib
from argparse import Namespace
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import numpy as np

from prepare_flux_training import prepare, train
from prepare_graffiti_training import prepare as prepare_candidates
from publish_huggingface import (archive_record, export_archive, export_graffiti, finish_package,
                                publish_run, upload_package, validate_reviewed_dataset, verify_package, write_json)


def reviewed_fixture(root):
    root.mkdir()
    hashes = {}
    for split, lat, phash in [('train', 43.34, '0000000000000000'), ('validation', 43.36, '7fffffffffffffff')]:
        directory = root / split
        directory.mkdir()
        Image.new('RGB', (256, 256), 'blue').save(directory / 'crop.jpg')
        row = {'id': split, 'group_id': split, 'panorama_id': split, 'file_name': 'crop.jpg',
               'text': 'c0runa graffiti on a wall', 'source': 'apple', 'la': lat, 'lo': -8.4,
               'phash': phash, 'score': .96}
        path = directory / 'metadata.jsonl'
        path.write_text(json.dumps(row) + '\n')
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(root / 'summary.json', {'reviewed': True, 'train': 1, 'validation': 1,
        'manifest_sha256': hashes, 'validation_buffer_metres': 100, 'minimum_confidence': .9,
        'snapshot': 'abc123', 'caption': 'c0runa graffiti on a wall'})
    return root


class PublishingTests(unittest.TestCase):
    def test_rebuilding_manifest_never_checksums_itself(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'README.md').write_text('fixture')
            finish_package(root, 'archive')
            (root / 'review.json').write_text('{}')
            finish_package(root, 'archive')
            manifest = verify_package(root)
            self.assertNotIn('package.json', manifest['files'])
            self.assertIn('review.json', manifest['files'])

    def test_expansion_preserves_reviewed_splits_bytes_and_prior_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = reviewed_fixture(root / 'source')
            write_json(source / 'review-exclusions.json', [{'id': 'bad-sign', 'reason': 'commercial sign'}])
            state = root / 'state'; state.mkdir()
            site = root / 'site'; site.mkdir()
            write_json(state / 'current.json', {'query_snapshot': 'abc123'})
            db = sqlite3.connect(state / 'abc123.sqlite3')
            db.executescript('CREATE TABLE records(id TEXT,payload TEXT); CREATE TABLE observations(id TEXT,group_key TEXT,category TEXT,score REAL,quality REAL);')
            rng = np.random.default_rng(19182)
            for index, identity in enumerate(('bad-sign', 'new-validation', 'new-training')):
                filename = identity + '.jpg'
                Image.fromarray(rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)).save(site / filename)
                row = {'id': identity, 'source': 'apple', 'view_id': identity + '_0', 'score': .95-index*.01,
                       'width': 400, 'height': 300, 'crop': filename, 'la': 43.38+index*.02, 'lo': -8.4}
                db.execute('INSERT INTO records VALUES (?,?)', (identity, json.dumps(row)))
                db.execute('INSERT INTO observations VALUES (?,?,?,?,?)', (identity, identity, 'graffiti', row['score'], 1))
            db.commit(); db.close()
            output = root / 'expanded'
            prepare_candidates(Namespace(output=output, state=state, site=site, expand_from=source,
                train=2, validation=2, confidence=.85, spacing=20))
            for split in ('train', 'validation'):
                self.assertEqual((source / split / 'crop.jpg').read_bytes(), (output / split / 'crop.jpg').read_bytes())
                rows = [json.loads(x) for x in (output / split / 'metadata.jsonl').read_text().splitlines()]
                self.assertEqual(rows[0]['id'], split)
                self.assertEqual(len(rows), 2)
                self.assertNotIn('bad-sign', {r['id'] for r in rows})
            self.assertFalse(json.loads((output / 'summary.json').read_text())['reviewed'])

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
            self.assertNotIn('--caption_column', plan['command'])
            self.assertEqual(plan['caption_strategy'], 'shared prompt encoded once')
            self.assertNotIn('--push_to_hub', plan['command'])
            dataset = load_dataset(str(root / 'flux/dataset'))
            self.assertEqual(dataset['train'][0]['id'], 'train')
            self.assertEqual(dataset['validation'][0]['id'], 'validation')
            self.assertEqual(json.loads((root / 'flux/training.json').read_text())['stage'], 'prepared')

    def test_flux_custom_captions_have_bounded_embedding_cache_and_dropout(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source=reviewed_fixture(root/'source')
            path=source/'train/metadata.jsonl'
            row=json.loads(path.read_text())
            row2={**row,'id':'second','file_name':'second.jpg','group_id':'second','panorama_id':'second',
                  'text':'Red rounded letters outlined in white on a metal shutter.'}
            import shutil
            shutil.copyfile(source/'train'/row['file_name'],source/'train'/row2['file_name'])
            path.write_text(json.dumps(row)+'\n'+json.dumps(row2)+'\n')
            summary=json.loads((source/'summary.json').read_text())
            summary['train']=2;summary['manifest_sha256']['train']=hashlib.sha256(path.read_bytes()).hexdigest()
            write_json(source/'summary.json',summary)
            plan=prepare(source,root/'flux')
            self.assertEqual(plan['caption_strategy'],'per-image captions')
            command=plan['command']
            self.assertEqual(command[command.index('--caption_column')+1],'text')
            self.assertEqual(command[command.index('--max_sequence_length')+1],'128')
            self.assertEqual(command[command.index('--caption_dropout')+1],'0.1')

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

    def test_captioned_model_exports_all_samples_recipe_and_separate_repositories(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);home=root/'home';home.mkdir()
            source=reviewed_fixture(home/'dataset')
            (home/'lora').mkdir();(home/'lora/pytorch_lora_weights.safetensors').write_bytes(b'test adapter fixture')
            write_json(home/'training.json',{'stage':'trained','model':'black-forest-labs/FLUX.2-klein-base-4B',
                'revision':'a'*40,'steps':2000,'step':2000,'rank':16,'resolution':512,'seed':19182,
                'train_count':1,'validation_count':1,'recipe':{'caption_strategy':'per-image captions',
                'text_max_sequence_length':128,'caption_dropout':.1,
                'command':['--max_sequence_length','128','--caption_dropout','0.1']}})
            write_json(home/'publishing.json',{'dataset_repo_id':'test/captioned-v3','model_repo_id':'test/captioned-model'})
            (home/'evaluation').mkdir();samples=[]
            for index in range(1,5):
                for mode in ('base','lora'):
                    name=f'{index}-{mode}.png'
                    Image.new('RGB',(16,16),'blue').save(home/'evaluation'/name)
                    samples.append({'file_name':name})
            write_json(home/'evaluation.json',{'samples':samples})
            result=publish_run(home,root/'package',namespace='test')
            self.assertEqual(result['dataset_repo'],'test/captioned-v3')
            self.assertEqual(result['model_repo'],'test/captioned-model')
            exported=json.loads((root/'package/model/training.json').read_text())
            self.assertEqual(exported['recipe']['text_max_sequence_length'],128)
            self.assertEqual(exported['recipe']['parameters']['caption_dropout'],.1)
            self.assertIn('evaluation/4-lora.png',verify_package(root/'package/model')['files'])
            self.assertNotIn('c0runa',(root/'package/model/README.md').read_text())


if __name__ == '__main__':
    unittest.main()
