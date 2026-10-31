import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from caption_images import build_dataset, collect_archive, digest, load_cached, open_store, parse_caption, recipe, reviewed_caption
from test_publish_huggingface import reviewed_fixture


def answer(caption='Green bubble lettering outlined in black covers a pale textured wall. The crop shows the painted shapes closely, with part of the piece cut off at the edge.'):
    return {'caption':caption,'text_legibility':'unreadable','visible_text':'','uncertain_details':[]}


class CaptionTests(unittest.TestCase):
    def test_invalid_or_truncated_generation_is_not_accepted(self):
        for raw in ['not json','{"caption":"unfinished',json.dumps(answer('green letters')),
                    json.dumps({**answer(),'text_legibility':'probably'})]:
            with self.assertRaises((ValueError,TypeError)):
                parse_caption(raw)
        self.assertEqual(parse_caption('```json\n'+json.dumps(answer())+'\n```')['visible_text'],'')

    def test_location_and_transcription_are_flagged(self):
        self.assertIn('location_guess',parse_caption(json.dumps(answer('A graffiti piece on a wall in Coruña has green and black rounded lettering with pale filling and a tightly cropped composition.')))['flags'])
        self.assertIn('text_in_description',parse_caption(json.dumps(answer('Green lettering on a pale wall spells the word "CORUNA" in large rounded shapes with thick black outlines, filling the rectangular crop.')))['flags'])

    def test_cache_requires_matching_image_and_recipe(self):
        with tempfile.TemporaryDirectory() as temp,open_store(Path(temp)/'captions.sqlite3') as db:
            db.execute('INSERT INTO captions VALUES (?,?,?,?)',('object','sha1','recipe1',json.dumps(answer())))
            self.assertIsNotNone(load_cached(db,'object','sha1','recipe1'))
            self.assertIsNone(load_cached(db,'object','sha2','recipe1'))
            self.assertIsNone(load_cached(db,'object','sha1','recipe2'))
            self.assertNotEqual(recipe(max_side=512)['id'],recipe(max_side=768)['id'])

    def test_database_selection_uses_representatives_and_balances_categories(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);state=root/'state';state.mkdir();site=root/'site';(site/'img').mkdir(parents=True)
            (state/'current.json').write_text(json.dumps({'query_snapshot':'frozen'}))
            from PIL import Image
            Image.new('RGB',(32,32),'green').save(site/'img/object.jpg')
            with sqlite3.connect(state/'frozen.sqlite3') as db:
                db.executescript('CREATE TABLE objects(id TEXT,category TEXT,score REAL);CREATE TABLE records(id TEXT,payload TEXT);')
                for identity,category,score,crop in [('a1','graffiti',.99,'/img/object.jpg'),('a2','graffiti',.98,'/img/object.jpg'),
                    ('b1','store_sign',.90,'/img/object.jpg'),('unsafe','graffiti',1.,'/img/../../.env.jpg')]:
                    db.execute('INSERT INTO records VALUES (?,?)',(identity,json.dumps({'id':identity,'category':category,'crop':crop})))
                    if identity!='a2':db.execute('INSERT INTO objects VALUES (?,?,?)',(identity,category,score))
            report=collect_archive(state,site,['graffiti','store_sign'],root/'input',limit=2)
            rows=list(map(json.loads,(root/'input/archive/metadata.jsonl').read_text().splitlines()))
            self.assertEqual({r['id'] for r in rows},{'a1','b1'})
            self.assertEqual(report['query_snapshot'],'frozen')
            for r in rows:self.assertEqual(digest(root/'input/archive'/r['file_name']),digest(site/'img/object.jpg'))

    def fixture(self,root):
        dataset=reviewed_fixture(root/'source');captions=root/'captions';captions.mkdir()
        records=[]
        for split in ('train','validation'):
            row=json.loads((dataset/split/'metadata.jsonl').read_text())
            records.append({'id':row['id'],'split':split,'file_name':row['file_name'],
                'image_sha256':digest(dataset/split/row['file_name']),'recipe_id':'recipe1',
                'model':'Qwen/Qwen3.5-4B','revision':'pinned',**parse_caption(json.dumps(answer()))})
        path=captions/'captions.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in records))
        (captions/'review.json').write_text(json.dumps({'reviewed':True,'method':'test review','reviewed_at':'now','captions_sha256':digest(path)}))
        return dataset,captions

    def test_build_preserves_images_ids_and_splits_without_candidate_text(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dataset,captions=self.fixture(root)
            store=root/'caption-store.sqlite3'
            build_dataset(dataset,captions,root/'result',store=store)
            for split in ('train','validation'):
                before=json.loads((dataset/split/'metadata.jsonl').read_text())
                after=json.loads((root/'result'/split/'metadata.jsonl').read_text())
                self.assertEqual(before['id'],after['id'])
                self.assertEqual(digest(dataset/split/before['file_name']),digest(root/'result'/split/after['file_name']))
                self.assertNotIn('c0runa',after['text'])
                self.assertEqual(after['caption_image_sha256'],digest(dataset/split/before['file_name']))
                with open_store(store) as db:
                    checked=reviewed_caption(db,after['id'],after['caption_image_sha256'],after['caption_recipe_id'])
                    self.assertEqual(checked['caption'],after['text'])
                    self.assertFalse(checked['review']['human_verified'])

    def test_build_refuses_changed_captions_or_image_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dataset,captions=self.fixture(root)
            path=captions/'captions.jsonl';path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'changed after review'):
                build_dataset(dataset,captions,root/'result')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dataset,captions=self.fixture(root)
            row=json.loads((dataset/'train/metadata.jsonl').read_text())
            with (dataset/'train'/row['file_name']).open('ab') as stream: stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'changed caption input'):
                build_dataset(dataset,captions,root/'result')


if __name__=='__main__':unittest.main()
