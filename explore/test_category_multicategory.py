import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import category_multicategory as app


class SummaryTest(unittest.TestCase):
    def test_detector_agreement_never_becomes_human_accuracy(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output = base / 'multicategory'
            run = output / 'runs/model'
            run.mkdir(parents=True)
            (output / 'sample.json').write_text(json.dumps({'id': 'new', 'items': [{'id': 'crop', 'sampling_hint': 'graffiti'}], 'sampling': {'graffiti': 1}}))
            (run / 'run.json').write_text(json.dumps({'sample_id': 'new'}))
            (run / 'predictions.jsonl').write_text(json.dumps({'id': 'crop', 'category': 'graffiti', 'confidence': .9}) + '\n')
            with sqlite3.connect(base / 'reviews.sqlite3') as db:
                db.execute('CREATE TABLE labels(sample_id TEXT,item_id TEXT,category TEXT)')
                db.execute("INSERT INTO labels VALUES('old','crop','graffiti')")
            with patch.object(app, 'BASE', base):
                summary = app.summarize(output)[0]
                self.assertEqual(summary['proposal_agreement'], {'graffiti': 1})
                self.assertEqual(summary['human_reviewed'], 0)
                self.assertIsNone(summary['human_accuracy'])
                with sqlite3.connect(base / 'reviews.sqlite3') as db:
                    db.execute("INSERT INTO labels VALUES('new','crop','other')")
                summary = app.summarize(output)[0]
                self.assertEqual(summary['human_reviewed'], 1)
                self.assertEqual(summary['human_accuracy'], 0)


if __name__ == '__main__':
    unittest.main()
