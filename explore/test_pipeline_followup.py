import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pipeline_followup as pipeline


class FollowupTests(unittest.TestCase):
    def test_backfill_waits_for_success_and_complete_inventory(self):
        for active, result, count, starts in [
            ('active', 'success', 10, False), ('inactive', 'exit-code', 10, False),
            ('inactive', 'success', 9, False), ('inactive', 'success', 10, True),
        ]:
            with self.subTest(active=active, result=result, count=count), tempfile.TemporaryDirectory() as folder:
                base = Path(folder)
                (base / 'vast_run.json').write_text('{"phase":"running"}')
                plan = {'apple_run': 'apple', 'google_run': 'google', 'backfill_run': 'backfill',
                        'apple_views': 10, 'google_views': 20}
                with patch.object(pipeline, 'BASE', base), patch.object(pipeline, 'PLAN', base / 'plan.json'), \
                     patch.object(pipeline, 'service', return_value={'ActiveState': active, 'Result': result}), \
                     patch.object(pipeline, 'completed', return_value=count), \
                     patch.object(pipeline.shutil, 'disk_usage') as disk, \
                     patch.object(pipeline.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
                    disk.return_value.free = 100 * 1024**3
                    pipeline.tick(plan, apply=True)
                    self.assertEqual(run.called, starts)
                    if starts:
                        pipeline.tick(plan, apply=True)
                        self.assertEqual(run.call_count, 1)

    def test_google_assets_must_be_complete_and_confined(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / 'run'
            view = run / 'views/view'
            view.mkdir(parents=True)
            (root / 'source.jpg').touch()
            (view / 'crop.jpg').touch()
            (view / 'mask.png').touch()
            (run / 'config.json').write_text('{"prompts":{"person":["person"]}}')
            record = {'attempted_categories': ['person'], 'source_path': 'source.jpg',
                      'predictions': [{'crop': 'run/views/view/crop.jpg', 'mask': 'run/views/view/mask.png'}]}
            result = view / 'result.json'
            result.write_text(json.dumps(record))
            with patch.object(pipeline, 'ROOT', root):
                self.assertEqual(pipeline.validate_assets(run), 1)
                (view / 'mask.png').unlink()
                with self.assertRaisesRegex(ValueError, 'asset'):
                    pipeline.validate_assets(run)
                record['predictions'][0]['mask'] = 'source.jpg'
                result.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, 'asset'):
                    pipeline.validate_assets(run)


if __name__ == '__main__':
    unittest.main()
