import base64
import hashlib
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from publish_crops_r2 import crop_path, upload, main


class CropPublicationTests(unittest.TestCase):
    def test_only_public_crops_inside_dataset_are_eligible(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            site = root / 'city/site'
            (site / 'img').mkdir(parents=True)
            (site / 'img/a.jpg').write_bytes(b'image')
            self.assertEqual(crop_path(site, 'img/a.jpg')[0], 'img/a.jpg')
            self.assertEqual(crop_path(site, '/img/a.jpg')[0], 'img/a.jpg')
            for bad in ['../original.jpg', '/img/../img/a.jpg', '/ctx/a.jpg',
                        '/proposal-context/apple/a.jpg', 'https://example.com/img/a.jpg', '/img/a.png']:
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    crop_path(site, bad)
            (root / 'private.jpg').write_bytes(b'private')
            (site / 'img/escape.jpg').symlink_to(root / 'private.jpg')
            with self.assertRaises(ValueError):
                crop_path(site, '/img/escape.jpg')

    def test_upload_checksums_resume_and_changed_files(self):
        with tempfile.TemporaryDirectory() as folder:
            site = Path(folder) / 'site'
            (site / 'img').mkdir(parents=True)
            path = site / 'img/a.jpg'
            path.write_bytes(b'jpeg')
            client = Mock()
            result = upload(client, site, 'bucket', '/img/a.jpg')
            request = client.put_object.call_args.kwargs
            self.assertEqual(request['Key'], 'crops/v1/img/a.jpg')
            self.assertEqual(request['ContentMD5'], base64.b64encode(hashlib.md5(b'jpeg').digest()).decode())
            self.assertEqual(request['Metadata']['sha256'], hashlib.sha256(b'jpeg').hexdigest())
            self.assertTrue(result[-1])
            self.assertFalse(upload(client, site, 'bucket', '/img/a.jpg', result[1:4])[-1])
            self.assertEqual(client.put_object.call_count, 1)
            path.write_bytes(b'changed jpeg')
            self.assertTrue(upload(client, site, 'bucket', '/img/a.jpg', result[1:4])[-1])
            self.assertEqual(client.put_object.call_count, 2)

    def test_offline_inventory_then_configured_destination_keeps_checkpoint_guard(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
            root = Path(folder)
            site, state, env = root / 'site', root / 'state', root / 'r2.env'
            (site / 'layers').mkdir(parents=True)
            (site / 'img').mkdir()
            (site / 'img/a.jpg').write_bytes(b'jpeg')
            (site / 'layers/manifest.json').write_text(' {"categories": []}')
            (site / 'data.json').write_text('{"signs": [{"id": "a"}]}')
            argv = ['upload', '--site', str(site), '--state', str(state), '--env-file', str(env), '--inventory-only']
            with patch('sys.argv', argv):
                main()
            self.assertFalse((state / 'destination.json').exists())
            env.write_text('R2_ACCOUNT_ID=test-account\nR2_BUCKET=test-crops\nR2_PUBLIC_URL=https://crops.example\n')
            with patch('sys.argv', argv):
                main()
            self.assertEqual(json.loads((state / 'destination.json').read_text()),
                             {'account': 'test-account', 'bucket': 'test-crops', 'prefix': 'crops/v1'})
            self.assertEqual(json.loads((state / 'status.json').read_text())['crops'], 1)
            with patch('sys.argv', argv + ['--account', 'different-account']):
                with self.assertRaisesRegex(ValueError, 'different destination'):
                    main()

    def test_failed_upload_does_not_return_success(self):
        with tempfile.TemporaryDirectory() as folder:
            site = Path(folder) / 'site'
            (site / 'img').mkdir(parents=True)
            (site / 'img/a.jpg').write_bytes(b'jpeg')
            client = Mock()
            client.put_object.side_effect = RuntimeError('network unavailable')
            with self.assertRaisesRegex(RuntimeError, 'network unavailable'):
                upload(client, site, 'bucket', '/img/a.jpg')


if __name__ == '__main__':
    unittest.main()
