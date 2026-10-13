import base64
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from publish_crops_r2 import crop_path, upload


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
