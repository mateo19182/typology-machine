"""Exercise saved human labels and invalid requests against the real HTTP handler."""
import json
import tempfile
import threading
import unittest
from pathlib import Path

import requests

import category_eval as app


class ReviewServerTest(unittest.TestCase):
    def test_labels_persist_and_invalid_writes_do_not_change_them(self):
        original = app.BASE
        with tempfile.TemporaryDirectory() as directory:
            app.BASE = Path(directory)
            app.write_json(app.BASE / "sample.json", {"id": "fixture", "categories": app.CATEGORIES, "items": [{"id": "crop"}]})
            (app.BASE / "multicategory").mkdir()
            app.write_json(app.BASE / "multicategory/sample.json", {"id": "new-fixture", "categories": app.CATEGORIES, "items": [{"id": "new-crop"}]})
            server = app.make_server("127.0.0.1", 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}"
            try:
                payload = {"sample_id": "fixture", "id": "crop", "category": "graffiti"}
                self.assertEqual(requests.post(url + "/api/label", json=payload).status_code, 200)
                self.assertEqual(requests.get(url + "/api/data").json()["labels"]["crop"]["category"], "graffiti")
                for override in ({"category": "invented"}, {"id": "missing"}, {"sample_id": "stale"}):
                    self.assertEqual(requests.post(url + "/api/label", json=payload | override).status_code, 400)
                self.assertEqual(requests.post(url + "/api/label", json=payload, headers={"Origin": "https://unrelated.example"}).status_code, 403)
                self.assertEqual(requests.get(url + "/api/export").json()["labels"]["crop"]["category"], "graffiti")
                new_payload = {"sample_id": "new-fixture", "id": "new-crop", "category": "street_name_plate"}
                self.assertEqual(requests.post(url + "/multicategory/api/label", json=new_payload).status_code, 200)
                self.assertEqual(requests.get(url + "/multicategory/api/data").json()["labels"]["new-crop"]["category"], "street_name_plate")
                self.assertEqual(len(requests.get(url + "/api/data").json()["labels"]), 1)
                self.assertEqual(requests.post(url + "/api/label", json=new_payload).status_code, 400)
                with app.connect() as db:
                    self.assertEqual(db.execute("SELECT category FROM labels").fetchone()[0], "graffiti")
                self.assertEqual(requests.post(url + "/api/label", json=payload | {"category": None}).status_code, 200)
                self.assertEqual(requests.get(url + "/api/data").json()["labels"], {})
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
                app.BASE = original


if __name__ == "__main__":
    unittest.main()
