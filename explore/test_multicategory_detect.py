import json
import threading
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io
from pathlib import Path

from multicategory_detect import atomic_json, legacy_results, merge_candidates, worker_for_panorama, reuse_completed_views, prefetch


class MultiCategoryTests(unittest.TestCase):
    def test_prefetch_overlaps_one_item_and_preserves_order(self):
        loaded_second = threading.Event()
        calls = []
        def load(i):
            calls.append(i)
            if i == 1:
                loaded_second.set()
            return (i,)
        iterator = prefetch(range(5), load)
        self.assertEqual(next(iterator), (0,))
        self.assertTrue(loaded_second.wait(timeout=2))
        self.assertEqual(calls, [0, 1])
        self.assertEqual(list(iterator), [(1,), (2,), (3,), (4,)])
        self.assertEqual(list(prefetch([], load)), [])
        self.assertEqual(list(prefetch(range(2), lambda _: None)), [None, None])

    def test_sampling_change_reuses_only_selected_compatible_complete_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"thin": 10, "year": None, "bbox": None, "revision": "abc", "threshold": .5}
            for name, updates in [("old", {"thin": 6}), ("wrong_model", {"revision": "xyz"}), ("wrong_threshold", {"threshold": .4})]:
                run = root / name
                (run / "views" / name).mkdir(parents=True)
                (run / "config.json").write_text(json.dumps({**config, **updates}))
                atomic_json(run / "views" / name / "result.json", {"view_id": name})
            (root / "old/views/unselected").mkdir()
            atomic_json(root / "old/views/unselected/result.json", {})
            (root / "old/views/partial").mkdir()
            out = root / "new";out.mkdir()
            selected = {"old", "wrong_model", "wrong_threshold", "partial"}
            self.assertEqual(reuse_completed_views(out, config, selected), 1)
            self.assertTrue((out / "views/old").is_symlink())
            self.assertEqual(json.loads((out / "views/old/result.json").read_text()), {"view_id": "old"})
            self.assertEqual([p.name for p in (out / "views").iterdir()], ["old"])
            self.assertEqual(reuse_completed_views(out, config, selected), 0)

    def test_worker_assignments_are_disjoint_and_cover_all_panoramas(self):
        panos = [str(i) for i in range(200)]
        for count in (1, 2, 4, 6):
            assignments = [{p for p in panos if worker_for_panorama(p, count) == w} for w in range(count)]
            self.assertEqual(set.union(*assignments), set(panos))
            self.assertEqual(sum(map(len, assignments)), len(panos))
            self.assertTrue(all(assignments))
            self.assertTrue(all(worker_for_panorama(int(p), count) == worker_for_panorama(p, count) for p in panos))

    def test_duplicate_provenance_and_cross_category_are_preserved(self):
        rows = [
            {"id": "a", "category_proposal": "graffiti", "box": [0, 0, 100, 100], "score": .7},
            {"id": "b", "category_proposal": "graffiti", "box": [1, 1, 101, 101], "score": .9},
            {"id": "c", "category_proposal": "shutter", "box": [0, 0, 100, 100], "score": .8},
            {"id": "d", "category_proposal": "graffiti", "box": [200, 0, 300, 100], "score": .6},
        ]
        result = merge_candidates(rows)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0]["duplicate_of"], "b")
        self.assertTrue(all("duplicate_of" not in row for row in result[1:]))

    def test_pinned_resume_rejects_mismatch_without_querying_latest_model(self):
        from multicategory_detect import main
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/'config.json').write_text(json.dumps({'revision':'pinned-revision','thin':10}))
            with patch('sys.argv',['multicategory_detect.py','--resume-run',str(path),'--thin','6']), patch('huggingface_hub.HfApi.model_info',side_effect=AssertionError('must not query latest')), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:main()
                self.assertEqual(raised.exception.code,2)
            self.assertEqual(json.loads((path/'config.json').read_text())['revision'],'pinned-revision')

    def test_only_completed_legacy_views_are_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            folder = base / "apple/store_sign"
            folder.mkdir(parents=True)
            (folder / "done.txt").write_text("p_1\n")
            (folder / "instances.jsonl").write_text(
                '{"pano":"p","face":1}\n{"pano":"p","face":3}\n{"pano":')
            done, rows = legacy_results(base, "apple")
            self.assertEqual(done, {"p_1"})
            self.assertEqual(list(rows), ["p_1"])
            atomic_json(base / "result.json", {"ok": True})
            self.assertFalse((base / "result.tmp").exists())


if __name__ == "__main__":
    unittest.main()
