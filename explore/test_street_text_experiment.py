import unittest
from street_text_experiment import merge, positions


class StreetTextGeometryTest(unittest.TestCase):
    def test_tiles_cover_image_edges_without_gaps(self):
        for length in [700, 1536, 1600, 3072, 4864]:
            starts = positions(length)
            self.assertEqual(starts[0], 0)
            self.assertGreaterEqual(starts[-1]+1536, length)
            self.assertTrue(all(b <= a+1536 for a, b in zip(starts, starts[1:])))

    def test_repeated_tile_detection_keeps_provenance_and_distinct_lines(self):
        rows = [dict(box=[10, 10, 110, 30], passes=['whole']),
                dict(box=[11, 10, 111, 30], passes=['tiles']),
                dict(box=[10, 40, 110, 60], passes=['tiles'])]
        kept = merge(rows)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]['passes'], ['tiles', 'whole'])
        self.assertEqual(kept[1]['passes'], ['tiles'])


if __name__ == '__main__':
    unittest.main()
