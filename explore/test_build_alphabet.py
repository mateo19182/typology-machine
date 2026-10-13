import unittest
from build_alphabet import character_regions, padded_character_box


class CharacterRegionsTest(unittest.TestCase):
    def test_preserves_model_boxes_and_filters_uncertain_characters(self):
        box_a = [[0,0],[10,0],[10,20],[0,20]]
        box_b = [[14,0],[35,0],[35,20],[14,20]]
        result = character_regions([[], 'AB', .99, [box_a, box_b], ['A', 'B'], [.99, .95]])
        self.assertEqual(result, [('A', box_a, .99)])

    def test_rejects_misaligned_confidences(self):
        self.assertEqual(character_regions([[], 'A B', .99, [[], []], ['A','B'], [.99,.99,.99]]), [])

    def test_rejects_legacy_output_and_uncertain_lines(self):
        self.assertEqual(character_regions([[], 'A', .99]), [])
        self.assertEqual(character_regions([[], 'A', .90, [[]], ['A'], [.99]]), [])

    def test_padding_stops_at_neighbor_midpoints(self):
        import numpy as np
        current = [[10,0],[20,0],[20,30],[10,30]]
        previous = [[0,0],[9,0],[9,30],[0,30]]
        following = [[21,0],[30,0],[30,30],[21,30]]
        result = padded_character_box(current, previous, following)
        self.assertAlmostEqual(result[:,0].min(), 9.5)
        self.assertAlmostEqual(result[:,0].max(), 20.5)
        np.testing.assert_array_equal(current, [[10,0],[20,0],[20,30],[10,30]])

    def test_touching_neighbors_receive_no_horizontal_expansion(self):
        current = [[10,0],[20,0],[20,30],[10,30]]
        result = padded_character_box(current, [[0,0],[10,0],[10,30],[0,30]], [[20,0],[30,0],[30,30],[20,30]])
        self.assertEqual(result[:,0].min(), 10)
        self.assertEqual(result[:,0].max(), 20)


if __name__ == '__main__':
    unittest.main()
