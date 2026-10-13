import unittest
from build_alphabet import character_regions


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


if __name__ == '__main__':
    unittest.main()
