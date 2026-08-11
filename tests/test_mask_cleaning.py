import unittest
import numpy as np

import cafin_core as cc


class MaskCleaningTests(unittest.TestCase):
    def test_binary_components_are_labelled_and_small_objects_removed(self):
        mask = np.zeros((20, 20), np.uint8)
        mask[2:8, 2:8] = 1
        mask[12:18, 12:18] = 1
        mask[10, 2] = 1
        cleaned = cc.clean_mask(mask, min_area=10, fill_holes=False)
        self.assertEqual(int(cleaned.max()), 2)
        self.assertEqual(int((cleaned > 0).sum()), 72)

    def test_hole_is_filled_with_its_cell_label(self):
        mask = np.zeros((12, 12), np.int32)
        mask[2:10, 2:10] = 7
        mask[5:7, 5:7] = 0
        cleaned = cc.clean_mask(mask, min_area=0, fill_holes=True)
        self.assertEqual(int(cleaned.max()), 1)
        self.assertGreater(int(cleaned[5, 5]), 0)

    def test_disconnected_same_id_is_split(self):
        mask = np.zeros((15, 15), np.int32)
        mask[2:6, 2:6] = 4
        mask[9:13, 9:13] = 4
        cleaned = cc.clean_mask(mask, min_area=0, split_disconnected=True)
        self.assertEqual(int(cleaned.max()), 2)

    def test_border_cells_can_be_removed(self):
        mask = np.zeros((12, 12), np.int32)
        mask[0:4, 2:6] = 1
        mask[6:10, 6:10] = 2
        cleaned = cc.clean_mask(mask, min_area=0, remove_border=True)
        self.assertEqual(int(cleaned.max()), 1)
        self.assertEqual(int((cleaned > 0).sum()), 16)


if __name__ == "__main__":
    unittest.main()
