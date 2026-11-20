import unittest

import numpy as np

from evaluation.feature_metrics import rbf_mmd_squared
from evaluation.structure_metrics import bifurcation_f1, hd95, vessel_structure_metrics


class StructureMetricTests(unittest.TestCase):
    def setUp(self):
        self.cross = np.zeros((41, 41), dtype=bool)
        self.cross[6:35, 20] = True
        self.cross[20, 6:35] = True

    def test_perfect_vessel_and_empty_masks(self):
        scores = vessel_structure_metrics(self.cross.astype(float), self.cross)
        for metric in ("cldice", "thin_recall", "bifurcation_f1"):
            self.assertAlmostEqual(scores[metric], 1.0)
        self.assertEqual(scores["hd95"], 0.0)
        empty = np.zeros_like(self.cross)
        self.assertEqual(hd95(empty, empty), 0.0)
        self.assertAlmostEqual(hd95(empty, self.cross), np.hypot(40, 40))

    def test_missing_branch_reduces_centerline_and_bifurcation_scores(self):
        prediction = self.cross.copy()
        prediction[20, :] = False
        prediction[6:35, 20] = True
        scores = vessel_structure_metrics(prediction.astype(float), self.cross)
        self.assertLess(scores["cldice"], 1.0)
        self.assertLess(scores["thin_recall"], 1.0)
        self.assertEqual(scores["bifurcation_f1"], 0.0)
        self.assertGreater(scores["hd95"], 0.0)

    def test_boundary_shift_and_bifurcation_tolerance(self):
        line = np.zeros((41, 41), dtype=bool)
        line[6:35, 10] = True
        shifted = np.roll(line, 3, axis=1)
        self.assertAlmostEqual(hd95(line, shifted), 3.0)
        shifted_cross = np.roll(self.cross, 2, axis=1)
        self.assertEqual(bifurcation_f1(shifted_cross, self.cross, tolerance=3), 1.0)
        self.assertEqual(bifurcation_f1(shifted_cross, self.cross, tolerance=1), 0.0)


class FeatureMetricTests(unittest.TestCase):
    def test_mmd_respects_permutation_normalization_and_distribution_change(self):
        samples = np.array([[1., 0., 0.], [0., 1., 0.]])
        self.assertAlmostEqual(rbf_mmd_squared(samples, samples[::-1]), 0.0, places=7)
        self.assertAlmostEqual(rbf_mmd_squared(samples, samples * 17), 0.0, places=7)
        self.assertAlmostEqual(rbf_mmd_squared(samples, -samples), 1 - np.exp(-2), places=7)
        self.assertAlmostEqual(rbf_mmd_squared(samples, -samples, block_size=1),
                               rbf_mmd_squared(samples, -samples, block_size=2))

    def test_empty_feature_bank_is_rejected(self):
        with self.assertRaises(ValueError):
            rbf_mmd_squared(np.empty((0, 3)), np.ones((2, 3)))


if __name__ == "__main__":
    unittest.main()
