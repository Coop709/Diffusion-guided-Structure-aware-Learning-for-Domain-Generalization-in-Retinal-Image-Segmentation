import unittest

import numpy as np

from evaluation.metrics import binary_metrics


class MetricTests(unittest.TestCase):
    def test_perfect_prediction(self):
        values = binary_metrics(np.array([[0.1, 0.9]]), np.array([[0, 1]]))
        self.assertAlmostEqual(values["dice"], 1.0)
        self.assertAlmostEqual(values["auc"], 1.0)
        self.assertAlmostEqual(values["accuracy"], 1.0)

    def test_tied_scores(self):
        values = binary_metrics(np.array([0.5, 0.5]), np.array([0, 1]))
        self.assertAlmostEqual(values["auc"], 0.5)


if __name__ == "__main__":
    unittest.main()
