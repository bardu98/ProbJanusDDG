import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'conformal'))
from protein_balanced import weighted_bounds


class ProteinWeightedTests(unittest.TestCase):
    def test_protein_masses_change_empirical_bounds(self):
        # Three mutations from protein A versus one from B. Each protein has half the mass.
        oof = {'y': np.array([1., 1., 1., 10.]), 'mu': np.zeros(4),
               'sigma': np.ones(4), 'fold_id': np.array([0, 0, 0, 1])}
        mu, sigma = np.zeros((2, 1)), np.ones((2, 1))
        protein = weighted_bounds(oof, mu, sigma, np.array([1/6, 1/6, 1/6, 1/2]),
                                  np.array([.6]), False)
        mutation = weighted_bounds(oof, mu, sigma, np.full(4, .25), np.array([.6]), False)
        np.testing.assert_array_equal(protein[0], [[-10.]])
        np.testing.assert_array_equal(protein[1], [[10.]])
        np.testing.assert_array_equal(mutation[0], [[-1.]])
        np.testing.assert_array_equal(mutation[1], [[1.]])

    def test_fold_scale_cancellation_and_nested_levels(self):
        oof = {'y': np.array([1., -2., 3., -4.]), 'mu': np.zeros(4),
               'sigma': np.array([.5, 1., 2., 3.]), 'fold_id': np.array([0, 0, 1, 1])}
        mu, sigma = np.array([[0., 1.], [2., 0.]]), np.array([[1., 2.], [3., 4.]])
        weights, levels = np.full(4, .25), np.array([.5, .6, .9])
        a = weighted_bounds(oof, mu, sigma, weights, levels, True)
        scaled = dict(oof, sigma=oof['sigma'] * np.array([.01, 100.])[oof['fold_id']])
        b = weighted_bounds(scaled, mu, sigma * np.array([.01, 100.])[:, None],
                            weights, levels, True)
        np.testing.assert_allclose(a, b)
        self.assertTrue((np.diff(a[0], axis=1) <= 0).all())
        self.assertTrue((np.diff(a[1], axis=1) >= 0).all())


if __name__ == '__main__':
    unittest.main()
