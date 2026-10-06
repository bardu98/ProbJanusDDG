import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "conformal"))
import unittest
import numpy as np
from cvplus import intervals, order_indices


class CVPlusTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([1., -2., 3., -4., .5, 2.5])
        self.mu = np.array([0., .1, .2, .3, .4, .5])
        self.sg = np.array([.5, 1., 2., .8, .6, 1.5])
        self.fold = np.array([0, 0, 1, 1, 2, 2])
        self.mt = np.array([[0., 1.], [2., -1.], [-3., 4.]])
        self.st = np.array([[1., 2.], [.5, 3.], [2., 1.]])

    def test_explicit_candidates(self):
        for adaptive in (False, True):
            lo, hi = intervals(self.y, self.mu, self.sg, self.fold, self.mt, self.st,
                               .2, adaptive, chunk_size=1)
            jl, ju = order_indices(6, .2)
            for j in range(2):
                lows, highs = [], []
                for i, k in enumerate(self.fold):
                    h = abs(self.y[i]-self.mu[i])
                    if adaptive:
                        h *= self.st[k, j]/self.sg[i]
                    lows.append(self.mt[k, j]-h)
                    highs.append(self.mt[k, j]+h)
                self.assertAlmostEqual(lo[j], sorted(lows)[jl-1])
                self.assertAlmostEqual(hi[j], sorted(highs)[ju-1])

    def test_per_model_scale_cancels(self):
        factors = np.array([.01, 100., 7.])
        first = intervals(self.y,self.mu,self.sg,self.fold,self.mt,self.st,.2)
        second = intervals(self.y,self.mu,self.sg*factors[self.fold],self.fold,
                           self.mt,self.st*factors[:,None],.2)
        np.testing.assert_allclose(first,second)

    def test_constant_sigma_matches_standard(self):
        sg, st = np.ones(6), np.ones((3,2))
        np.testing.assert_allclose(intervals(self.y,self.mu,sg,self.fold,self.mt,st,.2),
            intervals(self.y,self.mu,sg,self.fold,self.mt,st,.2,adaptive=False))

    def test_ranks_and_extreme_alpha(self):
        self.assertEqual(order_indices(2450,.1),(245,2206))
        self.assertEqual(order_indices(9,.1),(1,9))
        lo,hi = intervals(self.y,self.mu,self.sg,self.fold,self.mt,self.st,.01)
        self.assertTrue(np.isneginf(lo).all() and np.isposinf(hi).all())

    def test_no_forced_symmetry(self):
        lo,hi=intervals(self.y,self.mu,self.sg,self.fold,self.mt,self.st,.2)
        self.assertFalse(np.allclose((lo+hi)/2,self.mt.mean(axis=0)))

    def test_invalid_sigma(self):
        with self.assertRaises(ValueError):
            intervals(self.y,self.mu,np.zeros(6),self.fold,self.mt,self.st)


if __name__ == '__main__':
    unittest.main()
