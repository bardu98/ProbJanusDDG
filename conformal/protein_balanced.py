"""Protein-weighted empirical CV+ candidate quantiles (no finite-rank correction)."""
import numpy as np

def weighted_bounds(oof, mu, sigma, weights, levels, adaptive):
    score = np.abs(oof['y'] - oof['mu'])
    if adaptive: score = score / oof['sigma']
    fold = oof['fold_id']
    low, high = np.empty((mu.shape[1], len(levels))), np.empty((mu.shape[1], len(levels)))
    for j in range(mu.shape[1]):
        center = mu[fold, j]
        half = score * (sigma[fold, j] if adaptive else 1)
        for candidates, probs, output in [(center-half, 1-levels, low), (center+half, levels, high)]:
            order = np.argsort(candidates, kind='stable')
            cumulative = np.cumsum(weights[order].astype(np.longdouble))
            cumulative /= cumulative[-1]
            ix = np.searchsorted(cumulative, probs-1e-14, side='left')
            output[j] = candidates[order[ix]]
    assert np.isfinite(low).all() and np.isfinite(high).all() and (low <= high).all()
    return low, high
