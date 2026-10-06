"""Pure NumPy standard/adaptive CV+; no fitted-model or filesystem dependencies."""
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
import numpy as np


def order_indices(n, alpha):
    if n < 1 or not 0 < alpha < .5:
        raise ValueError('Require n >= 1 and 0 < alpha < 0.5')
    a = Decimal(str(alpha))
    # Decimal avoids accidental off-by-one at exact integer ranks.
    lower = int((a * (n + 1)).to_integral_value(rounding=ROUND_FLOOR))
    upper = int(((1-a) * (n + 1)).to_integral_value(rounding=ROUND_CEILING))
    return lower, upper


def intervals(y, mu_oof, sigma_oof, fold_id, mu_test, sigma_test,
              alpha=.1, adaptive=True, chunk_size=64):
    y, mu_oof, sigma_oof = [np.asarray(v, dtype=float) for v in (y, mu_oof, sigma_oof)]
    fold_id = np.asarray(fold_id)
    mu_test, sigma_test = [np.asarray(v, dtype=float) for v in (mu_test, sigma_test)]
    n = len(y)
    if y.ndim != 1 or mu_oof.shape != (n,) or sigma_oof.shape != (n,) or fold_id.shape != (n,):
        raise ValueError('OOF arrays must be aligned vectors')
    if mu_test.ndim != 2 or sigma_test.shape != mu_test.shape:
        raise ValueError('Test arrays must have shape (K, n_test)')
    if not np.issubdtype(fold_id.dtype, np.integer) or not np.array_equal(np.unique(fold_id), np.arange(mu_test.shape[0])):
        raise ValueError('Every fold must be present and indexed 0..K-1')
    if not all(np.isfinite(v).all() for v in (y, mu_oof, sigma_oof, mu_test, sigma_test)):
        raise ValueError('Non-finite predictions')
    if np.any(sigma_oof <= 0) or np.any(sigma_test <= 0) or chunk_size < 1:
        raise ValueError('Sigma and chunk size must be positive')
    score = np.abs(y-mu_oof)
    if adaptive:
        score = score / sigma_oof
    jl, ju = order_indices(n, alpha)
    lower, upper = np.empty(mu_test.shape[1]), np.empty(mu_test.shape[1])
    for start in range(0, mu_test.shape[1], chunk_size):
        stop = min(start+chunk_size, mu_test.shape[1])
        center = mu_test[fold_id, start:stop]
        half = score[:, None] * (sigma_test[fold_id, start:stop] if adaptive else 1.)
        lo, hi = center-half, center+half
        assert lo.shape == hi.shape == (n, stop-start)
        lower[start:stop] = -np.inf if jl == 0 else np.partition(lo, jl-1, axis=0)[jl-1]
        upper[start:stop] = np.inf if ju == n+1 else np.partition(hi, ju-1, axis=0)[ju-1]
    if np.any(lower > upper):
        raise ValueError('Inverted CV+ interval')
    return lower, upper


def metrics(y, lower, upper, alpha):
    y, lower, upper = [np.asarray(v) for v in (y, lower, upper)]
    width = upper-lower
    score = width + 2/alpha * (np.maximum(lower-y, 0)+np.maximum(y-upper, 0))
    return dict(coverage=float(np.mean((lower <= y)&(y <= upper))),
                mean_width=float(np.mean(width)), median_width=float(np.median(width)),
                interval_score=float(np.mean(score)))


def uncertainty_rank(sigma_oof, fold_id, sigma_test):
    """Diagnostic only: mean fold-specific OOF percentile of test sigma.

    Scale-invariant to multiplying each model's sigma by its own positive factor.
    Fixed percentile groups [0,.2,.4,.6,.8,1]; no test-dependent cutpoints.
    """
    ranks = []
    for k, test in enumerate(sigma_test):
        reference = np.sort(sigma_oof[fold_id == k])
        left = np.searchsorted(reference, test, side='left')
        right = np.searchsorted(reference, test, side='right')
        ranks.append((left+right)/(2*len(reference)))
    return np.mean(ranks, axis=0)
