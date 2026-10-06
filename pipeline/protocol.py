"""Split and epoch-selection rules, independent of training and test outcomes."""
import numpy as np
import pandas as pd


def fold_plan(data):
    if sorted(data.cvfold.unique()) != list(range(5)):
        raise ValueError('Expected the five original folds 0..4')
    if data.groupby('wt_seq').cvfold.nunique().max() != 1:
        raise ValueError('A WT protein occurs in more than one fold')
    sequences = {
        k: set(data.loc[data.cvfold == k, 'wt_seq']) | set(data.loc[data.cvfold == k, 'mut_seq'])
        for k in range(5)
    }
    for k in range(5):
        for j in range(k):
            if sequences[k] & sequences[j]:
                raise ValueError(f'Sequence overlap between folds {k} and {j}')
    plans = []
    for outer in range(5):
        early = (outer + 1) % 5
        train = [k for k in range(5) if k not in (outer, early)]
        refit = [k for k in range(5) if k != outer]
        plans.append(dict(
            outer_fold=outer, early_stopping_fold=early,
            inner_train_folds=train, refit_folds=refit,
            inner_train_rows=np.flatnonzero(data.cvfold.isin(train)).tolist(),
            early_stopping_rows=np.flatnonzero(data.cvfold == early).tolist(),
            refit_rows=np.flatnonzero(data.cvfold.isin(refit)).tolist(),
            outer_rows=np.flatnonzero(data.cvfold == outer).tolist(),
        ))
    return plans


def choose_epoch(curve, max_epochs):
    """Raw equal-protein MSE minimum; earliest epoch breaks ties."""
    curve = pd.DataFrame(curve).sort_values('epoch').reset_index(drop=True)
    if curve.epoch.tolist() != list(range(1, max_epochs + 1)):
        raise ValueError('Incomplete or duplicated inner validation curve')
    if not np.isfinite(curve['mse_protein']).all():
        raise ValueError('Non-finite inner selection score')
    best = curve.loc[curve.mse_protein.idxmin()]
    return dict(selected_epoch=int(best.epoch), selected_mse_protein=float(best.mse_protein),
                minimum_at_cap=bool(best.epoch == max_epochs))


def protein_mse(y, mu, proteins):
    """Mean within each WT protein, then equally across proteins."""
    y, mu = [np.asarray(v, dtype=float) for v in (y, mu)]
    proteins = np.asarray(proteins)
    if not (y.shape == mu.shape == proteins.shape) or y.ndim != 1 or len(y) == 0:
        raise ValueError('Misaligned validation arrays')
    if not np.isfinite([y, mu]).all():
        raise ValueError('Invalid predictions')
    _, labels, counts = np.unique(proteins, return_inverse=True, return_counts=True)
    squared = (y - mu) ** 2
    return dict(mse_protein=float(np.mean(np.bincount(labels, weights=squared) / counts)),
                mse_mutation=float(np.mean(squared)))
