import os
"""Recreate the four-panel CV+ paper figures from the MSE-selected run."""
from pathlib import Path
import argparse
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(os.environ.get("NAIVE_RUN_DIR", ROOT/"runs/naive")).resolve()
sys.path.insert(0, str(ROOT / 'conformal'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'pipeline'))
from cvplus import intervals, uncertainty_rank, metrics
from protein_balanced import weighted_bounds
from plotting import violin_points

METHODS = ('standard_cvplus', 'adaptive_cvplus')
COLORS = ('#252525', '#c62828')
LEVELS = np.arange(51, 100) / 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregation', choices=('median', 'mean'), default='median')
    parser.add_argument('--weighting', choices=('mutation_corrected', 'protein_empirical'),
                        default='protein_empirical')
    args = parser.parse_args()
    out = HERE / 'figures'
    out.mkdir(exist_ok=True)
    with np.load(HERE / 'predictions/oof_all.npz') as z:
        oof = {key: z[key] for key in z.files}
    train = pd.read_parquet(ROOT / 'data/S2450.parquet').reset_index(drop=True)
    np.testing.assert_array_equal(oof['y'], train.ddG.to_numpy())
    weights = (1 / (train.wt_seq.nunique() * train.groupby('wt_seq').wt_seq.transform('size'))).to_numpy()
    np.testing.assert_allclose(weights.sum(), 1)
    np.testing.assert_allclose(pd.Series(weights).groupby(train.wt_seq).sum(), 1 / train.wt_seq.nunique())
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    rows = []
    for number, name in enumerate(('S669L', 'S461L'), start=1):
        with np.load(HERE / 'predictions' / f'{name}_cvplus.npz') as z:
            p = {key: z[key] for key in z.files}
        frame = pd.read_parquet(ROOT / 'data' / f'{name}.parquet').reset_index(drop=True)
        y, rank = (p[key] for key in ('y', 'uncertainty_rank'))
        mu = np.median(p['mu_folds'], axis=0) if args.aggregation == 'median' else p['mu_folds'].mean(axis=0)
        np.testing.assert_array_equal(y, frame.ddG.to_numpy())
        np.testing.assert_allclose(mu, p['mu_ensemble'] if args.aggregation == 'median' else p['mu_ensemble_mean_reference'])
        np.testing.assert_allclose(rank, uncertainty_rank(oof['sigma'], oof['fold_id'], p['sigma_folds']))
        groups = np.digitize(rank, [.2, .4, .6, .8])
        error = np.abs(y - mu)
        error_groups = np.digitize(error, np.quantile(error, [.2, .4, .6, .8]))
        widths, covered, curves = [], [], []
        for method in METHODS:
            prefix = f'{method}_{args.weighting}_a0.1'
            lo, hi = p[prefix + '_lower'], p[prefix + '_upper']
            widths.append(hi - lo)
            covered.append((lo <= y) & (y <= hi))
            if args.weighting == 'protein_empirical':
                lows, highs = weighted_bounds(oof, p['mu_folds'], p['sigma_folds'],
                                               weights, LEVELS, method == METHODS[1])
                np.testing.assert_array_equal(lows[:, 39], lo)
                np.testing.assert_array_equal(highs[:, 39], hi)
                curves.append((lows <= y[:, None]) & (y[:, None] <= highs))
            else:
                columns = []
                for level in LEVELS:
                    low, high = intervals(oof['y'], oof['mu'], oof['sigma'], oof['fold_id'],
                                          p['mu_folds'], p['sigma_folds'],
                                          alpha=round(1 - level, 2), adaptive=(method == METHODS[1]))
                    if np.isclose(level, .9):
                        np.testing.assert_array_equal(low, lo)
                        np.testing.assert_array_equal(high, hi)
                    columns.append((low <= y) & (y <= high))
                curves.append(np.asarray(columns).T)
            row = dict(dataset=name, method=method, weighting=args.weighting,
                       **metrics(y, lo, hi, alpha=.1))
            rows.append(row)
        widths, covered, curves = map(np.asarray, (widths, covered, curves))
        change = widths[1] - widths[0]
        clusters = [np.flatnonzero(frame.wt_seq.to_numpy() == protein)
                    for protein in np.unique(frame.wt_seq)]
        rng = np.random.default_rng(0)
        boot_curves = np.empty((4000, 2, len(LEVELS)))
        for b in range(len(boot_curves)):
            sample = np.concatenate([clusters[k] for k in rng.integers(len(clusters), size=len(clusters))])
            boot_curves[b] = curves[:, sample, :].mean(axis=1)
        curve_ci = np.quantile(boot_curves, [.025, .975], axis=0)
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.6), layout='constrained')
        bins = np.histogram_bin_edges(widths.ravel(), bins='fd')
        for j, label in enumerate(('Standard CV+', 'Adaptive CV+')):
            axes[0, 0].plot(100 * LEVELS, 100 * curves[j].mean(axis=0),
                            color=COLORS[j], label=label)
            axes[0, 0].fill_between(100 * LEVELS, 100 * curve_ci[0, j],
                                    100 * curve_ci[1, j], color=COLORS[j], alpha=.12)
            axes[0, 1].hist(widths[j], bins=bins,
                            weights=np.full(len(y), 100 / len(y)), histtype='step',
                            lw=1.8, color=COLORS[j], label=label)
            axes[1, 1].plot(range(1, 6),
                            [100 * covered[j, groups == g].mean() for g in range(5)],
                            'o-', color=COLORS[j], label=label)
        axes[0, 0].plot([50, 100], [50, 100], '--', color='.5')
        axes[0, 0].set(title='A  Marginal coverage', xlabel='Nominal coverage (%)',
                       ylabel='Observed coverage (%)', xlim=(50, 100), ylim=(45, 100))
        axes[0, 1].set(title='B  Distribution of interval widths',
                       xlabel='90% prediction-interval width (kcal/mol)',
                       ylabel='Mutations per bin (%)')
        violin_points(axes[1, 0], np.arange(1, 6),
                      [change[error_groups == g] for g in range(5)], COLORS[1], .65)
        axes[1, 0].axhline(0, color=COLORS[0], ls='--')
        axes[1, 0].set(title='C  Width allocation vs observed error',
                       xlabel='Observed absolute-error quintile (post hoc)',
                       ylabel='Adaptive minus standard width (kcal/mol)', xticks=range(1, 6))
        axes[1, 1].axhline(90, color='.5', ls='--')
        axes[1, 1].set(title='D  Coverage by predicted uncertainty',
                       xlabel='Mean OOF sigma-percentile group',
                       ylabel='Observed coverage (%)', xticks=range(1, 6))
        for ax in axes.flat:
            ax.grid(alpha=.15)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(frameon=False, fontsize=8)
        stem = f'{name}_naive_uncertainty'
        if args.weighting == 'protein_empirical':
            stem += '_protein'
        if args.aggregation == 'mean':
            stem += '_mean'
        if args.weighting == 'protein_empirical':
            fig.suptitle(f'{name} · one-hot FFNN · equal-protein calibration', fontsize=10)
        for ext in ('png', 'pdf'):
            fig.savefig(out / f'{stem}.{ext}', dpi=300)
        plt.close(fig)
        point = dict(dataset=name, aggregation=args.aggregation, weighting=args.weighting, n_mutations=len(y),
                     n_proteins=len(clusters), pearson=pearsonr(y, mu).statistic,
                     spearman=spearmanr(y, mu).statistic,
                     mse=np.mean((y - mu) ** 2), mae=np.mean(error),
                     rho_uncertainty_error=spearmanr(rank, error).statistic,
                     adaptive_minus_standard_width=np.mean(change))
        rows.append(point)
        print(name, point, flush=True)
    summary_stem = 'paper_figure_summary'
    if args.weighting == 'protein_empirical':
        summary_stem += '_protein'
    if args.aggregation == 'mean':
        summary_stem += '_mean'
    pd.DataFrame(rows).to_csv(out / f'{summary_stem}.csv', index=False)


if __name__ == '__main__':
    main()
