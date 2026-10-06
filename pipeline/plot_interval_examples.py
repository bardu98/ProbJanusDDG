import os
"""Readable, width-stratified examples of protein-weighted adaptive CV+ intervals."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(os.environ.get("PROBJANUS_RUN_DIR", Path(__file__).resolve().parent)).resolve()
INTERVAL = 'adaptive_cvplus_protein_empirical_a0.1'
COLORS = ['#b3cde3', '#8cbae6', '#5d9bd3', '#3879b9', '#1e578c']


def select_evenly_by_width(width, n_per_group=7):
    """Fixed width-rank selection, independent of target and prediction error."""
    order = np.argsort(width, kind='stable')
    groups = np.array_split(order, 5)
    picks = []
    for q, group in enumerate(groups, start=1):
        ranks = np.rint(np.linspace(0, len(group) - 1, n_per_group)).astype(int)
        if len(np.unique(ranks)) != n_per_group:
            raise ValueError('Too few observations in a width group')
        picks.extend((q, int(group[r]), int(r + 1), len(group)) for r in ranks)
    return groups, picks


def render(name, aggregation):
    out = HERE / 'figures'
    out.mkdir(exist_ok=True)
    frame = pd.read_parquet(ROOT / 'data' / f'{name}.parquet').reset_index(drop=True)
    with np.load(HERE / 'predictions' / f'{name}_cvplus.npz') as z:
        y = z['y']
        folds = z['mu_folds']
        lo, hi = z[INTERVAL + '_lower'], z[INTERVAL + '_upper']
    np.testing.assert_array_equal(y, frame.ddG.to_numpy())
    point = folds.mean(axis=0) if aggregation == 'mean' else np.median(folds, axis=0)
    width = hi - lo
    groups, picks = select_evenly_by_width(width)
    rows = []
    for q, idx, rank, count in picks:
        rows.append(dict(dataset=name, aggregation=aggregation, width_quintile=q,
                         width_rank_in_quintile=rank, quintile_size=count,
                         row_id=idx, protein_id=frame.loc[idx, 'id'],
                         mutation=frame.loc[idx, 'mts'], observed=y[idx], predicted=point[idx],
                         lower=lo[idx], upper=hi[idx], interval_width=width[idx],
                         covered=bool(lo[idx] <= y[idx] <= hi[idx])))
    selected = pd.DataFrame(rows)
    if selected.row_id.nunique() != len(selected):
        raise AssertionError('Duplicate selected mutation')
    label = 'mean' if aggregation == 'mean' else 'median'
    fig, axes = plt.subplots(5, 2, figsize=(11.5, 11.2),
                             gridspec_kw={'width_ratios': [5.6, 1.7], 'wspace': .12},
                             sharex='col', layout='constrained')
    left = selected[['observed', 'lower']].min().min()
    right = selected[['observed', 'upper']].max().max()
    xlim = (float(left - .45), float(right + .45))
    for q in range(1, 6):
        sub = selected[selected.width_quintile == q].reset_index(drop=True)
        values = sub.interval_width.to_numpy()
        yy = np.arange(len(sub))[::-1]
        ax, bx = axes[q-1]
        color = COLORS[q-1]
        ax.hlines(yy, sub.lower, sub.upper, color=color, lw=3.2, zorder=1)
        ax.plot(sub.lower, yy, '|', color=color, ms=8, mew=1.4, zorder=2)
        ax.plot(sub.upper, yy, '|', color=color, ms=8, mew=1.4, zorder=2)
        ax.scatter(sub.predicted, yy, s=32, color='#142c44', zorder=4,
                   label=f'Prediction ({label})' if q == 1 else None)
        ax.scatter(sub.observed, yy, s=42, color='#c44e20', marker='x', linewidths=1.5,
                   zorder=5, label='Observed' if q == 1 else None)
        ax.set_yticks(yy, [f'{r.protein_id} {r.mutation}' for r in sub.itertuples()], fontsize=8)
        ax.tick_params(axis='y', length=0, pad=7)
        ax.set_xlim(xlim)
        ax.set_ylim(-.6, len(sub)-.4)
        ax.axvline(0, color='.8', lw=.8, zorder=0)
        ax.grid(axis='x', alpha=.18)
        whole = width[groups[q-1]]
        ax.set_title(f'Width quintile {q} · all {len(whole)} mutations: '
                     f'{whole.min():.2f}–{whole.max():.2f} kcal/mol',
                     loc='left', fontsize=9, pad=5)
        bx.barh(yy, values, color=color, height=.57, edgecolor='none')
        bx.set_yticks([])
        bx.set_ylim(ax.get_ylim())
        bx.set_xlim(0, max(9.4, width.max()+.9))
        bx.set_xticks([0, 4, 8])
        bx.grid(axis='x', alpha=.15)
        bx.set_axisbelow(True)
        for position, value in zip(yy, values):
            bx.text(value + .13, position, f'{value:.1f}', va='center', fontsize=7.3)
        for spine in ('top','right','left'):
            ax.spines[spine].set_visible(False)
            bx.spines[spine].set_visible(False)
    axes[0,0].legend(loc='upper left', frameon=False, fontsize=8)
    axes[-1,0].set_xlabel('ddG (kcal/mol) · horizontal line = 90% adaptive CV+ interval')
    axes[-1,1].set_xlabel('Interval width')
    fig.suptitle(f'{name}: mutation-level predictions and intervals\n'
                 f'Equal-protein calibration · {label} point prediction', fontsize=12)
    stem = f'{name}_adaptive_protein_interval_examples_{label}'
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'{stem}.{ext}', dpi=260, bbox_inches='tight')
    plt.close(fig)
    selected.to_csv(out / f'{stem}.csv', index=False)
    print(name, aggregation, 'saved', stem, 'selected', len(selected), 'of', len(y),
          'width range', (float(width.min()), float(width.max())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregation', choices=('median',), default='median')
    args = parser.parse_args()
    for name in ('S669L', 'S461L'):
        render(name, args.aggregation)


if __name__ == '__main__':
    main()
