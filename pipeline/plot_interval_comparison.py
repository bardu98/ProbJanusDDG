import os
"""Paired standard/adaptive CV+ intervals on readable, width-stratified examples."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from intervals_for_plot import bounds

from plot_interval_examples import select_evenly_by_width

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(os.environ.get("PROBJANUS_RUN_DIR", Path(__file__).resolve().parent)).resolve()
ADAPTIVE = 'adaptive_cvplus_protein_empirical_a0.1'
STANDARD = 'standard_cvplus_protein_empirical_a0.1'
BLUE = '#2473ad'
GRAY = '#858585'
ORANGE = '#c44e20'


def render(name, aggregation, level=90):
    frame = pd.read_parquet(ROOT / 'data' / f'{name}.parquet').reset_index(drop=True)
    with np.load(HERE / 'predictions' / f'{name}_cvplus.npz') as z:
        y = z['y']
        fold_mu = z['mu_folds']
    sl, su, al, au = bounds(name, level)
    np.testing.assert_array_equal(y, frame.ddG.to_numpy())
    mu = fold_mu.mean(axis=0) if aggregation == 'mean' else np.median(fold_mu, axis=0)
    wa, ws = au-al, su-sl
    groups, picks = select_evenly_by_width(wa)
    rows = []
    for q, idx, rank, count in picks:
        rows.append(dict(dataset=name, aggregation=aggregation, nominal_percent=level, adaptive_width_quintile=q,
                         rank_in_quintile=rank, quintile_size=count, row_id=idx,
                         protein_id=frame.loc[idx,'id'], mutation=frame.loc[idx,'mts'],
                         predicted=mu[idx], observed=y[idx],
                         adaptive_lower=al[idx], adaptive_upper=au[idx], adaptive_width=wa[idx],
                         standard_lower=sl[idx], standard_upper=su[idx], standard_width=ws[idx],
                         adaptive_covered=bool(al[idx]<=y[idx]<=au[idx]),
                         standard_covered=bool(sl[idx]<=y[idx]<=su[idx])))
    selected = pd.DataFrame(rows)
    assert selected.row_id.nunique() == len(selected) == 35
    fig, axes = plt.subplots(5, 2, figsize=(11.8, 11.4),
                             gridspec_kw={'width_ratios':[5.8,1.6], 'wspace':.11},
                             sharex='col', layout='constrained')
    left = selected[['observed','adaptive_lower','standard_lower']].min().min()
    right = selected[['observed','adaptive_upper','standard_upper']].max().max()
    xlim = (float(left-.45), float(right+.45))
    max_width = float(max(wa.max(),ws.max()))
    for q in range(1,6):
        sub = selected[selected.adaptive_width_quintile==q].reset_index(drop=True)
        yy = np.arange(len(sub))[::-1]
        ax, bx = axes[q-1]
        # Slight vertical separation keeps both true interval endpoints visible.
        ax.hlines(yy+.18, sub.standard_lower, sub.standard_upper,
                  color=GRAY, lw=2, zorder=1)
        ax.plot(sub.standard_lower, yy+.18, '|', color=GRAY, ms=6, mew=1.1, zorder=2)
        ax.plot(sub.standard_upper, yy+.18, '|', color=GRAY, ms=6, mew=1.1, zorder=2)
        ax.hlines(yy-.18, sub.adaptive_lower, sub.adaptive_upper,
                  color=BLUE, lw=3, zorder=1)
        ax.plot(sub.adaptive_lower, yy-.18, '|', color=BLUE, ms=7, mew=1.2, zorder=2)
        ax.plot(sub.adaptive_upper, yy-.18, '|', color=BLUE, ms=7, mew=1.2, zorder=2)
        ax.scatter(sub.predicted, yy, s=33, color='#142c44', zorder=4)
        ax.scatter(sub.observed, yy, s=42, color=ORANGE, marker='x', linewidths=1.5,zorder=5)
        ax.set_yticks(yy,[f'{r.protein_id} {r.mutation}' for r in sub.itertuples()],fontsize=8)
        ax.tick_params(axis='y',length=0,pad=7)
        ax.set_xlim(xlim)
        ax.set_ylim(-.6,len(sub)-.4)
        ax.axvline(0,color='.83',lw=.8,zorder=0)
        ax.grid(axis='x',alpha=.16)
        whole = wa[groups[q-1]]
        ax.set_title(f'Adaptive width quintile {q} · all {len(whole)} mutations: '
                     f'{whole.min():.2f}–{whole.max():.2f} kcal/mol',
                     loc='left',fontsize=9,pad=5)
        bx.barh(yy+.18,sub.standard_width,color=GRAY,height=.24)
        bx.barh(yy-.18,sub.adaptive_width,color=BLUE,height=.28)
        bx.set_ylim(ax.get_ylim())
        bx.set_xlim(0,max(9.2,max_width+1.2) if level==90 else max_width*1.22)
        bx.set_yticks([])
        bx.set_xticks([0,4,8]) if level==90 else bx.xaxis.set_major_locator(plt.MaxNLocator(3))
        bx.grid(axis='x',alpha=.15)
        bx.set_axisbelow(True)
        for pos,value in zip(yy,sub.adaptive_width):
            bx.text(value+.12,pos-.18,f'{value:.1f}',va='center',fontsize=7.2,color=BLUE)
        for spine in ('top','right','left'):
            ax.spines[spine].set_visible(False)
            bx.spines[spine].set_visible(False)
    legend = [Line2D([0],[0],color=GRAY,lw=2,label='Standard CV+'),
              Line2D([0],[0],color=BLUE,lw=3,label='Adaptive CV+'),
              Line2D([0],[0],marker='o',color='none',markerfacecolor='#142c44',
                     markeredgecolor='#142c44',markersize=5,label=f'Prediction ({aggregation})'),
              Line2D([0],[0],marker='x',color=ORANGE,lw=0,markersize=6,label='Observed')]
    axes[0,0].legend(handles=legend,loc='upper left',frameon=False,fontsize=7.5)
    axes[-1,0].set_xlabel('ddG (kcal/mol) · true CV+ interval endpoints')
    axes[-1,1].set_xlabel('Interval width')
    fig.suptitle(f'{name}: standard vs adaptive {level}% CV+ intervals\n'
                 f'Equal-protein calibration · {aggregation} point prediction',fontsize=12)
    out = HERE/'figures'
    stem = f'{name}_adaptive_vs_standard_protein_{aggregation}' + ('' if level==90 else f'_{level}pct')
    for ext in ('png','pdf'):
        fig.savefig(out/f'{stem}.{ext}',dpi=260,bbox_inches='tight')
    plt.close(fig)
    selected.to_csv(out/f'{stem}.csv',index=False)
    print(name,aggregation,'standard width',float(ws.min()),float(ws.max()),
          'adaptive width',float(wa.min()),float(wa.max()),'saved',stem,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregation',choices=('median',),default='median')
    parser.add_argument('--level',type=int,default=90)
    args=parser.parse_args()
    for name in ('S669L','S461L'):
        render(name,args.aggregation,args.level)


if __name__=='__main__':
    main()
