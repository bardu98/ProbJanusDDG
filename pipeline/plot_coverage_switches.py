import os
"""Show every coverage switch between protein-weighted standard/adaptive CV+."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from intervals_for_plot import bounds

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(os.environ.get("PROBJANUS_RUN_DIR", Path(__file__).resolve().parent)).resolve()
BASE = 'cvplus_protein_empirical_a0.1'
GRAY = '#858585'
BLUE = '#2473ad'
ORANGE = '#c44e20'


def render(name, aggregation, level=90):
    frame = pd.read_parquet(ROOT/'data'/f'{name}.parquet').reset_index(drop=True)
    with np.load(HERE/'predictions'/f'{name}_cvplus.npz') as z:
        y = z['y']
        fold_mu = z['mu_folds']
    sl, su, al, au = bounds(name, level)
    np.testing.assert_array_equal(y,frame.ddG.to_numpy())
    point = fold_mu.mean(axis=0) if aggregation=='mean' else np.median(fold_mu,axis=0)
    smargin = np.minimum(y-sl,su-y)
    amargin = np.minimum(y-al,au-y)
    std_hit = smargin>=0
    adapt_hit = amargin>=0
    rescued = np.flatnonzero(~std_hit & adapt_hit)
    lost = np.flatnonzero(std_hit & ~adapt_hit)
    rows=[]
    for kind, indices in [('rescued',rescued),('lost',lost)]:
        # Largest margin change first; all switch cases are shown.
        for idx in sorted(indices,key=lambda i:abs(amargin[i]-smargin[i]),reverse=True):
            rows.append(dict(dataset=name,aggregation=aggregation,nominal_percent=level,kind=kind,
                             row_id=int(idx),protein_id=frame.loc[idx,'id'],
                             mutation=frame.loc[idx,'mts'],observed=y[idx],predicted=point[idx],
                             standard_lower=sl[idx],standard_upper=su[idx],
                             adaptive_lower=al[idx],adaptive_upper=au[idx],
                             standard_width=su[idx]-sl[idx],adaptive_width=au[idx]-al[idx],
                             standard_margin=smargin[idx],adaptive_margin=amargin[idx]))
    selected=pd.DataFrame(rows)
    if selected.empty:
        raise ValueError(f'No coverage switches for {name}')
    selected['standard_covered']=selected.standard_margin>=0
    selected['adaptive_covered']=selected.adaptive_margin>=0
    out=HERE/'figures'
    stem=f'{name}_coverage_switches_protein_{aggregation}' + ('' if level==90 else f'_{level}pct')
    selected.to_csv(out/f'{stem}.csv',index=False)
    n=len(selected)
    gap=.8 if len(rescued) and len(lost) else 0
    positions=np.arange(n)[::-1].astype(float)
    positions[len(rescued):]-=gap
    fig,(ax,bx)=plt.subplots(1,2,figsize=(11.3,max(4.4,1.0*n+.9) if level==90 else max(5.,.48*n+2.)),
                             gridspec_kw={'width_ratios':[5.7,2.0],'wspace':.15},
                             layout='constrained')
    for r,pos in zip(selected.itertuples(),positions):
        ax.hlines(pos+.16,r.standard_lower,r.standard_upper,color=GRAY,lw=2.2,zorder=1)
        ax.hlines(pos-.16,r.adaptive_lower,r.adaptive_upper,color=BLUE,lw=3,zorder=1)
        for lo,hi,offset,color in [(r.standard_lower,r.standard_upper,.16,GRAY),
                                   (r.adaptive_lower,r.adaptive_upper,-.16,BLUE)]:
            ax.plot([lo,hi],[pos+offset]*2,'|',color=color,ms=7,mew=1.2,zorder=2)
        ax.scatter(r.predicted,pos,s=35,color='#142c44',zorder=4)
        ax.scatter(r.observed,pos,s=50,marker='x',color=ORANGE,linewidths=1.6,zorder=5)
        bx.plot([r.standard_margin,r.adaptive_margin],[pos,pos],color='.7',lw=1,zorder=1)
        bx.scatter(r.standard_margin,pos,s=38,color=GRAY,zorder=3)
        bx.scatter(r.adaptive_margin,pos,s=45,color=BLUE,zorder=4)
    labels=[f'{r.protein_id} {r.mutation}' for r in selected.itertuples()]
    ax.set_yticks(positions,labels,fontsize=9)
    ax.tick_params(axis='y',length=0,pad=7)
    bx.set_yticks([])
    ax.set_ylim(positions[-1]-.65,positions[0]+.7)
    bx.set_ylim(ax.get_ylim())
    left=selected[['observed','standard_lower','adaptive_lower']].min().min()
    right=selected[['observed','standard_upper','adaptive_upper']].max().max()
    ax.set_xlim(float(left-.4),float(right+.4))
    ax.axvline(0,color='.82',lw=.8)
    bx.axvspan(0,max(.1,selected[['standard_margin','adaptive_margin']].max().max()+.15),
               color='#eaf5ec',zorder=0)
    bx.axvline(0,color='.35',lw=1.2,zorder=2)
    margin_min=selected[['standard_margin','adaptive_margin']].min().min()
    margin_max=selected[['standard_margin','adaptive_margin']].max().max()
    bx.set_xlim(float(margin_min-.12),float(margin_max+.15))
    if len(rescued) and len(lost):
        separator=(positions[len(rescued)-1]+positions[len(rescued)])/2
        for axis in (ax,bx):axis.axhline(separator,color='.65',lw=.9,ls='--')
    for axis in (ax,bx):
        axis.grid(axis='x',alpha=.15)
        for spine in ('top','right','left'):axis.spines[spine].set_visible(False)
    handles=[Line2D([0],[0],color=GRAY,lw=2,label='Standard CV+'),
             Line2D([0],[0],color=BLUE,lw=3,label='Adaptive CV+'),
             Line2D([0],[0],marker='o',color='none',markerfacecolor='#142c44',
                    markeredgecolor='#142c44',markersize=5,label=f'Prediction ({aggregation})'),
             Line2D([0],[0],marker='x',color=ORANGE,lw=0,markersize=6,label='Observed')]
    ax.legend(handles=handles,loc='upper right',frameon=False,fontsize=8,ncol=2)
    ax.set_xlabel(f'ddG (kcal/mol) · actual {level}% interval endpoints')
    bx.set_xlabel('Margin (kcal/mol); >0 = covered')
    ax.set_title('Intervals on the same mutation',loc='left',fontsize=10)
    bx.set_title('Distance to nearest interval edge',loc='left',fontsize=10)
    fig.suptitle(f'{name}: all coverage switches at nominal {level}%\n'
                 f'Equal-protein calibration · rescued {len(rescued)} above line, lost {len(lost)} below',
                 fontsize=12)
    for ext in ('png','pdf'):
        fig.savefig(out/f'{stem}.{ext}',dpi=260,bbox_inches='tight')
    plt.close(fig)
    print(name,aggregation,'rescued',len(rescued),'lost',len(lost),'saved',stem,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregation',choices=('median',),default='median')
    parser.add_argument('--level',type=int,default=90)
    args=parser.parse_args()
    for name in ('S669L','S461L'):
        render(name,args.aggregation,args.level)


if __name__=='__main__':main()
