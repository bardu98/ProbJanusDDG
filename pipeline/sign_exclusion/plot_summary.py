import os
"""Plot direction-call yield, observed sign accuracy, and protein reach."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(os.environ.get("PROBJANUS_RUN_DIR", ROOT/"pipeline")).resolve()
HERE = RUN/"sign_exclusion"
(RUN/"figures").mkdir(parents=True,exist_ok=True)
LEVELS = [50, 60, 70, 80, 85, 90, 95]
TEAL, RED, INK, GRAY = '#168b8a', '#db735c', '#19364b', '#718391'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                     'text.color':INK, 'axes.labelcolor':INK, 'xtick.color':GRAY,
                     'ytick.color':INK, 'pdf.fonttype':42, 'ps.fonttype':42})
table = pd.read_csv(HERE/'summary.csv')
fig = plt.figure(figsize=(12, 7.8), facecolor='white')
gs = fig.add_gridspec(2, 3, width_ratios=[3.5, 1.5, 2.3],
                      left=.09, right=.975, top=.86, bottom=.08,
                      wspace=.18, hspace=.55)
for row, name in enumerate(['S669L','S461L']):
    d = table[(table.dataset==name)&table.nominal_percent.isin(LEVELS)].set_index('nominal_percent').loc[LEVELS]
    n, p = int(d.n_mutations.iloc[0]), int(d.n_proteins.iloc[0])
    y = np.arange(len(d))
    axes = [fig.add_subplot(gs[row,k]) for k in range(3)]
    for ax in axes:
        ax.set_ylim(len(d)-.4, -.75)
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.tick_params(length=0)
        for j in y[::2]: ax.axhspan(j-.43,j+.43,color='#f3f7f8',zorder=0)
    a,b,c = axes
    good=100*d.correct_sign_mutations/n
    bad=100*(d.zero_excluded_mutations-d.correct_sign_mutations)/n
    np.testing.assert_allclose(good+bad, d.zero_excluded_percent)
    a.barh(y, good, height=.50, color=TEAL,zorder=3)
    a.barh(y, bad, left=good, height=.50, color=RED,zorder=3)
    for j,r in enumerate(d.itertuples()):
        a.text(r.zero_excluded_percent+.8,j, f'{r.zero_excluded_percent:.1f}% ({r.zero_excluded_mutations})',va='center',fontsize=9, color=INK)
    a.set_xlim(0,56)
    a.set_xticks([0,10,20,30,40,50],['0','10','20','30','40','50'])
    a.set_yticks(y,[f'{v}%' for v in LEVELS])
    a.set_ylabel('Nominal interval level',labelpad=12)
    a.set_xlabel('Mutations with a direction call (%)',labelpad=9)
    a.set_axisbelow(True)
    a.grid(axis='x',color='#e4ebee',lw=.7)
    a.text(0,1.18,f'{chr(65+row)}   {name}',transform=a.transAxes,fontsize=12,weight='bold')
    a.text(.39,1.18,f'{n} mutations · {p} proteins',transform=a.transAxes,fontsize=10,color=GRAY)
    b.set_xlim(0,1); b.set_xticks([]); b.set_yticks([])
    b.set_title('Sign accuracy',fontsize=11,weight='bold',pad=12)
    for j,r in enumerate(d.itertuples()):
        if r.zero_excluded_mutations:
            b.text(.5,j-.17,f'{r.sign_accuracy_percent:.1f}%',ha='center',va='center',fontsize=12,weight='bold',color=TEAL)
            b.text(.5,j+.25,f'{r.correct_sign_mutations}/{r.zero_excluded_mutations} calls',ha='center',va='center',fontsize=8,color=GRAY)
        else:
            b.text(.5,j,'—',ha='center',va='center',fontsize=12,color=GRAY)
    c.set_xlim(0,82); c.set_yticks([])
    c.set_xticks([0,20,40,60])
    c.set_title('Proteins with ≥1 call',fontsize=11,weight='bold',pad=12)
    c.barh(y,d.proteins_with_any_zero_excluded_percent,height=.5,color='#b3c6d6',zorder=3)
    for j,r in enumerate(d.itertuples()):
        c.text(r.proteins_with_any_zero_excluded_percent+1.2,j,
               f'{r.proteins_with_any_zero_excluded_percent:.1f}%',va='center',fontsize=9)
    c.set_xlabel('All test proteins (%)',labelpad=9)
    c.set_axisbelow(True); c.grid(axis='x',color='#e4ebee',lw=.7)
fig.legend(handles=[Patch(color=TEAL,label='Correct sign'),Patch(color=RED,label='Incorrect sign')],
           loc='upper right',bbox_to_anchor=(.975,.995),frameon=False,ncol=2,fontsize=10)
for ext in ['pdf','png','svg']:
    target=RUN/'figures'/f'sign_exclusion_summary.{ext}'
    fig.savefig(target,dpi=240,facecolor='white')
plt.close(fig)
(HERE/'figure_caption.txt').write_text('Direction-call yield and observed sign accuracy at selected nominal levels. A direction call is made when the adaptive protein-weighted CV+ interval lies strictly above or below zero. Left: the percentage of all test mutations receiving a call, split into correct and incorrect signs; labels also give the number of calls. Center: observed sign accuracy among called mutations, with correct/total counts. Right: the percentage of distinct full-WT proteins with at least one call. Observed zero targets count as incorrect. A dash indicates undefined accuracy when no calls are made. Calibration weights each protein equally; sign accuracy weights each called mutation equally. S461L overlaps S669L, so the panels are not independent replications.\n')
print('Saved PNG, PDF, SVG.')
