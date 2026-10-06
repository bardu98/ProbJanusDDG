import os
"""Paired descriptive evaluation, using exactly the same protein-weighted CV+."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
PARENT=ROOT/"reference"
HERE=Path(os.environ.get("NAIVE_RUN_DIR", ROOT/"runs/naive")).resolve()
sys.path[:0]=[str(ROOT/'conformal'),str(ROOT)]
from protein_balanced import weighted_bounds
from cvplus import metrics


def generate_comparison():
    train=pd.read_parquet(ROOT/'data/S2450.parquet').reset_index(drop=True)
    weights=(1/(train.wt_seq.nunique()*train.groupby('wt_seq').wt_seq.transform('size'))).to_numpy()
    np.testing.assert_allclose(pd.Series(weights).groupby(train.wt_seq).sum(),1/train.wt_seq.nunique())
    levels=np.arange(50,96)/100
    points=[]; records=[]; signs=[]
    colors={'ProbJanusDDG':'#2473ad','One-hot FFNN':'#d47a35'}
    store={}
    for model,folder in [('ProbJanusDDG',PARENT),('One-hot FFNN',HERE)]:
        oof=dict(np.load(folder/'predictions/oof_all.npz'))
        np.testing.assert_array_equal(oof['row_id'],np.arange(len(train)))
        np.testing.assert_array_equal(oof['y'],train.ddG)
        np.testing.assert_array_equal(oof['fold_id'],train.cvfold)
        for name in ('S669L','S461L'):
            frame=pd.read_parquet(ROOT/'data'/f'{name}.parquet').reset_index(drop=True)
            z=dict(np.load(folder/'predictions'/f'{name}_cvplus.npz'))
            y=frame.ddG.to_numpy(); np.testing.assert_array_equal(y,z['y'])
            store[model,name]=z
            for aggregation,mu in [('median',np.median(z['mu_folds'],axis=0)),('mean',np.mean(z['mu_folds'],axis=0))]:
                err=y-mu
                points.append(dict(model=model,dataset=name,aggregation=aggregation,
                    pearson=pearsonr(y,mu).statistic,spearman=spearmanr(y,mu).statistic,
                    mse=np.mean(err**2),rmse=np.sqrt(np.mean(err**2)),mae=np.mean(abs(err)),
                    mse_equal_protein=pd.Series(err**2).groupby(frame.wt_seq).mean().mean()))
            for adaptive,method in [(False,'standard'),(True,'adaptive')]:
                lo,hi=weighted_bounds(oof,z['mu_folds'],z['sigma_folds'],weights,levels,adaptive)
                # Existing Janus/naive 80/90/95 endpoints must reproduce bit for bit.
                for pct,alpha in [(80,.2),(90,.1),(95,.05)]:
                    tag=f'{method}_cvplus_protein_empirical_a{alpha}'
                    np.testing.assert_array_equal(lo[:,pct-50],z[tag+'_lower'])
                    np.testing.assert_array_equal(hi[:,pct-50],z[tag+'_upper'])
                np.testing.assert_array_less(-1e-10,np.diff(hi,axis=1))
                assert (np.diff(lo,axis=1)<=1e-10).all()
                for j,level in enumerate(levels):
                    covered=(y>=lo[:,j])&(y<=hi[:,j]); width=hi[:,j]-lo[:,j]
                    row=dict(model=model,dataset=name,method=method,nominal=level,
                        **metrics(y,lo[:,j],hi[:,j],1-level),
                        coverage_equal_protein=pd.Series(covered).groupby(frame.wt_seq).mean().mean(),
                        width_equal_protein=pd.Series(width).groupby(frame.wt_seq).mean().mean(),
                        n_mutations=len(frame),n_proteins=frame.wt_seq.nunique())
                    records.append(row)
                    if adaptive:
                        pos=lo[:,j]>0; neg=hi[:,j]<0; called=pos|neg
                        correct=(pos&(y>0))|(neg&(y<0))
                        signs.append(dict(model=model,dataset=name,nominal=level,
                            n_calls=called.sum(),call_percent=100*called.mean(),n_correct=correct.sum(),
                            sign_accuracy_percent=100*correct.sum()/called.sum() if called.any() else np.nan))
                if model=='One-hot FFNN':
                    np.savez_compressed(HERE/'predictions'/f'{name}_{method}_protein_grid.npz',
                        levels=levels,y=y,lower=lo,upper=hi,mu_median=np.median(z['mu_folds'],axis=0))
    pt=pd.DataFrame(points); cv=pd.DataFrame(records); sg=pd.DataFrame(signs)
    pt.to_csv(HERE/'comparison_point_metrics.csv',index=False)
    cv.to_csv(HERE/'comparison_intervals.csv',index=False)
    sg.to_csv(HERE/'comparison_sign_calls.csv',index=False)
    cv[cv.nominal.isin([.5,.8,.9,.95])].to_csv(HERE/'comparison_intervals_key_levels.csv',index=False)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    for name in ('S669L','S461L'):
        fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
        for model in colors:
            for method,style in [('standard','--'),('adaptive','-')]:
                sub=cv[(cv.model==model)&(cv.dataset==name)&(cv.method==method)]
                label=f'{model}, {method}'
                for col,(metric,ylabel) in enumerate([('coverage','Mutation coverage (%)'),('coverage_equal_protein','Equal-protein coverage (%)'),('mean_width','Mean interval width (kcal/mol)')]):
                    axes[0,col].plot(100*sub.nominal,sub[metric]*(100 if col<2 else 1),style,color=colors[model],label=label)
                    axes[0,col].set(xlabel='Nominal level (%)',ylabel=ylabel)
            part=sg[(sg.model==model)&(sg.dataset==name)]
            axes[1,0].plot(100*part.nominal,part.call_percent,color=colors[model],label=model)
            axes[1,1].plot(100*part.nominal,part.sign_accuracy_percent,color=colors[model])
            axes[1,2].plot(part.call_percent,part.sign_accuracy_percent,color=colors[model])
            for level in [.5,.8,.9]:
                r=part[np.isclose(part.nominal,level)].iloc[0]
                if r.n_calls:
                    axes[1,2].scatter(r.call_percent,r.sign_accuracy_percent,color=colors[model],s=22)
                    axes[1,2].annotate(f'{int(level*100)}%',(r.call_percent,r.sign_accuracy_percent),xytext=(3,4),textcoords='offset points',fontsize=7,color=colors[model])
        for ax in axes[0,:2]: ax.plot([50,95],[50,95],':',color='.5',lw=1)
        axes[1,0].set(xlabel='Nominal level (%)',ylabel='Mutations with sign calls (%)')
        axes[1,1].set(xlabel='Nominal level (%)',ylabel='Observed sign accuracy (%)')
        axes[1,2].set(xlabel='Mutations with sign calls (%)',ylabel='Observed sign accuracy (%)')
        for ax in axes[1,1:]: ax.set_ylim(0,103)
        for label,ax in zip('ABCDEF',axes.flat):
            ax.text(-.14,1.03,label,transform=ax.transAxes,fontweight='bold'); ax.grid(alpha=.18)
        axes[0,2].legend(fontsize=7,loc='upper left')
        axes[1,0].legend(fontsize=8)
        for ext in ['png','pdf']: fig.savefig(HERE/'figures'/f'{name}_comparison.{ext}',dpi=200)
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9,8),layout='constrained')
    for row,name in enumerate(['S669L','S461L']):
        for col,model in enumerate(colors):
            z=store[model,name]; y=z['y']; mu=np.median(z['mu_folds'],axis=0)
            ax=axes[row,col]; ax.scatter(y,mu,s=11,alpha=.35,color=colors[model],edgecolors='none')
            low=min(y.min(),mu.min()); high=max(y.max(),mu.max())
            ax.plot([low,high],[low,high],':',color='.5')
            r=pt[(pt.model==model)&(pt.dataset==name)&(pt.aggregation=='median')].iloc[0]
            ax.set(title=f'{name} · {model}',xlabel='Observed ΔΔG (kcal/mol)',ylabel='Predicted ΔΔG (kcal/mol)')
            ax.text(.04,.96,f'r = {r.pearson:.3f}\nρ = {r.spearman:.3f}\nMSE = {r.mse:.3f}',transform=ax.transAxes,va='top')
    for ext in ['png','pdf']: fig.savefig(HERE/'figures'/f'point_predictions.{ext}',dpi=200)
    plt.close(fig)
    def md(frame):
        cols=list(frame.columns); lines=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
        for vals in frame.itertuples(index=False,name=None):
            lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in vals)+' |')
        return '\n'.join(lines)
    epochs=pd.read_csv(HERE/'selected_epochs.csv')
    text='''# Baseline FFNN one-hot: confronto con ProbJanusDDG

Sequenze complete one-hot: FFNN condivisa 42→64→64 con ReLU sui residui (WT, WT−mutante, distanza relativa firmata e assoluta), pooling medio/massimo mascherato e vettore del sito mutato, readout 192→64→2. Nessun taglio delle sequenze. Nessun ESM, convoluzione o attention. Training direzionale con inversi e Gaussian NLL (beta=0); inferenza con media antisimmetrica e scala simmetrica positiva. Adam 1e-4, batch 6; GPU float32 (Janus usava GPU bfloat16).

Stessi dati e split verificati tramite hash e uguaglianza del piano. Per ogni fold esterno, tre fold per training interno, uno per MSE medio per proteina su 300 epoche; refit da zero sui quattro fold per l'epoca selezionata. Ogni fold esterno è escluso da training e selezione della propria epoca. Semi identici al protocollo Janus. Predizione principale: mediana dei cinque refit. Media riportata come riferimento.

Il CV+ usa la stessa funzione `weighted_bounds` di Janus, con pesi 1/(P*n_p), quantili empirici dei candidati e accoppiamento tra residui OOF e predizioni di test dello stesso refit. Standard e adattivo, livelli 50–95%. Nessun intervallo è ricentrato sulla mediana. La variante pesata resta empirica, senza una nuova garanzia esatta dimostrata. Coverage per mutazione e coverage media per proteina sono riportate separatamente.

Questo confronto cambia sia architettura sia rappresentazione con aggregazione globale senza interazioni esplicite tra residui: non isola il solo effetto degli embedding. L'architettura e gli iperparametri sono fissati prima della valutazione del baseline; i benchmark erano già stati esaminati per Janus. S461L si sovrappone a S669L; non sono repliche indipendenti. I confronti sono descrittivi, senza test di significatività.

## Epoche selezionate

'''+md(epochs)+'\n\n## Performance puntuale (mediana)\n\n'+md(pt[pt.aggregation=='median'])
    text+='\n\n## Intervalli al 90%\n\n'+md(cv[np.isclose(cv.nominal,.9)][['model','dataset','method','coverage','coverage_equal_protein','mean_width','interval_score']])
    text+='\n\n## Segno: livelli 50%, 80%, 90%\n\n'+md(sg[sg.nominal.isin([.5,.8,.9])])
    text+='\n\n## Figure\n\n- `figures/S669L_comparison.pdf` e `S461L_comparison.pdf`: copertura, ampiezza, chiamate di segno e accuratezza.\n- `figures/point_predictions.pdf`: osservato contro predetto.\n- `figures/epoch_selection.pdf`: selezione delle epoche per fold.\n'
    (HERE/'report.md').write_text(text)
    (HERE/'comparison_verification.json').write_text(json.dumps(dict(same_cvplus_function=True,
        existing_endpoints_reproduced=True,protein_weights_equal=True,intervals_nested=True,
        point_metric_rows=len(pt),interval_metric_rows=len(cv),sign_rows=len(sg)),indent=2)+'\n')
    print(pt[pt.aggregation=='median'].to_string(index=False),flush=True)

if __name__=='__main__': generate_comparison()
