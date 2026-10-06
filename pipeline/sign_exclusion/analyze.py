import os
"""Direction calls from protein-weighted adaptive CV+ intervals, 50--95%."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RUN=Path(os.environ.get("PROBJANUS_RUN_DIR", ROOT/"pipeline")).resolve()
OUT=RUN/"sign_exclusion"
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'conformal'))
from protein_balanced import weighted_bounds


def main():
    oof=dict(np.load(RUN/'predictions/oof_all.npz'))
    train=pd.read_parquet(ROOT/'data/S2450.parquet').reset_index(drop=True)
    np.testing.assert_array_equal(oof['row_id'],np.arange(len(train)))
    np.testing.assert_array_equal(oof['y'],train.ddG)
    np.testing.assert_array_equal(oof['fold_id'],train.cvfold)
    weights=(1/(train.wt_seq.nunique()*train.groupby('wt_seq').wt_seq.transform('size'))).to_numpy()
    np.testing.assert_allclose(pd.Series(weights).groupby(train.wt_seq).sum(),1/train.wt_seq.nunique())
    levels=np.arange(50,96)/100
    summaries=[]
    for name in ('S669L','S461L'):
        d=pd.read_parquet(ROOT/'data'/f'{name}.parquet').reset_index(drop=True)
        z=dict(np.load(RUN/'predictions'/f'{name}_cvplus.npz'))
        np.testing.assert_array_equal(z['y'],d.ddG)
        mu=np.median(z['mu_folds'],axis=0)
        np.testing.assert_array_equal(mu,z['mu_ensemble'])
        lo,hi=weighted_bounds(oof,z['mu_folds'],z['sigma_folds'],weights,levels,True)
        assert (lo<=hi).all()
        assert (np.diff(lo,axis=1)<=1e-12).all()
        assert (np.diff(hi,axis=1)>=-1e-12).all()
        for percent,alpha in ((80,.2),(90,.1),(95,.05)):
            stem=f'adaptive_cvplus_protein_empirical_a{alpha}'
            np.testing.assert_array_equal(lo[:,percent-50],z[stem+'_lower'])
            np.testing.assert_array_equal(hi[:,percent-50],z[stem+'_upper'])
        positive=lo>0
        negative=hi<0
        definite=positive|negative
        assert not (positive&negative).any()
        assert (np.diff(definite.astype(int),axis=1)<=0).all()
        group,unique=pd.factorize(d.wt_seq,sort=True)
        nprot=len(unique)
        details=[]
        for j,level in enumerate(levels):
            pos=positive[:,j];neg=negative[:,j];call=definite[:,j]
            correct=(pos & (d.ddG.to_numpy()>0)) | (neg & (d.ddG.to_numpy()<0))
            opposite=(pos & (d.ddG.to_numpy()<0)) | (neg & (d.ddG.to_numpy()>0))
            neutral=call & (d.ddG.to_numpy()==0)
            assert np.isfinite(d.ddG.to_numpy()).all()
            assert int(correct.sum()+opposite.sum()+neutral.sum())==int(call.sum())
            accuracy=100*float(correct.sum())/int(call.sum()) if call.any() else np.nan
            protein_pos=len(np.unique(group[pos]));protein_neg=len(np.unique(group[neg]))
            protein_any=len(np.unique(group[call]))
            by_protein=pd.Series(call).groupby(group).mean()
            summaries.append(dict(dataset=name,nominal_percent=int(round(level*100)),
                n_mutations=len(d),n_proteins=nprot,
                stabilizing_mutations=int(pos.sum()),stabilizing_percent=100*float(pos.mean()),
                destabilizing_mutations=int(neg.sum()),destabilizing_percent=100*float(neg.mean()),
                zero_excluded_mutations=int(call.sum()),zero_excluded_percent=100*float(call.mean()),
                indeterminate_mutations=int((~call).sum()),
                correct_sign_mutations=int(correct.sum()),opposite_sign_mutations=int(opposite.sum()),
                zero_target_calls=int(neutral.sum()),sign_accuracy_percent=accuracy,
                proteins_with_stabilizing=protein_pos,proteins_with_destabilizing=protein_neg,
                proteins_with_any_zero_excluded=protein_any,proteins_with_any_zero_excluded_percent=100*protein_any/nprot,
                zero_excluded_fraction_equal_protein_percent=100*float(by_protein.mean())))
            details.append(pd.DataFrame(dict(dataset=name,nominal_percent=int(round(level*100)),
                row_id=np.arange(len(d)),protein_group=group,protein_id=d.id,mutation=d.mts,
                mu_median=mu,observed_ddG=d.ddG,lower=lo[:,j],upper=hi[:,j],
                sign_correct=np.where(call,correct.astype(float),np.nan),
                opposite_sign=opposite,
                classification=np.where(pos,'stabilizing',np.where(neg,'destabilizing','indeterminate')))))
        pd.concat(details,ignore_index=True).to_csv(OUT/f'{name}_mutation_classifications.csv',index=False)
        np.savez_compressed(OUT/f'{name}_intervals.npz',nominal=levels,row_id=np.arange(len(d)),
                            protein_group=group,mu_median=mu,lower=lo,upper=hi,
                            classification=positive.astype(np.int8)-negative.astype(np.int8))
    table=pd.DataFrame(summaries)
    table.to_csv(OUT/'summary.csv',index=False)
    columns=['nominal_percent','stabilizing_mutations','destabilizing_mutations',
             'zero_excluded_mutations','zero_excluded_percent','proteins_with_any_zero_excluded',
             'proteins_with_any_zero_excluded_percent','sign_accuracy_percent']
    lines=['# Interval exclusion of zero','',
        'Adaptive empirical CV+, equal protein weights in calibration; median point prediction.',
        'Stabilizing: lower > 0. Destabilizing: upper < 0. Touching zero is indeterminate.',
        'Mutation percentages use all mutations. Protein percentages use all unique WT sequences,',
        'and count proteins with at least one mutation whose interval excludes zero.',
        'Sign accuracy = 100 * correct-sign calls / all zero-excluding calls (equal mutation weights).',
        'Observed zero targets count as incorrect; no calls gives undefined accuracy (--).','']
    for name in ('S669L','S461L'):
        sub=table[table.dataset==name]
        lines+=['## '+name,'','| Nominal % | Stabilizing | Destabilizing | Total mutations | % mutations | Proteins with >=1 | % proteins | Correct sign (%) |',
               '|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in sub[columns].itertuples(index=False,name=None):
            lines.append('| '+' | '.join(('--' if pd.isna(v) else f'{v:.2f}') if i in (4,6,7) else str(int(v)) for i,v in enumerate(row))+' |')
        lines.append('')
    (OUT/'table.md').write_text('\n'.join(lines)+'\n')
    tex=[r'\begin{table*}[!ht]',r'\centering',
         r'\caption{Direction calls from adaptive protein-weighted CV+ intervals at each nominal level. $N_+$ and $N_-$ count intervals strictly above and strictly below zero. $N_{\ne0}$ is their sum; its percentage uses all test mutations (669 or 461). $P_{\ge1}$ counts distinct WT proteins with at least one such mutation; its percentage uses all test proteins (87 or 46). Proteins can have both types of mutation and are counted once in $P_{\ge1}$. Sign acc. is the percentage of zero-excluding calls whose assigned sign matches the observed target, with equal weight per called mutation. Zero targets count as incorrect; -- denotes no calls. This empirical accuracy is distinct from nominal coverage.}',
         r'\label{tab:zero_exclusion}',r'\small',r'\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.04}',
         r'\begin{tabular}{r rrrrr rrrrr}',r'\toprule',
         r'& \multicolumn{5}{c}{S669L: 669 mutations, 87 proteins} & \multicolumn{5}{c}{S461L: 461 mutations, 46 proteins}\\',
         r'\cmidrule(lr){2-6}\cmidrule(lr){7-11}',
         r'Nominal (\%) & $N_+$ & $N_-$ & $N_{\ne0}$ (\%) & $P_{\ge1}$ (\%) & Sign acc. (\%) & $N_+$ & $N_-$ & $N_{\ne0}$ (\%) & $P_{\ge1}$ (\%) & Sign acc. (\%)\\',r'\midrule']
    for pct in range(50,96):
        entries=[str(pct)]
        for name in ('S669L','S461L'):
            row=table[(table.dataset==name)&(table.nominal_percent==pct)].iloc[0]
            entries += [str(int(row.stabilizing_mutations)),str(int(row.destabilizing_mutations)),
                        f'{int(row.zero_excluded_mutations)} ({row.zero_excluded_percent:.2f})',
                        f'{int(row.proteins_with_any_zero_excluded)} ({row.proteins_with_any_zero_excluded_percent:.2f})',
                        '--' if pd.isna(row.sign_accuracy_percent) else f'{row.sign_accuracy_percent:.2f}']
        tex.append(' & '.join(entries)+r'\\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (OUT/'zero_exclusion.tex').write_text('\n'.join(tex)+'\n')
    print(table[table.nominal_percent.isin([50,80,90,95])][['dataset']+columns].to_string(index=False))

if __name__=='__main__':main()
