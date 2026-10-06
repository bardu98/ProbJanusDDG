"""Verify archived naive checkpoints, selection and the actual training path."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import numpy as np
import pandas as pd
import torch

SOURCE=Path(__file__).resolve().parent
ROOT=SOURCE.parent
sys.path[:0]=[str(SOURCE),str(ROOT/'pipeline'),str(ROOT)]
from model import NaiveGaussianFFNN,encode,predict
from protocol import fold_plan,choose_epoch
from evaluate import generate

def main():
    torch.set_num_threads(1)
    ref=SOURCE/'reference'
    d=pd.read_parquet(ROOT/'data/S2450.parquet').reset_index(drop=True)
    plans=fold_plan(d)
    manifest=json.loads((ref/'manifest.json').read_text())
    assert plans==manifest['folds']
    for name,sha in manifest['data_sha256'].items():
        assert hashlib.sha256((ROOT/'data'/f'{name}.parquet').read_bytes()).hexdigest()==sha
    for rel,sha in [( 'model.py',manifest['source_sha256']['cv+_21sett_2026/naive_regressor/model.py'])]:
        assert hashlib.sha256((SOURCE/rel).read_bytes()).hexdigest()==sha
    selected=pd.read_csv(ref/'selected_epochs.csv').sort_values('outer_fold')
    datasets={'S2450':d,**{n:pd.read_parquet(ROOT/'data'/f'{n}.parquet').reset_index(drop=True) for n in ['S669L','S461L']}}
    encoded={n:encode(frame) for n,frame in datasets.items()}
    checks=[]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for k,plan in enumerate(plans):
        curve=pd.read_csv(ref/'selection'/f'outer{k}_curve.csv')
        assert choose_epoch(curve,300)['selected_epoch']==int(selected.iloc[k].selected_epoch)
        ckpt=ref/'checkpoints'/f'refit_f{k}.pt'
        sha=hashlib.sha256(ckpt.read_bytes()).hexdigest()
        model=NaiveGaussianFFNN().to(device)
        model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True))
        for name,frame in datasets.items():
            a,b,c=encoded[name]
            if name=='S2450':
                rows=plan['outer_rows']; a=[a[i] for i in rows];b=[b[i] for i in rows];c=c[rows]
                path=ref/'predictions'/f'oof_f{k}.npz'
            else:path=ref/'predictions'/f'{name}_f{k}.npz'
            z=dict(np.load(path))
            assert str(z['checkpoint_sha256'])==sha
            mu,sigma=predict(model,a,b,c)
            # Exact reproduction on the original CUDA FP32 environment.
            np.testing.assert_allclose(mu,z['mu'],rtol=1e-5,atol=1e-6)
            np.testing.assert_allclose(sigma,z['sigma'],rtol=1e-5,atol=1e-6)
            checks.append(dict(fold=k,dataset=name,max_mu_error=float(abs(mu-z['mu']).max()),max_sigma_error=float(abs(sigma-z['sigma']).max()),bitwise_equal=bool(np.array_equal(mu,z['mu']) and np.array_equal(sigma,z['sigma']))))
            print(checks[-1],flush=True)
    # Exercise fit(), including direct Gaussian loss, inverse augmentation,
    # optimizer, validation and restart state; no external test-based tuning.
    if device=='cuda':
        import run
        original=run.HERE
        with tempfile.TemporaryDirectory() as td:
            run.HERE=Path(td)
            for sub in ['checkpoints','training']:(run.HERE/sub).mkdir()
            a,b,c=encoded['S2450']; ids=np.argsort([len(x) for x in a])[:12]
            a=[a[i] for i in ids];b=[b[i] for i in ids];c=c[ids]
            y=torch.tensor(d.iloc[ids].ddG.to_numpy(),dtype=torch.float32)
            validation=(a,b,c,y.numpy(),d.iloc[ids].wt_seq.to_numpy())
            model,hist=run.fit(a,b,c,y,1,123,'smoke',validation)
            assert len(hist)==1 and np.isfinite(hist.mse_protein).all()
            resumed,hist2=run.fit(a,b,c,y,1,123,'smoke',validation)
            for key,value in model.state_dict().items():torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
        run.HERE=original
    generate(ref)
    report=dict(device=device,parameters=sum(p.numel() for p in model.parameters()),selected_epochs=selected.selected_epoch.tolist(),prediction_checks=checks,training_smoke_test=device=='cuda',resume_check=device=='cuda',evaluation_completed=True)
    (ref/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Naive pipeline verified.',flush=True)

if __name__=='__main__':main()
