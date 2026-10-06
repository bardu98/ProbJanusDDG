"""Independent 3+1 inner selection, four-fold fresh refits, held-out OOF CV+."""
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

SOURCE=Path(__file__).resolve().parent
ROOT=SOURCE.parent
PARENT=ROOT/"pipeline"
HERE=Path(os.environ.get("NAIVE_RUN_DIR", ROOT/"runs/naive")).resolve()
sys.path[:0]=[str(SOURCE),str(PARENT),str(ROOT)]
# Limit CPU overhead per GPU worker.
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import pandas as pd
import torch
from model import NaiveGaussianFFNN, encode, predict, batch
from protocol import fold_plan, choose_epoch, protein_mse
from pjanus.losses import beta_nll


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path,obj):
    path=Path(path); tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n'); tmp.replace(path)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def save_torch(path,value):
    tmp=path.with_suffix('.tmp'); torch.save(value,tmp); tmp.replace(path)


def fit(a,b,centers,y,epochs,seed,tag,validation=None):
    """Epoch-boundary restart preserves model, optimizer and all RNG states."""
    seed_all(seed)
    model=NaiveGaussianFFNN().to('cuda'); opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    # Each mutation and its reverse contribute equally, as in Janus training.
    aw=a+b; bm=b+a; cc=np.concatenate([centers,centers]); target=torch.cat([y,-y]).to('cuda')
    path=HERE/'checkpoints'/f'{tag}_resume.pt'
    history=[]; start=1
    if path.exists():
        saved=torch.load(path,weights_only=False,map_location='cpu')
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        history=saved['history']; start=saved['epoch']+1
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng']); np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
    for ep in range(start,epochs+1):
        model.train(); began=time.time(); total=0.
        permutation=torch.randperm(len(target))
        for idx in permutation.split(6):
            mu,sigma=model(*batch(aw,bm,cc,idx,'cuda'),directional=True)
            loss=beta_nll(target[idx],mu,sigma,beta=0.)
            if not torch.isfinite(loss): raise ValueError(f'{tag}: non-finite loss at {ep}')
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total+=loss.item()*len(idx)
        row=dict(epoch=ep,train_loss=total/len(target),seconds=time.time()-began)
        if validation is not None:
            va,vb,vc,vy,proteins=validation
            mu,sigma=predict(model,va,vb,vc)
            row.update(protein_mse(vy,mu,proteins))
        history.append(row)
        if ep==1 or ep%5==0 or ep==epochs:
            pd.DataFrame(history).to_csv(HERE/'training'/f'{tag}_hist.csv',index=False)
            print(f'[{tag}] epoch {ep}/{epochs}, NLL={row["train_loss"]:.4f}, protein MSE={row.get("mse_protein",float("nan")):.4f}, {row["seconds"]:.2f}s',flush=True)
        if ep%10==0 or ep==epochs:
            save_torch(path,dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=ep,history=history,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate()))
    return model,pd.DataFrame(history)


def worker(k,max_epochs):
    torch.set_num_threads(1)
    d=pd.read_parquet(ROOT/'data/S2450.parquet').reset_index(drop=True)
    plan=fold_plan(d)[k]
    marker=HERE/f'fold{k}_complete.json'
    if marker.exists():
        result=json.loads(marker.read_text())
        for path,sha in result['artifacts'].items():
            assert digest(HERE/path)==sha, f'Artifact changed: {path}'
        return result['selection']
    a,b,centers=encode(d); y=torch.tensor(d.ddG.to_numpy(),dtype=torch.float32)
    tr=plan['inner_train_rows']; va=plan['early_stopping_rows']; ref=plan['refit_rows']; outer=plan['outer_rows']
    save_json(HERE/f'fold{k}_status.json',dict(state='inner_training',outer_fold=k))
    inner,curve=fit([a[i] for i in tr],[b[i] for i in tr],centers[tr],y[tr],max_epochs,10000+100*k,f'inner{k}',
                   ([a[i] for i in va],[b[i] for i in va],centers[va],d.iloc[va].ddG.to_numpy(),d.iloc[va].wt_seq.to_numpy()))
    curve.to_csv(HERE/'selection'/f'outer{k}_curve.csv',index=False)
    selection=dict(outer_fold=k,early_stopping_fold=(k+1)%5,**choose_epoch(curve,max_epochs))
    save_json(HERE/'selection'/f'outer{k}.json',selection)
    del inner
    save_json(HERE/f'fold{k}_status.json',dict(state='refitting',**selection))
    model,_=fit([a[i] for i in ref],[b[i] for i in ref],centers[ref],y[ref],selection['selected_epoch'],100*k,f'refit{k}')
    ckpt=HERE/'checkpoints'/f'refit_f{k}.pt'
    save_torch(ckpt,model.state_dict()); sha=digest(ckpt)
    artifacts=[ckpt,HERE/'selection'/f'outer{k}_curve.csv']
    def save_predictions(name,frame,aw,bm,centers,rows):
        mu,sigma=predict(model,aw,bm,centers)
        mr,sr=predict(model,bm,aw,centers)
        np.testing.assert_allclose(mu,-mr,atol=1e-7,rtol=0)
        np.testing.assert_allclose(sigma,sr,atol=1e-7,rtol=0)
        path=HERE/'predictions'/name
        np.savez_compressed(path,row_id=rows,y=frame.ddG.to_numpy(),mu=mu,sigma=sigma,
            fold_id=np.full(len(frame),k),selected_epoch=selection['selected_epoch'],checkpoint_sha256=sha)
        artifacts.append(path)
    save_predictions(f'oof_f{k}.npz',d.iloc[outer],[a[i] for i in outer],[b[i] for i in outer],centers[outer],np.array(outer))
    for name in ('S669L','S461L'):
        frame=pd.read_parquet(ROOT/'data'/f'{name}.parquet').reset_index(drop=True)
        aw,bm,cc=encode(frame)
        save_predictions(f'{name}_f{k}.npz',frame,aw,bm,cc,np.arange(len(frame)))
    save_json(marker,dict(selection=selection,artifacts={str(p.relative_to(HERE)):digest(p) for p in artifacts}))
    save_json(HERE/f'fold{k}_status.json',dict(state='complete',**selection))
    return selection


def preflight(max_epochs):
    torch.set_num_threads(1)
    for sub in ('training','checkpoints','selection','predictions','figures'): (HERE/sub).mkdir(exist_ok=True)
    d=pd.read_parquet(ROOT/'data/S2450.parquet').reset_index(drop=True)
    plans=fold_plan(d)
    parent=json.loads((ROOT/'reference/protocol.json').read_text())
    assert plans==json.loads((SOURCE/'reference/manifest.json').read_text())['folds'], 'Split plan differs from archived naive run'
    benchmark=pd.read_csv(ROOT/'reference/fold_plan.csv')
    for plan,row in zip(plans,benchmark.to_dict('records')):
        assert plan['outer_fold']==row['outer_fold'] and plan['early_stopping_fold']==row['early_stopping_fold']
        assert ','.join(map(str,plan['inner_train_folds']))==row['inner_train_folds']
        assert ','.join(map(str,plan['refit_folds']))==row['refit_folds']
    seed_all(123)
    a,b,centers=encode(d)
    assert all(len(w)==len(seq) for w,seq in zip(a,d.wt_seq))
    assert all(int((w!=m).sum())==1 for w,m in zip(a,b))
    model=NaiveGaussianFFNN()
    mu,sg=predict(model,a[:20],b[:20],centers[:20]); rev,sr=predict(model,b[:20],a[:20],centers[:20])
    np.testing.assert_array_equal(mu,-rev); np.testing.assert_array_equal(sg,sr)
    identity,_=predict(model,a[:20],a[:20],centers[:20]); np.testing.assert_array_equal(identity,np.zeros(20))
    sample=batch(a,b,centers,range(6),'cpu')
    loss=beta_nll(torch.tensor(d.ddG.to_numpy()[:6],dtype=torch.float32),*model(*sample,directional=True),beta=0.)
    loss.backward(); assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    synthetic=pd.DataFrame(dict(wt_seq=['ACD','ACDEFGHIK'],mut_seq=['CCD','ACDEFGHIA']))
    sa,sb,sc=encode(synthetic)
    assert sc.tolist()==[0,8]
    # Masked pooling must ignore padding and other batch members.
    together=predict(model,sa,sb,sc)
    alone=predict(model,sa[:1],sb[:1],sc[:1])
    np.testing.assert_allclose([t[0] for t in together],[t[0] for t in alone],atol=1e-7)
    # Check against an explicit per-residue forward without any padding.
    w,m,ll,cc=batch(sa,sb,sc,[0],'cpu')
    with torch.no_grad():
        ohw=torch.nn.functional.one_hot(w,21)[...,:20].float()
        ohm=torch.nn.functional.one_hot(m,21)[...,:20].float()
        rel=(torch.arange(3)[None,:]-cc[:,None]).float()/ll[:,None]
        hh=model.residue(torch.cat([ohw,ohw-ohm,rel[...,None],rel.abs()[...,None]],-1))
        expected=model.readout(torch.cat([hh.mean(1),hh.amax(1),hh[:,0]],-1))
        torch.testing.assert_close(model.raw(w,m,ll,cc),expected)
    for name in ('S669L','S461L'):
        external=pd.read_parquet(ROOT/'data'/f'{name}.parquet')
        ew,em,ec=encode(external)
        assert all(len(w)==len(seq) for w,seq in zip(ew,external.wt_seq))
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for full-sequence training')
    sources=[SOURCE/'model.py',SOURCE/'run.py',SOURCE/'compare.py',PARENT/'protocol.py',PARENT/'evaluate.py',ROOT/'pjanus/losses.py',ROOT/'conformal/protein_balanced.py']
    manifest=dict(protocol='naive_onehot_nested_3_1_4_refit_CVplus',max_epochs=max_epochs,
       input='complete WT/mutant sequences, one-hot 20 AA; relative signed/absolute distance (i-j)/L; masked padding',
       architecture=dict(residue=[42,64,64],pooling=['masked_mean','masked_max','mutation_site'],readout=[192,64,2]),parameters=sum(p.numel() for p in model.parameters()),
       training='directional Gaussian NLL beta=0, reverse augmentation; equal mutation weights',
       inference='antisymmetric mean, symmetric raw scale then softplus + 0.001',
       optimizer='Adam',lr=1e-4,batch_size=6,precision='float32',device='cuda',
       epoch_selection='raw minimum equal-protein validation MSE; earliest tie; fresh refit',
       inner_seeds=[10000+100*k for k in range(5)],refit_seeds=[100*k for k in range(5)],
       point_aggregation='median',calibration='same empirical equal-protein CV+ implementation as Janus',
       folds=plans,data_sha256={n:digest(ROOT/'data'/f'{n}.parquet') for n in ('S2450','S669L','S461L')},
       source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
       versions=dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__))
    for name,sha in manifest['data_sha256'].items(): assert sha==parent['data_sha256'][name]
    path=HERE/'manifest.json'
    if path.exists(): assert json.loads(path.read_text())==manifest, 'Manifest changed; do not mix runs'
    else: save_json(path,manifest)
    save_json(HERE/'preflight.json',dict(split_plan_identical=True,data_hashes_identical=True,
       encoding_checks=True,antisymmetry=True,scale_symmetry=True,finite_gradients=True,parameter_count=manifest['parameters']))
    print('Preflight passed:',manifest['parameters'],'parameters; same data and folds as Janus.',flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--max-epochs',type=int,default=300); parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    HERE.mkdir(parents=True,exist_ok=True)
    with (HERE/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        preflight(args.max_epochs)
        if args.preflight: return
        save_json(HERE/'status.json',dict(state='training',pid=os.getpid(),started=time.time()))
        try:
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures=[pool.submit(worker,k,args.max_epochs) for k in range(5)]
                rows=[f.result() for f in futures]
            pd.DataFrame(rows).sort_values('outer_fold').to_csv(HERE/'selected_epochs.csv',index=False)
            save_json(HERE/'status.json',dict(state='evaluating',pid=os.getpid()))
            from evaluate import generate
            generate(HERE)
            from compare import generate_comparison
            generate_comparison()
            save_json(HERE/'status.json',dict(state='complete',finished=time.time(),selected_epochs=[r['selected_epoch'] for r in rows]))
        except BaseException:
            save_json(HERE/'status.json',dict(state='failed',error=traceback.format_exc())); raise

if __name__=='__main__': main()
