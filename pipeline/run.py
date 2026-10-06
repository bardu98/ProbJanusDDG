"""Nested epoch selection: three folds fit, one selects, four refit, one scores."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
os.environ['PJANUS_RESULTS'] = str(HERE / 'training')
os.environ['PJANUS_CKPT'] = str(HERE / 'checkpoints')
os.environ['PJANUS_DATA'] = str(ROOT / 'data')
os.environ['PJANUS_EMB'] = str(ROOT / 'embeddings')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def status(state, **kwargs):
    write_json(HERE / 'status.json', dict(state=state, pid=os.getpid(), updated=time.time(), **kwargs))


def verified(marker):
    if not marker.exists():
        return None
    record = json.loads(marker.read_text())
    for name, expected in record['artifacts'].items():
        if not (HERE / name).exists() or digest(HERE / name) != expected:
            raise RuntimeError(f'Completed-stage artifact changed or missing: {name}')
    return record


def complete(marker, paths, **kwargs):
    write_json(marker, dict(**kwargs, artifacts={str(p.relative_to(HERE)): digest(p) for p in paths}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--preflight', action='store_true', help='Validate data and write the fixed protocol; do not train')
    args = parser.parse_args()
    if args.max_epochs < 1 or args.seed < 0:
        parser.error('max-epochs must be positive and seed nonnegative')
    for sub in ('training', 'checkpoints', 'selection', 'predictions', 'figures'):
        (HERE / sub).mkdir(exist_ok=True)
    with open(HERE / 'run.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(args)
        except BaseException:
            status('failed', error=traceback.format_exc())
            raise


def execute(args):
    import numpy as np
    import pandas as pd
    import torch
    from pjanus.data import load, con_inversi, RaggedEmbeddings
    from pjanus.train import train, predict
    from pjanus.model import build
    from protocol import fold_plan, choose_epoch, protein_mse
    from inner_train import fit_inner

    torch.set_num_threads(4)
    if not args.preflight and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for training; install a CUDA-enabled PyTorch build')
    data = load('S2450').reset_index(drop=True)
    plan = fold_plan(data)
    sources = [HERE / n for n in ('run.py', 'protocol.py', 'inner_train.py', 'evaluate.py')]
    sources += [ROOT / 'pjanus' / n for n in ('train.py', 'model.py', 'data.py', 'losses.py', 'criteria.py')]
    sources += [ROOT / n for n in ('config.py', 'conformal/cvplus.py', 'conformal/protein_balanced.py')]
    manifest = dict(
        protocol='nested_3_train_1_mse_protein_4_refit_1_outer_v3', seed=args.seed,
        max_epochs=args.max_epochs, criterion='antisymmetric prediction, mean squared error within WT protein, equal mean over proteins',
        criterion_calibration='none', criterion_evaluation='all inner-validation mutations',
        tie_break='earliest epoch',
        patience=None, smoothing=None, early_fold_rule='(outer_fold + 1) % 5',
        inner_seeds=[args.seed + 10000 + 100 * k for k in range(5)],
        refit_seeds=[args.seed + 100 * k for k in range(5)],
        refit_from_scratch=True, epoch_transfer='same number of epochs; no optimizer-step rescaling',
        beta=0.0, batch_size=6, lr=1e-4, correct_conv_mask=False,
        point_aggregation='median of five refit predictions',
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        data_sha256={n: digest(ROOT / 'data' / f'{n}.parquet') for n in ('S2450', 'S669L', 'S461L')},
        embedding_sha256={p.name: digest(p) for p in sorted((ROOT / 'embeddings').glob('long_*'))},
        versions=dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__, pandas=pd.__version__),
        folds=plan,
    )
    dest = HERE / 'manifest.json'
    if dest.exists() and json.loads(dest.read_text()) != manifest:
        raise RuntimeError('Protocol, data, environment or sources changed. Refusing to mix runs.')
    write_json(dest, manifest)
    pd.DataFrame([dict(outer_fold=p['outer_fold'], early_stopping_fold=p['early_stopping_fold'],
        inner_train_folds=','.join(map(str, p['inner_train_folds'])),
        refit_folds=','.join(map(str, p['refit_folds'])),
        inner_mutations=len(p['inner_train_rows']), validation_mutations=len(p['early_stopping_rows']),
        refit_mutations=len(p['refit_rows']), outer_mutations=len(p['outer_rows'])) for p in plan]
    ).to_csv(HERE / 'fold_plan.csv', index=False)
    print(pd.read_csv(HERE / 'fold_plan.csv').to_string(index=False), flush=True)
    for p in plan:
        va = data.iloc[p['early_stopping_rows']].reset_index(drop=True)
        protein_mse(np.zeros(len(va)), np.zeros(len(va)), va.wt_seq.to_numpy())
    if args.preflight:
        print('Preflight passed: disjoint WT/mutant sequences; fixed protocol recorded.', flush=True)
        return

    status('loading_embeddings')
    rag = RaggedEmbeddings()
    selections = []
    # Only S2450 is loaded throughout selection and refitting. Each outer fold is unused.
    for p in plan:
        k, v = p['outer_fold'], p['early_stopping_fold']
        tag = f'inner_s{args.seed}_outer{k}_val{v}'
        selection_marker = HERE / 'selection' / f'outer{k}_complete.json'
        record = verified(selection_marker)
        if record is None:
            status('inner_training', outer_fold=k, early_stopping_fold=v, max_epochs=args.max_epochs,
                   history=f'training/{tag}_hist.csv')
            inner_val = data.iloc[p['early_stopping_rows']].reset_index(drop=True)
            cache = verified(HERE / 'selection' / f'outer{k}_inner_cache.json')
            if cache is None:
                inner_train = con_inversi(data.iloc[p['inner_train_rows']]).reset_index(drop=True)
                print(f'OUTER {k}: training on {p["inner_train_folds"]}, selecting on fold {v}', flush=True)
                fit_inner(inner_train, inner_val, args.max_epochs, manifest['inner_seeds'][k],
                          rag, tag, HERE)
            else:
                print(f'OUTER {k}: reusing verified inner predictions; selecting on fold {v}', flush=True)
            saved = [HERE / 'training' / f'{tag}_hist.csv']
            curve_rows = []
            for ep in range(1, args.max_epochs + 1):
                path = HERE / 'selection' / f'{tag}_e{ep}_oof.npy'
                y, mu, sigma = np.load(path).astype(float)
                np.testing.assert_array_equal(y, inner_val.ddG.to_numpy())
                if not np.isfinite([y, mu, sigma]).all() or not (sigma > 0).all():
                    raise ValueError(f'Invalid inner predictions: {path}')
                curve_rows.append(dict(epoch=ep, **protein_mse(y, mu, inner_val.wt_seq.to_numpy())))
                saved.append(path)
            curve = pd.DataFrame(curve_rows)
            selected = choose_epoch(curve, args.max_epochs)
            curve_path = HERE / 'selection' / f'outer{k}_curve.csv'
            pd.DataFrame(curve).to_csv(curve_path, index=False)
            saved.append(curve_path)
            complete(selection_marker, saved, outer_fold=k, early_stopping_fold=v, **selected)
            record = verified(selection_marker)
        e = record['selected_epoch']
        selections.append({key: value for key, value in record.items() if key != 'artifacts'})
        pd.DataFrame(selections).to_csv(HERE / 'selected_epochs.csv', index=False)
        print(f'OUTER {k}: protein MSE selected epoch {e}; refit from scratch on {p["refit_folds"]}', flush=True)
        refit_tag = f'refit_s{args.seed}_outer{k}'
        checkpoint = HERE / 'checkpoints' / f'{refit_tag}_e{e}.pt'
        refit_marker = HERE / 'checkpoints' / f'outer{k}_complete.json'
        refit_record = verified(refit_marker)
        if refit_record is None:
            status('refit_training', outer_fold=k, selected_epoch=e, history=f'training/{refit_tag}_hist.csv')
            refit_data = con_inversi(data.iloc[p['refit_rows']]).reset_index(drop=True)
            model = train(refit_data, None, e, manifest['refit_seeds'][k], rag, refit_tag,
                          bs=6, lr=1e-4, beta=0., snap_eps=(e,),
                          snap_prefix=str(HERE / 'checkpoints' / refit_tag), correct_conv_mask=False)
            del model
            torch.cuda.empty_cache()
            complete(refit_marker, [checkpoint, HERE / 'training' / f'{refit_tag}_hist.csv'],
                     outer_fold=k, selected_epoch=e, checkpoint=str(checkpoint.relative_to(HERE)))
        elif refit_record['selected_epoch'] != e:
            raise RuntimeError('Selection/refit epoch mismatch')

    # All five choices are frozen before any external labels or outer predictions are used.
    external = {name: load(name).reset_index(drop=True) for name in ('S669L', 'S461L')}
    for p, selected in zip(plan, selections):
        k, e = p['outer_fold'], selected['selected_epoch']
        marker = HERE / 'predictions' / f'outer{k}_complete.json'
        if verified(marker) is not None:
            continue
        checkpoint = HERE / 'checkpoints' / f'refit_s{args.seed}_outer{k}_e{e}.pt'
        checkpoint_sha = digest(checkpoint)
        status('predicting', outer_fold=k, selected_epoch=e)
        model = build(correct_conv_mask=False)
        model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True))
        saved = []
        datasets = [('oof', data.iloc[p['outer_rows']].reset_index(drop=True), np.array(p['outer_rows']))]
        datasets += [(name, frame, np.arange(len(frame))) for name, frame in external.items()]
        for name, frame, indices in datasets:
            mu, sigma = predict(model, frame, rag, bs=32)
            if not np.isfinite([mu, sigma]).all() or not (sigma > 0).all():
                raise ValueError(f'Invalid {name} predictions for fold {k}')
            path = HERE / 'predictions' / f'{name}_f{k}.npz'
            np.savez_compressed(path, row_id=indices, y=frame.ddG.to_numpy(), mu=mu, sigma=sigma,
                                fold_id=np.full(len(frame), k), selected_epoch=e, checkpoint_sha256=checkpoint_sha)
            saved.append(path)
        del model
        torch.cuda.empty_cache()
        complete(marker, saved, outer_fold=k, selected_epoch=e, checkpoint_sha256=checkpoint_sha)
    status('evaluating')
    from evaluate import generate
    generate(HERE)
    status('complete', selected_epochs=[s['selected_epoch'] for s in selections], completed=time.time())
    print('Completed: selected_epochs.csv, metrics.csv, report.md, figures/ and predictions/.', flush=True)


if __name__ == '__main__':
    main()
