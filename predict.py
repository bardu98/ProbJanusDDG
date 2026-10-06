"""Median predictions and protein-weighted CV+ intervals for WT/mutant pairs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'conformal'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path,
                        help='CSV or parquet with full wt_seq and mut_seq columns')
    parser.add_argument('--output', type=Path, default=Path('predictions.csv'))
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'pretrained')
    parser.add_argument('--embeddings', type=Path, default=ROOT / 'embeddings')
    parser.add_argument('--coverage', type=float, default=.9)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    args.model_dir = args.model_dir.resolve()
    if not .5 <= args.coverage < 1:
        parser.error('coverage must be in [0.5, 1)')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; choose --device cpu')
    os.environ['PJANUS_EMB'] = str(args.embeddings.resolve())
    os.environ['PJANUS_DATA'] = str(ROOT / 'data')
    from pjanus.data import RaggedEmbeddings
    from pjanus.model import build
    from pjanus.train import predict
    from protein_balanced import weighted_bounds

    torch.set_num_threads(4)
    frame = pd.read_parquet(args.input) if args.input.suffix == '.parquet' else pd.read_csv(args.input)
    if not {'wt_seq', 'mut_seq'} <= set(frame):
        parser.error('input must contain wt_seq and mut_seq columns')
    frame = frame.reset_index(drop=True)
    if len(frame) == 0:
        parser.error('input is empty')
    for row in frame.itertuples():
        if not isinstance(row.wt_seq, str) or not isinstance(row.mut_seq, str):
            parser.error('sequence columns must contain strings')
        if not 20 <= len(row.wt_seq) <= 3719 or len(row.wt_seq) != len(row.mut_seq):
            parser.error('WT and MUT must have equal lengths between 20 and 3719 residues')
        if sum(a != b for a, b in zip(row.wt_seq, row.mut_seq)) != 1:
            parser.error('each input row must contain exactly one amino acid substitution')
    targets = frame.copy()
    targets['ddG'] = 0.0  # PairSet requires this field; input labels are never used for inference.
    bank = RaggedEmbeddings()
    missing = (set(frame.wt_seq) | set(frame.mut_seq)) - set(bank.index)
    if missing:
        parser.error('Sequences missing from the embedding cache. Run scripts/embed.py '
                     '--input your_pairs.csv --output embeddings_custom, then use '
                     '--embeddings embeddings_custom')
    train = pd.read_parquet(ROOT / 'data/S2450.parquet').reset_index(drop=True)
    path = args.model_dir / 'oof_all.npz'
    if not path.exists():
        path = args.model_dir / 'predictions/oof_all.npz'
    with np.load(path) as z:
        oof = {key: z[key] for key in z.files}
    np.testing.assert_array_equal(oof['row_id'], np.arange(len(train)))
    np.testing.assert_array_equal(oof['y'], train.ddG)
    np.testing.assert_array_equal(oof['fold_id'], train.cvfold)
    epochs = pd.read_csv(args.model_dir / 'selected_epochs.csv').sort_values('outer_fold')
    np.testing.assert_array_equal(epochs.outer_fold, np.arange(5))
    manifest_path = args.model_dir / 'manifest.json'
    metadata = json.loads(manifest_path.read_text()) if args.model_dir == ROOT / 'pretrained' else None
    if metadata is not None:
        assert digest(path) == metadata['oof_sha256']
        assert digest(ROOT / 'data/S2450.parquet') == metadata['training_data_sha256']
    mus, sigmas = [], []
    for row in epochs.itertuples():
        k, e = int(row.outer_fold), int(row.selected_epoch)
        if metadata is not None:
            checkpoint = args.model_dir / metadata['folds'][k]['checkpoint']
            assert digest(checkpoint) == metadata['folds'][k]['sha256']
            assert e == metadata['folds'][k]['selected_epoch']
        else:
            candidates = list((args.model_dir / 'checkpoints').glob(f'refit_s*_outer{k}_e{e}.pt'))
            if len(candidates) != 1:
                raise ValueError('Expected exactly one checkpoint for outer fold ' + str(k))
            checkpoint = candidates[0]
            with np.load(args.model_dir / 'predictions' / f'oof_f{k}.npz') as z:
                assert digest(checkpoint) == str(z['checkpoint_sha256'])
                assert int(z['selected_epoch']) == e
                ids = np.flatnonzero(train.cvfold.to_numpy() == k)
                np.testing.assert_array_equal(z['mu'], oof['mu'][ids])
                np.testing.assert_array_equal(z['sigma'], oof['sigma'][ids])
        model = build(args.device, correct_conv_mask=False)
        model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
        mu, sigma = predict(model, targets, bank, device=args.device, bs=32, antisim=True)
        mus.append(mu); sigmas.append(sigma)
        del model
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    mu, sigma = np.asarray(mus), np.asarray(sigmas)
    assert np.isfinite([mu, sigma]).all() and (sigma > 0).all()
    weights = (1 / (train.wt_seq.nunique() * train.groupby('wt_seq').wt_seq.transform('size'))).to_numpy()
    result = frame.copy()
    result['predicted_ddG'] = np.median(mu, axis=0)
    result['nominal_coverage'] = args.coverage
    for name, adaptive in [('standard', False), ('adaptive', True)]:
        low, high = weighted_bounds(oof, mu, sigma, weights, np.array([args.coverage]), adaptive)
        result[name + '_lower'], result[name + '_upper'] = low[:, 0], high[:, 0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print('Saved', len(result), 'median predictions with protein-weighted CV+ intervals to', args.output)


if __name__ == '__main__':
    main()
