"""Recompute all OOF/S669L/S461L predictions from the five published checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ['PJANUS_DATA'] = str(ROOT / 'data')
os.environ['PJANUS_EMB'] = str(ROOT / 'embeddings')
from pjanus.data import RaggedEmbeddings, load
from pjanus.model import build
from pjanus.train import predict


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproducibility_audit/checkpoint_predictions')
    parser.add_argument('--atol', type=float, default=1e-6)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('CUDA is required to reproduce the archived bfloat16 inference')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'pretrained/manifest.json').read_text())
    assert digest(ROOT / 'pjanus/model.py') == manifest['model_source_sha256']
    assert digest(ROOT / 'data/S2450.parquet') == manifest['training_data_sha256']
    bank = RaggedEmbeddings()
    train = load('S2450').reset_index(drop=True)
    external = {name: load(name).reset_index(drop=True) for name in ['S669L', 'S461L']}
    rows = []
    started = time.time()
    for info in manifest['folds']:
        k, epoch = info['outer_fold'], info['selected_epoch']
        checkpoint = ROOT / 'pretrained' / info['checkpoint']
        assert digest(checkpoint) == info['sha256']
        model = build('cuda', correct_conv_mask=False)
        model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True))
        datasets = [('oof', train.loc[train.cvfold == k].reset_index(drop=True)), *external.items()]
        for name, frame in datasets:
            mu, sigma = predict(model, frame, bank, bs=32, antisim=True)
            path = ROOT / 'reference/predictions' / f'{name}_f{k}.npz'
            with np.load(path) as z:
                saved = {key: z[key] for key in z.files}
            assert int(saved['selected_epoch']) == epoch
            assert str(saved['checkpoint_sha256']) == info['sha256']
            np.testing.assert_array_equal(saved['y'], frame.ddG)
            mu_diff = float(np.abs(mu - saved['mu']).max())
            sigma_diff = float(np.abs(sigma - saved['sigma']).max())
            row = {'fold': k, 'dataset': name, 'rows': len(frame),
                   'max_absolute_mean_difference': mu_diff,
                   'max_absolute_scale_difference': sigma_diff,
                   'mean_bitwise_identical': bool(np.array_equal(mu, saved['mu'])),
                   'scale_bitwise_identical': bool(np.array_equal(sigma, saved['sigma'])),
                   'passed': bool(mu_diff <= args.atol and sigma_diff <= args.atol)}
            rows.append(row)
            saved['mu'], saved['sigma'] = mu, sigma
            np.savez_compressed(args.output / path.name, **saved)
            print(json.dumps(row), flush=True)
        del model
        torch.cuda.empty_cache()
    passed = all(row['passed'] for row in rows)
    result = {'passed': passed, 'atol': args.atol, 'inference_batch_size': 32,
              'precision': 'CUDA bfloat16 autocast as in the archived pipeline',
              'training_performed': False, 'comparisons': rows,
              'elapsed_seconds': time.time() - started, 'torch_version': str(torch.__version__),
              'gpu': torch.cuda.get_device_name()}
    (args.output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    if not passed:
        raise RuntimeError('Fresh checkpoint predictions differ from the archived predictions; see verification.json')
    print('All 15 checkpoint/dataset comparisons passed.', flush=True)


if __name__ == '__main__':
    main()
