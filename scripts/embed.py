"""Generate a frozen ESM2 ragged cache for benchmark or user-supplied full sequences."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL = 'facebook/esm2_t33_650M_UR50D'
REVISION = '08e4846e537177426273712802403f7ba8261b6c'


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='CSV or parquet with wt_seq and mut_seq')
    parser.add_argument('--output', type=Path, default=ROOT / 'embeddings')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; choose --device cpu')
    if args.input:
        d = pd.read_parquet(args.input) if args.input.suffix == '.parquet' else pd.read_csv(args.input)
    else:
        d = pd.concat([pd.read_parquet(ROOT / 'data' / (name + '.parquet'))
                       for name in ['S2450', 'S669L', 'S461L']], ignore_index=True)
    if not {'wt_seq', 'mut_seq'} <= set(d):
        parser.error('input must contain wt_seq and mut_seq')
    sequences = set(d.wt_seq) | set(d.mut_seq)
    if not sequences or any(not isinstance(s, str) or not s or any(a not in 'ACDEFGHIKLMNPQRSTVWY' for a in s) for s in sequences):
        parser.error('sequences must be nonempty strings of the 20 standard amino acids')
    sequences = sorted(sequences, key=lambda s: (len(s), s))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'embed.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        flat_path, done_path = out / 'long_flat.npy', out / 'done.npy'
        if flat_path.exists() and not done_path.exists():
            # Never overwrite an archived or externally restored cache.
            index = json.loads((out / 'long_index.json').read_text())
            offset = np.load(out / 'long_off.npy')
            flat = np.load(flat_path, mmap_mode='r')
            assert flat.shape[1] == 1280 and flat.dtype == np.float16
            for sequence in sequences:
                if sequence not in index:
                    raise ValueError('Existing cache misses sequences; use a NEW --output directory')
                i = index[sequence]
                x = flat[offset[i]:offset[i + 1]]
                assert x.shape == (len(sequence), 1280) and np.isfinite(x).all()
            print('Existing cache contains all requested sequences; left unchanged.')
            return
        lengths = np.array(list(map(len, sequences)), dtype=np.int64)
        offsets = np.r_[0, np.cumsum(lengths)]
        index = {s: i for i, s in enumerate(sequences)}
        manifest = {'model': MODEL, 'revision': REVISION, 'dimension': 1280,
                    'inference': 'float32, TF32 disabled', 'storage': 'float16',
                    'representation': 'final last_hidden_state, BOS/EOS removed',
                    'full_sequences': True, 'truncation': False, 'index': index}
        if (out / 'extraction.json').exists():
            assert json.loads((out / 'extraction.json').read_text()) == manifest
        else:
            write(out / 'extraction.json', manifest)
            write(out / 'long_index.json', index)
            np.save(out / 'long_off.npy', offsets)
            np.save(out / 'long_lens.npy', lengths)
        flat = np.lib.format.open_memmap(flat_path, mode='r+' if flat_path.exists() else 'w+',
                                        dtype=np.float16, shape=(int(offsets[-1]), 1280))
        if not done_path.exists():
            np.save(done_path, np.zeros(len(sequences), dtype=np.uint8))
        done = np.lib.format.open_memmap(done_path, mode='r+')
        assert flat.shape == (int(offsets[-1]), 1280) and len(done) == len(sequences)
        if done.all():
            assert np.isfinite(flat).all()
            print('Completed cache verified; no extraction needed.')
            return
        from transformers import AutoTokenizer, EsmModel
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION,
                                                  local_files_only=args.local_files_only)
        model = EsmModel.from_pretrained(MODEL, revision=REVISION, add_pooling_layer=False,
                                         local_files_only=args.local_files_only).to(args.device).eval()
        model.requires_grad_(False)
        started = time.time()
        with torch.inference_mode():
            for i, sequence in enumerate(sequences):
                if done[i]:
                    assert np.isfinite(flat[offsets[i]:offsets[i + 1]]).all()
                    continue
                tokens = tokenizer(sequence, return_tensors='pt', truncation=False)
                assert tokens['input_ids'].shape[1] == len(sequence) + 2
                result = model(**{k: v.to(args.device) for k, v in tokens.items()}).last_hidden_state
                result = result[0, 1:-1].float().cpu().numpy()
                assert result.shape == (len(sequence), 1280) and np.isfinite(result).all()
                result = result.astype(np.float16)
                assert np.isfinite(result).all()
                flat[offsets[i]:offsets[i + 1]] = result
                flat.flush()
                done[i] = 1
                done.flush()
                if i % 25 == 0 or done.all():
                    print('ESM2', int(done.sum()), '/', len(sequences),
                          'elapsed seconds', round(time.time() - started, 1), flush=True)
        write(out / 'extraction_complete.json', {'completed': int(done.sum()),
                                                 'elapsed_seconds': time.time() - started})


if __name__ == '__main__':
    main()
