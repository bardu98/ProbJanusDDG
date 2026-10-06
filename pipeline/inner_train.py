"""Same Adam/Gaussian NLL training; select with equal-protein validation MSE."""
import time
import numpy as np
import pandas as pd
import torch
from pjanus.data import PairSet, collate
from pjanus.losses import beta_nll
from pjanus.model import build
from pjanus.train import set_seed, predict
from protocol import protein_mse


def fit_inner(tr, va, epochs, seed, rag, tag, out):
    set_seed(seed)
    dl = torch.utils.data.DataLoader(PairSet(tr, rag), batch_size=6, shuffle=True,
        num_workers=6, collate_fn=collate, persistent_workers=True)
    model = build('cuda', correct_conv_mask=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    history = []
    print(f'[{tag}] fresh initialization; equal-protein MSE validation', flush=True)
    for ep in range(1, epochs + 1):
        model.train()
        started, total, batches = time.time(), 0., 0
        for xw, xm, lengths, y in dl:
            xw, xm = xw.to('cuda', non_blocking=True).float(), xm.to('cuda', non_blocking=True).float()
            lengths, y = lengths.to('cuda', non_blocking=True), y.to('cuda', non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                mu, sigma = model(xw, xm, lengths, train=True)
            loss = beta_nll(y, mu.float(), sigma.float(), 0.)
            if not torch.isfinite(loss):
                raise ValueError(f'Non-finite training loss at epoch {ep}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
            batches += 1
        mu, sigma = predict(model, va, rag, bs=32, antisim=True)
        scores = protein_mse(va.ddG.to_numpy(), mu, va.wt_seq.to_numpy())
        path = out / 'selection' / f'{tag}_e{ep}_oof.npy'
        with open(path.with_suffix('.tmp'), 'wb') as f:
            np.save(f, np.stack([va.ddG.to_numpy(), mu, sigma]))
        path.with_suffix('.tmp').replace(path)
        row = dict(epoch=ep, train_loss=total / batches, seconds=time.time() - started, **scores)
        history.append(row)
        hist = out / 'training' / f'{tag}_hist.csv'
        pd.DataFrame(history).to_csv(hist.with_suffix('.tmp'), index=False)
        hist.with_suffix('.tmp').replace(hist)
        if ep == 1 or ep % 5 == 0:
            print(f'[{tag}] ep {ep:3d} loss {row["train_loss"]:+.4f} '
                  f'protein MSE {row["mse_protein"]:.4f} '
                  f'mutation MSE {row["mse_mutation"]:.4f} | {row["seconds"]:.1f}s', flush=True)
    del model, opt, dl
    torch.cuda.empty_cache()
    return pd.DataFrame(history)
