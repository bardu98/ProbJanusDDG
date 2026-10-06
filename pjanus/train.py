"""Shared training and prediction functions for nested CV+; no legacy CV CLI."""
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from config import CKPT, DIM, NFOLD, RESULTS
from pjanus.criteria import criteria
from pjanus.data import PairSet, RaggedEmbeddings, collate, con_inversi, load
from pjanus.losses import beta_nll
from pjanus.model import build

HIST_COLS = ["ep", "train_loss", "val_mse", "val_r", "rho_sig", "sig_med",
             "val_nll", "sd_z", "val_bnll", "val_nllprof", "val_is90", "val_is80", "val_r_antisim"]


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, df, rag, device="cuda", bs=32, antisim=True):
    """Predizione in batch ordinati per lunghezza decrescente: meno padding sprecato.

    antisim=True  -> mu = (m_AB - m_BA)/2, sigma = softplus((s_AB + s_BA)/2): CIO' CHE SI
                     CONSEGNA. Va usato per le out-of-fold salvate (il conforme calibra su
                     quello che il modello consegna davvero) e per il test.
    antisim=False -> il solo passaggio diretto: la stessa quantita' su cui si ottimizza.
                     Va usato per le curve di validazione da cui esce E*, per decisione di
                     Guido (26/08/2026): E* si sceglie sulla quantita' che si ottimizza,
                     come in un early stopping classico.
    """
    model.eval()
    ds = PairSet(df, rag)
    order = np.argsort(-np.array([len(s) for s in df.wt_seq]))
    mu = np.zeros(len(ds), np.float32)
    sg = np.zeros(len(ds), np.float32)
    for i in range(0, len(order), bs):
        sel = order[i:i + bs]
        xw, xm, ln, _ = collate([ds[j] for j in sel])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m, s = model(xw.to(device).float(), xm.to(device).float(), ln.to(device),
                         train=not antisim)
        mu[sel] = m.float().cpu().numpy()
        sg[sel] = s.float().cpu().numpy()
    return mu, sg


def train(tr, va, epochs, seed, rag, tag, device="cuda", bs=6, lr=1e-4, beta=0.5,
          snap_eps=(), snap_prefix=None, init=None, oof_every=0,
          correct_conv_mask=False):
    """Una corsa. Adam a lr costante, beta-NLL dalla prima epoca.

    `init` carica il checkpoint dello stadio A nel percorso di mu. Il caricamento e' `strict=False`
    e si ASSERISCE che l'unica chiave mancante sia `Linear_sig`: se mancasse altro, il checkpoint
    non corrisponde a questa architettura e si deve fallire subito, non addestrare a meta'.
    """
    set_seed(seed)
    dl = torch.utils.data.DataLoader(
        PairSet(tr, rag), batch_size=bs, shuffle=True, num_workers=6,
        collate_fn=collate, persistent_workers=True)
    model = build(device, correct_conv_mask=correct_conv_mask)
    if init:
        r = model.load_state_dict(torch.load(init, map_location=device), strict=False)
        assert not r.unexpected_keys, f"chiavi impreviste nel checkpoint: {r.unexpected_keys}"
        assert all("Linear_sig" in k for k in r.missing_keys), \
            f"mancano chiavi diverse da Linear_sig: {r.missing_keys}"
        print(f"  [init] {os.path.basename(init)} nel percorso di mu; Linear_sig nuovo",
              flush=True)
    else:
        print("  [init] da zero: nessun checkpoint caricato", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for xw, xm, ln, y in dl:
            xw = xw.to(device, non_blocking=True).float()
            xm = xm.to(device, non_blocking=True).float()
            ln = ln.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mu, sg = model(xw, xm, ln, train=True)
            loss = beta_nll(y, mu.float(), sg.float(), beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1

        row = [ep, tot / nb]
        if va is not None:
            yv = va.ddG.to_numpy()
            # DUE predizioni sullo stesso insieme di validazione, con usi separati:
            #   md, sd -> direzionale: alimenta i criteri, quindi E*
            #   m,  s  -> antisimmetrico: e' cio' che si consegna, e va nelle OOF salvate
            md, sd = predict(model, va, rag, device, antisim=False)
            m, s = predict(model, va, rag, device, antisim=True)
            c = criteria(yv, md, sd, beta=beta)
            row += [c["val_mse"], float(pearsonr(yv, md)[0]),
                    float(spearmanr(sd, np.abs(yv - md))[0]), float(np.median(sd)),
                    c["val_nll"], float(((yv - md) / sd).std()),
                    c["val_bnll"], c["val_nllprof"], c["val_is90"], c["val_is80"]]
            # r della versione consegnata, per non perderla di vista: e' quella che conta
            row.append(float(pearsonr(yv, m)[0]))
        hist.append(row)
        cols = HIST_COLS if va is not None else HIST_COLS[:2]
        pd.DataFrame(hist, columns=cols).to_csv(f"{RESULTS}/{tag}_hist.csv", index=False)

        if oof_every and va is not None and snap_prefix and ep % oof_every == 0:
            # Le out-of-fold a OGNI epoca (o ogni oof_every). Costo di calcolo nullo -- le
            # predizioni di validazione sono gia' state fatte -- e circa 15 KB per fold per
            # epoca. Serve perche' la calibrazione conforme richiede le out-of-fold ALLA STESSA
            # epoca del modello finale, ma E* si conosce solo DOPO che la CV e' finita: salvarle
            # tutte evita di dover rifare la cross-validation.
            np.save(f"{snap_prefix}_e{ep}_oof.npy", np.stack([yv, m, s]))
        if ep in snap_eps and snap_prefix:
            # le tre epoche candidate sono annidate e il training e' deterministico dato il seme:
            # il checkpoint all'epoca 30 e' bit per bit quello di una corsa fermata a 30.
            torch.save(model.state_dict(), f"{snap_prefix}_e{ep}.pt")
            if va is not None and not oof_every:
                np.save(f"{snap_prefix}_e{ep}_oof.npy", np.stack([yv, m, s]))

        if ep % 5 == 0 or ep == 1:
            msg = f"  [{tag}] ep {ep:3d} loss {tot/nb:+.4f}"
            if va is not None:
                msg += (f" | val mse {row[2]:.4f} r_dir={row[3]:+.3f}"
                        f" r_ANTI={row[12]:+.3f}"
                        f" | rho(sig,|e|)={row[4]:+.3f} sd(z)={row[7]:.1f}"
                        f" | IS90 {row[10]:.4f}")
            print(msg + f" | {time.time()-t0:.0f}s", flush=True)
    return model

