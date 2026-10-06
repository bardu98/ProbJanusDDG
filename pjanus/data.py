"""
Dati: i tre insiemi tabellari e lo store di embedding a lunghezza variabile.

Gli embedding ESM-2 sono CONGELATI: nessun gradiente li attraversa. Sono precalcolati una volta
da scripts/embed.py e letti in sola lettura tramite memmap.
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from config import DATA, DIM, EMB


class RaggedEmbeddings:
    """Embedding a lunghezza variabile in formato flat + offset: nessun padding su disco."""

    def __init__(self, tag="long"):
        for suf in ("_flat.npy", "_off.npy", "_index.json"):
            p = os.path.join(EMB, tag + suf)
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"manca {p}. Genera lo store con: python scripts/embed.py")
        self.flat = np.load(os.path.join(EMB, f"{tag}_flat.npy"), mmap_mode="r")
        self.off = np.load(os.path.join(EMB, f"{tag}_off.npy"))
        self.index = json.load(open(os.path.join(EMB, f"{tag}_index.json")))

    def get(self, i):
        return np.asarray(self.flat[self.off[i]:self.off[i + 1]], dtype=np.float32)


def load(name):
    """S2450 (addestramento, con la colonna cvfold del benchmark) oppure S669L / S461L."""
    return pd.read_parquet(os.path.join(DATA, f"{name}.parquet"))


def con_inversi(df):
    """Raddoppia con i versi invertiti: scambia wild-type e mutante, nega l'etichetta.

    E' il dataset di addestramento del JanusDDG originale (`s2450_fold_i.pkl` PIU'
    `s2450_fold_i_inv.pkl`). Serve solo perche' il modello viene addestrato in forma
    DIREZIONALE: vedi il docstring di pjanus/model.py. Nessun embedding nuovo, entrambe
    le sequenze sono gia' nello store.
    """
    inv = df.copy()
    inv["wt_seq"], inv["mut_seq"] = df.mut_seq.to_numpy(), df.wt_seq.to_numpy()
    inv["ddG"] = -df.ddG.to_numpy()
    inv["verso"] = "inv"
    return pd.concat([df.assign(verso="dir"), inv], ignore_index=True)


class PairSet(torch.utils.data.Dataset):
    """Una riga = una mutazione: (embedding wild-type, embedding mutante, ddG)."""

    def __init__(self, df, rag):
        self.a = [rag.index[s] for s in df.wt_seq]
        self.b = [rag.index[s] for s in df.mut_seq]
        self.y = df.ddG.to_numpy(np.float32)
        self.rag = rag

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.rag.get(self.a[i]), self.rag.get(self.b[i]), self.y[i]


def collate(batch):
    """Padding a destra fino alla lunghezza massima del batch; `length` porta la maschera."""
    L = max(x[0].shape[0] for x in batch)
    B = len(batch)
    xw = np.zeros((B, L, DIM), np.float16)
    xm = np.zeros((B, L, DIM), np.float16)
    ln = np.zeros(B, np.int64)
    y = np.zeros(B, np.float32)
    for k, (a, b, yy) in enumerate(batch):
        n = a.shape[0]
        xw[k, :n] = a
        xm[k, :n] = b
        ln[k] = n
        y[k] = yy
    return (torch.from_numpy(xw), torch.from_numpy(xm),
            torch.from_numpy(ln), torch.from_numpy(y))
