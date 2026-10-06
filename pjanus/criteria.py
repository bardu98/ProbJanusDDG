"""
I criteri di validazione registrati a ogni epoca, e il quantile conforme che ne sta alla base.

QUALE CURVA SI SOMMA PER SCEGLIERE L'EPOCA. Il modello e' addestrato in beta-NLL, ma la
verosimiglianza non e' un criterio di arresto utilizzabile perche' non e' invariante di scala: il
suo minimo cade all'epoca 1-2, dove sigma non ha ancora imparato nulla. Si registrano quindi sei
curve e si sceglie a valle. Le due verosimiglianze non profilate sono riportate solo per
documentare il loro fallimento; il criterio dichiarato e' `val_is90`, che valuta direttamente
l'oggetto consegnato -- l'intervallo conforme -- al livello che poi si riporta.
"""
import numpy as np

COLS = ("val_mse", "val_nll", "val_bnll", "val_nllprof", "val_is90", "val_is80")


def conformal_quantile(scores, alpha):
    """k-esima statistica d'ordine con k = ceil((n+1)(1-alpha)).

    Il +1 non e' cosmetico: e' cio' che rende la copertura >= 1-alpha in campione finito e non
    soltanto asintoticamente. Se k > n l'insieme di predizione e' tutto R.
    """
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])


def interval_score(y, mu, sg, alpha, rng):
    """Interval score di Gneiting-Raftery, calibrato in croce sulle due meta' dell'insieme.

        IS = (u - l) + (2/alpha) * (l - y)+ + (2/alpha) * (y - u)+

    larghezza piu' penalita' per mancato contenimento. In ciascuna meta' sigma e' normalizzata
    sulla propria mediana, cosi' il punteggio e' insensibile alla scala assoluta di sigma.
    """
    n = len(y)
    if n < 20:
        return float("nan")
    idx = rng.permutation(n)
    tot, m = 0.0, 0
    for a, b in ((idx[: n // 2], idx[n // 2:]), (idx[n // 2:], idx[: n // 2])):
        sa = sg[a] / np.median(sg[a])
        sb = sg[b] / np.median(sg[b])
        q = conformal_quantile(np.abs(y[a] - mu[a]) / sa, alpha)
        if not np.isfinite(q):
            continue
        h = q * sb
        lo, hi = mu[b] - h, mu[b] + h
        s = 2 * h + (2 / alpha) * (np.maximum(lo - y[b], 0) + np.maximum(y[b] - hi, 0))
        tot += float(s.mean())
        m += 1
    return tot / m if m else float("nan")


def criteria(y, mu, sg, beta=0.5, seed=0):
    """Le sei curve candidate, calcolate sullo stesso insieme di validazione."""
    rng = np.random.default_rng(seed)
    e = y - mu
    z = e / sg
    lsg = np.log(sg)
    return {
        # non guarda sigma, per costruzione
        "val_mse": float(np.mean(e ** 2)),
        # verosimiglianza nuda: non invariante di scala, argmin all'epoca 1
        "val_nll": float(np.mean(lsg + 0.5 * z ** 2)),
        # la loss di addestramento: stessa patologia attenuata
        "val_bnll": float(np.mean((lsg + 0.5 * z ** 2) * sg ** (2 * beta))),
        # verosimiglianza profilata: esattamente invariante di scala
        "val_nllprof": float(np.mean(lsg) + 0.5 * np.log(np.mean(z ** 2)) + 0.5),
        # valutano l'intervallo conforme, cioe' l'oggetto consegnato
        "val_is90": interval_score(y, mu, sg, 0.10, rng),
        "val_is80": interval_score(y, mu, sg, 0.20, np.random.default_rng(seed)),
    }
