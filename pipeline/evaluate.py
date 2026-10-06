"""Evaluate the five completed refits; no choices are based on external outcomes."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'conformal'))
from cvplus import intervals, metrics, uncertainty_rank
from protein_balanced import weighted_bounds


def generate(out, train_data=None, test_data=None):
    out = Path(out)
    pred = out / 'predictions'
    if train_data is None:
        train_data = pd.read_parquet(ROOT / 'data/S2450.parquet')
    if test_data is None:
        test_data = {n: pd.read_parquet(ROOT / 'data' / f'{n}.parquet') for n in ('S669L', 'S461L')}
    d = train_data.reset_index(drop=True)
    fold = d.cvfold.to_numpy(dtype=int)
    mu_oof, sigma_oof = np.full(len(d), np.nan), np.full(len(d), np.nan)
    seen = np.zeros(len(d), dtype=int)
    hashes = []
    epochs = pd.read_csv(out / 'selected_epochs.csv').sort_values('outer_fold')
    for k in range(5):
        with np.load(pred / f'oof_f{k}.npz') as z:
            idx = np.flatnonzero(fold == k)
            np.testing.assert_array_equal(z['row_id'], idx)
            np.testing.assert_array_equal(z['y'], d.iloc[idx].ddG)
            np.testing.assert_array_equal(z['fold_id'], np.full(len(idx), k))
            assert int(z['selected_epoch']) == int(epochs.iloc[k].selected_epoch)
            mu_oof[idx], sigma_oof[idx] = z['mu'], z['sigma']
            hashes.append(str(z['checkpoint_sha256']))
            seen[idx] += 1
    assert np.all(seen == 1) and np.isfinite([mu_oof, sigma_oof]).all() and (sigma_oof > 0).all()
    oof = dict(row_id=np.arange(len(d)), y=d.ddG.to_numpy(), mu=mu_oof, sigma=sigma_oof, fold_id=fold)
    np.savez_compressed(pred / 'oof_all.npz', **oof)
    weights = (1 / (d.wt_seq.nunique() * d.groupby('wt_seq').wt_seq.transform('size'))).to_numpy()
    levels = np.array([.8, .9, .95])
    records, point_records = [], []
    for name, frame in test_data.items():
        frame = frame.reset_index(drop=True)
        mus, sigmas = [], []
        for k in range(5):
            with np.load(pred / f'{name}_f{k}.npz') as z:
                assert str(z['checkpoint_sha256']) == hashes[k]
                assert int(z['selected_epoch']) == int(epochs.iloc[k].selected_epoch)
                np.testing.assert_array_equal(z['row_id'], np.arange(len(frame)))
                np.testing.assert_array_equal(z['y'], frame.ddG)
                mus.append(z['mu']); sigmas.append(z['sigma'])
        mu, sigma = np.array(mus), np.array(sigmas)
        y = frame.ddG.to_numpy()
        median, mean = np.median(mu, axis=0), mu.mean(axis=0)
        rank = uncertainty_rank(sigma_oof, fold, sigma)
        payload = dict(y=y, mu=median, mu_ensemble=median, mu_ensemble_mean_reference=mean,
                       mu_folds=mu, sigma_folds=sigma, levels=levels,
                       uncertainty_rank=rank, uncertainty_group=np.digitize(rank, [.2, .4, .6, .8]),
                       selected_epochs=epochs.selected_epoch.to_numpy())
        for aggregation, estimate in [('median', median), ('mean', mean)]:
            point_records.append(dict(dataset=name, aggregation=aggregation,
                pearson=float(pearsonr(y, estimate).statistic), mae=float(np.abs(y-estimate).mean()),
                rmse=float(np.sqrt(np.mean((y-estimate)**2)))))
        for adaptive, method in [(False, 'standard_cvplus'), (True, 'adaptive_cvplus')]:
            wl, wu = weighted_bounds(oof, mu, sigma, weights, levels, adaptive)
            for j, level in enumerate(levels):
                alpha = round(1-level, 2)
                cl, cu = intervals(oof['y'], mu_oof, sigma_oof, fold, mu, sigma, alpha, adaptive)
                for weighting, lo, hi in [('mutation_corrected', cl, cu), ('protein_empirical', wl[:, j], wu[:, j])]:
                    tag = f'{method}_{weighting}_a{alpha}'
                    payload[tag+'_lower'], payload[tag+'_upper'] = lo, hi
                    covered = (y >= lo) & (y <= hi)
                    protein_summary = pd.DataFrame(dict(protein=frame.wt_seq, covered=covered, width=hi-lo)).groupby('protein').mean()
                    records.append(dict(dataset=name, method=method, weighting=weighting, nominal=level,
                        **metrics(y, lo, hi, alpha), coverage_equal_protein=float(protein_summary.covered.mean()),
                        width_equal_protein=float(protein_summary.width.mean()),
                        n_mutations=len(frame), n_proteins=frame.wt_seq.nunique()))
        np.savez_compressed(pred / f'{name}_cvplus.npz', **payload)
    results, points = pd.DataFrame(records), pd.DataFrame(point_records)
    results.to_csv(out / 'metrics.csv', index=False)
    points.to_csv(out / 'point_metrics.csv', index=False)
    (out / 'figures').mkdir(exist_ok=True)
    fig, axes = plt.subplots(5, 1, figsize=(9, 12), layout='constrained')
    for k, ax in enumerate(axes):
        curve = pd.read_csv(out / 'selection' / f'outer{k}_curve.csv')
        ax.plot(curve.epoch, curve.mse_protein)
        ax.axvline(int(epochs.iloc[k].selected_epoch), color='C1', linestyle='--')
        ax.set(title=f'Outer fold {k}: validation fold {(k+1)%5}', xlabel='Epoch', ylabel='Equal-protein MSE')
        ax.grid(alpha=.2)
    for ext in ('png', 'pdf'):
        fig.savefig(out / 'figures' / f'epoch_selection.{ext}', dpi=180)
    plt.close(fig)
    for dataset, group in results.groupby('dataset'):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        for (method, weighting), part in group.groupby(['method', 'weighting']):
            label = f'{method} / {weighting}'
            axes[0].plot(part.nominal, part.coverage, 'o-', label=label)
            axes[1].plot(part.nominal, part.mean_width, 'o-', label=label)
        axes[0].plot(levels, levels, '--', color='gray')
        axes[0].set(xlabel='Nominal coverage', ylabel='Observed mutation-weighted coverage')
        axes[1].set(xlabel='Nominal coverage', ylabel='Mean width')
        for ax in axes:
            ax.grid(alpha=.2); ax.legend(fontsize=6)
        fig.suptitle(dataset)
        for ext in ('png', 'pdf'):
            fig.savefig(out / 'figures' / f'{dataset}_coverage.{ext}', dpi=180)
        plt.close(fig)
    report = '''# CV+ con selezione delle epoche interna — 21 settembre 2026

Per ogni fold esterno: training su tre fold; scelta dell'epoca sul quarto;
riaddestramento da zero su tutti e quattro per lo stesso numero di epoche.
Il fold esterno viene utilizzato solo dopo il riaddestramento per le predizioni OOF.
Il training minimizza la loss gaussiana (beta-NLL con beta=0), mentre l'epoca
minimizza l'MSE antisimmetrico medio per proteina sul fold interno completo.
Non si calibra alcun intervallo durante la selezione dell'epoca.
Nessuna somma di curve tra fold esterni. La scelta del criterio MSE è successiva
alle analisi esplorative precedenti su S461L e S669L: questi benchmark non sono
quindi una conferma indipendente del confronto tra criteri di selezione.
Minimo grezzo su 1..300 epoche (o sul limite dichiarato nel manifest); spareggio
all'epoca precedente, senza smoothing o patience. I minimi al limite sono segnalati.

Gli intervalli usano lo stesso checkpoint per residui OOF e predizioni di test.
La predizione puntuale principale è la mediana dei cinque refit; anche la media
è riportata come diagnostica. Sono riportati CV+ con ranghi corretti e quantili
empirici con pari peso per proteina, standard e adattivi. Quest'ultima variante
resta esplorativa: la separazione del tuning non risolve le ipotesi gerarchiche
né giustifica automaticamente una garanzia formale di coverage.
Le coverage per mutazione e a pari peso per proteina sono distinte nelle tabelle.
S461L e S669L si sovrappongono; non sono repliche indipendenti.

## Epoche selezionate

```
'''
    report += epochs.to_string(index=False) + '\n```\n\n## Metriche al 90%\n\n```\n'
    report += results[np.isclose(results.nominal, .9)].to_string(index=False) + '\n```\n'
    (out / 'report.md').write_text(report)
    print(epochs.to_string(index=False), flush=True)
    print(results[np.isclose(results.nominal, .9)].to_string(index=False), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path(__file__).resolve().parent)
    generate(parser.parse_args().run_dir)
