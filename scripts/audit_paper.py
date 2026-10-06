"""Recompute manuscript statistics from frozen predictions; never fit or tune models."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'reproducibility_audit'
sys.path.insert(0, str(ROOT / 'pipeline'))
from protocol import fold_plan, protein_mse, choose_epoch


def active_text(path):
    return '\n'.join(re.sub(r'(?<!\\)%.*', '', line) for line in path.read_text().splitlines())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_npz(path):
    with np.load(path) as z:
        return {key: z[key] for key in z.files}


def bounds(oof, mu, sigma, weights, levels, adaptive):
    """Independent vectorized weighted candidate CDF, matching the declared tie tolerance."""
    score = np.abs(oof['y'] - oof['mu'])
    if adaptive:
        score = score / oof['sigma']
    centers = mu[oof['fold_id']]
    half = score[:, None] * (sigma[oof['fold_id']] if adaptive else 1.)
    outputs = []
    for values, quantiles in [(centers - half, 1 - levels), (centers + half, levels)]:
        order = np.argsort(values, axis=0, kind='stable')
        sorted_values = np.take_along_axis(values, order, axis=0)
        mass = np.cumsum(weights[order].astype(np.longdouble), axis=0)
        mass /= mass[-1:]
        result = np.empty((mu.shape[1], len(levels)))
        for j in range(mu.shape[1]):
            positions = np.searchsorted(mass[:, j], quantiles - 1e-14, side='left')
            result[j] = sorted_values[positions, j]
        outputs.append(result)
    assert np.isfinite(outputs).all() and (outputs[0] <= outputs[1]).all()
    return outputs


def scale_rank(oof, sigma):
    ranks = []
    for k in range(5):
        sample = np.sort(oof['sigma'][oof['fold_id'] == k])
        ranks.append((np.searchsorted(sample, sigma[k], side='left')
                      + np.searchsorted(sample, sigma[k], side='right')) / (2 * len(sample)))
    return np.mean(ranks, axis=0)


def summarize(folder, train, external):
    oof = read_npz(folder / 'oof_all.npz')
    np.testing.assert_array_equal(oof['row_id'], np.arange(len(train)))
    np.testing.assert_array_equal(oof['y'], train.ddG)
    np.testing.assert_array_equal(oof['fold_id'], train.cvfold)
    assert np.isfinite([oof['mu'], oof['sigma']]).all() and (oof['sigma'] > 0).all()
    weights = (1 / (train.wt_seq.nunique() * train.groupby('wt_seq').wt_seq.transform('size'))).to_numpy()
    np.testing.assert_allclose(pd.Series(weights).groupby(train.wt_seq).sum(), 1 / train.wt_seq.nunique())
    levels = np.arange(50, 100) / 100
    stats, direction = {}, []
    for name, frame in external.items():
        z = read_npz(folder / (name + '_cvplus.npz'))
        np.testing.assert_array_equal(z['y'], frame.ddG)
        mu = np.median(z['mu_folds'], axis=0)
        np.testing.assert_array_equal(mu, z['mu_ensemble'])
        rank = scale_rank(oof, z['sigma_folds'])
        np.testing.assert_allclose(rank, z['uncertainty_rank'], rtol=0, atol=1e-14)
        group = np.digitize(rank, [.2, .4, .6, .8])
        error = np.abs(frame.ddG.to_numpy() - mu)
        error_group = np.digitize(error, np.quantile(error, [.2, .4, .6, .8]))
        item = {'pearson': float(pearsonr(frame.ddG, mu).statistic),
                'spearman': float(spearmanr(frame.ddG, mu).statistic),
                'mae': float(error.mean()), 'mse': float(np.mean(error ** 2)),
                'uncertainty_error_rho': float(spearmanr(rank, error).statistic),
                'rows': len(frame), 'proteins': int(frame.wt_seq.nunique()),
                'intervals': {}}
        widths = []
        statuses = []
        for method, adaptive in [('standard', False), ('adaptive', True)]:
            lo, hi = bounds(oof, z['mu_folds'], z['sigma_folds'], weights, levels, adaptive)
            for nominal, alpha in [(80, .2), (90, .1), (95, .05)]:
                key = f'{method}_cvplus_protein_empirical_a{alpha}'
                np.testing.assert_allclose(lo[:, nominal - 50], z[key + '_lower'], rtol=0, atol=1e-12)
                np.testing.assert_allclose(hi[:, nominal - 50], z[key + '_upper'], rtol=0, atol=1e-12)
            covered = (lo <= frame.ddG.to_numpy()[:, None]) & (hi >= frame.ddG.to_numpy()[:, None])
            j = 40  # 90% in the 50--99 grid.
            width = hi[:, j] - lo[:, j]
            protein_coverage = pd.DataFrame({'protein': frame.wt_seq, 'covered': covered[:, j]}).groupby('protein').covered.mean().mean()
            group_coverage = [100 * covered[group == g, j].mean() for g in range(5)]
            interval_score = width + 20 * (np.maximum(lo[:, j] - frame.ddG, 0)
                                          + np.maximum(frame.ddG - hi[:, j], 0))
            item['intervals'][method] = {
                'mutation_coverage_percent': float(100 * covered[:, j].mean()),
                'protein_coverage_percent': float(100 * protein_coverage),
                'mean_width': float(width.mean()), 'minimum_width': float(width.min()),
                'maximum_width': float(width.max()), 'interval_score': float(interval_score.mean()),
                'uncertainty_group_sd_pp': float(np.std(group_coverage, ddof=0)),
                'uncertainty_group_coverage_percent': group_coverage,
                'mean_absolute_grid_discrepancy_pp': float(100 * np.abs(covered[:, 1:].mean(axis=0) - levels[1:]).mean()),
            }
            widths.append(width)
            statuses.append(covered[:, j])
            if adaptive:
                assert (np.diff(lo, axis=1) <= 1e-12).all()
                assert (np.diff(hi, axis=1) >= -1e-12).all()
                for percent in range(50, 96):
                    column = percent - 50
                    positive, negative = lo[:, column] > 0, hi[:, column] < 0
                    calls = positive | negative
                    correct = (positive & (frame.ddG.to_numpy() > 0)) | (negative & (frame.ddG.to_numpy() < 0))
                    protein_calls = frame.loc[calls, 'wt_seq'].nunique()
                    count = int(calls.sum())
                    direction.append({'dataset': name, 'nominal_percent': percent,
                        'stabilizing_mutations': int(positive.sum()), 'destabilizing_mutations': int(negative.sum()),
                        'zero_excluded_mutations': count, 'zero_excluded_percent': 100 * count / len(frame),
                        'proteins_with_any_zero_excluded': int(protein_calls),
                        'proteins_with_any_zero_excluded_percent': 100 * protein_calls / frame.wt_seq.nunique(),
                        'correct_sign_mutations': int(correct.sum()),
                        'sign_accuracy_percent': 100 * correct.sum() / count if count else np.nan})
        delta = widths[1] - widths[0]
        item.update({'narrower': int((delta < 0).sum()), 'wider': int((delta > 0).sum()),
                     'rescued_90': int((~statuses[0] & statuses[1]).sum()),
                     'lost_90': int((statuses[0] & ~statuses[1]).sum()),
                     'error_quintile_width_changes': [float(delta[error_group == g].mean()) for g in range(5)],
                     'high_minus_low_width_contrast': float(delta[error_group == 4].mean() - delta[error_group == 0].mean())})
        stats[name] = item
    return stats, pd.DataFrame(direction)


def excel_values(path):
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            shared = [''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        sheet = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        result = {}
        for row in sheet.findall('.//s:row', ns)[1:]:
            cells = []
            for cell in row.findall('s:c', ns):
                value = cell.find('s:v', ns)
                text = value.text if value is not None else ''.join(cell.itertext())
                if cell.get('t') == 's':
                    text = shared[int(text)]
                cells.append(text)
            if len(cells) >= 3:
                result[cells[1]] = float(cells[2])
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inner-predictions-dir', type=Path,
                        help='Optional archived selection/ directory to recheck all 1500 inner predictions')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    manuscript = ROOT / 'paper_overleaf/paper_cvplus.tex'
    paper = active_text(manuscript)
    train = pd.read_parquet(ROOT / 'data/S2450.parquet').reset_index(drop=True)
    external = {name: pd.read_parquet(ROOT / 'data' / (name + '.parquet')).reset_index(drop=True)
                for name in ['S669L', 'S461L']}
    plan = fold_plan(train)
    epochs = pd.read_csv(ROOT / 'reference/selected_epochs.csv').sort_values('outer_fold')
    curve_rows = []
    for k in range(5):
        curve = pd.read_csv(ROOT / 'reference/selection' / f'outer{k}_curve.csv')
        chosen = choose_epoch(curve, 300)
        assert chosen['selected_epoch'] == int(epochs.iloc[k].selected_epoch)
        error = 0.
        if args.inner_predictions_dir:
            val = train.loc[train.cvfold == (k + 1) % 5].reset_index(drop=True)
            for ep in range(1, 301):
                y, mu, sigma = np.load(args.inner_predictions_dir / f'inner_s0_outer{k}_val{(k+1)%5}_e{ep}_oof.npy')
                np.testing.assert_array_equal(y, val.ddG)
                actual = protein_mse(y, mu, val.wt_seq)['mse_protein']
                error = max(error, abs(actual - curve.loc[curve.epoch == ep, 'mse_protein'].iloc[0]))
            assert error <= 1e-12
        curve_rows.append({'fold': k, 'selected_epoch': chosen['selected_epoch'],
                           'raw_inner_predictions_checked': bool(args.inner_predictions_dir),
                           'maximum_mse_difference': error})
    stats, calls = summarize(ROOT / 'reference/predictions', train, external)
    naive_path = OUT / 'naive_reference'
    naive_provenance = json.loads((naive_path / 'provenance.json').read_text())
    for name, expected in naive_provenance['files_sha256'].items():
        assert digest(naive_path / name) == expected
    naive, _ = summarize(naive_path, train, external)
    checks = []

    def check(label, reported, actual, decimals=2, source='paper_cvplus.tex'):
        expected = f'{float(actual):.{decimals}f}'
        printed = f'{float(reported):.{decimals}f}'
        checks.append({'item': label, 'reported': printed, 'recomputed': float(actual),
                       'expected_printed': expected, 'status': 'PASS' if printed == expected else 'MISMATCH',
                       'source': source})

    # Parse the actual main point-performance table rather than trusting a copied CSV.
    section = paper.split(r'\label{tab:point_prediction_performance}')[1].split(r'\end{table}')[0]
    for line in section.splitlines():
        if line.strip().startswith(('S669L &', 'S461L &')):
            parts = line.split('&'); name = parts[0].strip()
            for field, cell in zip(['pearson', 'spearman', 'mae', 'mse'], parts[1:]):
                number = re.search(r'\d+\.\d+', cell).group()
                check(name + ' point ' + field, float(number), stats[name][field], len(number.split('.')[1]))

    # Parse the actual coverage table, including population SD across the five groups.
    section = paper.split(r'\label{tab:coverage}')[1].split(r'\end{table*}')[0]
    dataset = None
    fields = ['mutation_coverage_percent', 'protein_coverage_percent', 'mean_width',
              'interval_score', 'uncertainty_group_sd_pp']
    for line in section.splitlines():
        if '& Standard &' not in line and '& Adaptive &' not in line:
            continue
        cells = line.split('&')
        if cells[0].strip():
            dataset = cells[0].strip()
        method = cells[1].strip().lower()
        for field, cell in zip(fields, cells[2:]):
            number = re.search(r'\d+\.\d+', cell).group()
            check(dataset + ' ' + method + ' ' + field, float(number),
                  stats[dataset]['intervals'][method][field], len(number.split('.')[1]))

    # Textual diagnostics explicitly stated in Results.
    for name, stated in [('S669L', [4.06, 3.63]), ('S461L', [11.90, 11.63])]:
        for method, value in zip(['standard', 'adaptive'], stated):
            check(name + ' grid discrepancy ' + method, value,
                  stats[name]['intervals'][method]['mean_absolute_grid_discrepancy_pp'])
    for method, stated in [('standard', [4.82, 5.19]), ('adaptive', [3.36, 8.10])]:
        for field, value in zip(['minimum_width', 'maximum_width'], stated):
            check('S669L ' + method + ' ' + field, value, stats['S669L']['intervals'][method][field])
    for field, value in [('narrower', 352), ('wider', 317), ('rescued_90', 7), ('lost_90', 1)]:
        check('S669L ' + field, value, stats['S669L'][field], 0)
    for field, value in [('rescued_90', 4), ('lost_90', 2)]:
        check('S461L ' + field, value, stats['S461L'][field], 0)
    check('S669L lowest-error mean width change', -.11, stats['S669L']['error_quintile_width_changes'][0])
    check('S669L highest-error mean width change', .28, stats['S669L']['error_quintile_width_changes'][4])
    check('S669L high-minus-low contrast', .39, stats['S669L']['high_minus_low_width_contrast'])
    check('S461L high-minus-low contrast', .427, stats['S461L']['high_minus_low_width_contrast'], 3)
    for name, value in [('S669L', .22), ('S461L', .23)]:
        check(name + ' uncertainty-error Spearman', value, stats[name]['uncertainty_error_rho'])

    # Every active numeric cell in the direction-call table.
    table_path = ROOT / 'paper_overleaf/tables/zero_exclusion.tex'
    for line in active_text(table_path).splitlines():
        if not re.match(r'^\s*\d+\s*&', line):
            continue
        cells = line.split('&'); nominal = int(cells[0])
        for name, block in [('S669L', cells[1:6]), ('S461L', cells[6:11])]:
            r = calls[(calls.dataset == name) & (calls.nominal_percent == nominal)].iloc[0]
            reported = [float(x) for cell in block for x in re.findall(r'\d+(?:\.\d+)?', cell)]
            fields = ['stabilizing_mutations', 'destabilizing_mutations', 'zero_excluded_mutations',
                      'zero_excluded_percent', 'proteins_with_any_zero_excluded',
                      'proteins_with_any_zero_excluded_percent', 'sign_accuracy_percent']
            assert len(reported) == len(fields)
            for field, number in zip(fields, reported):
                decimals = 2 if 'percent' in field else 0
                check(f'{name} {nominal}% {field}', number, r[field], decimals, 'tables/zero_exclusion.tex')
    # Abstract uses one decimal and must be checked without double rounding.
    for nominal, rate, accuracy in [(50, 34.2, 94.8), (70, 15.5, 99.0)]:
        r = calls[(calls.dataset == 'S669L') & (calls.nominal_percent == nominal)].iloc[0]
        check(f'Abstract S669L {nominal}% call rate', rate, r.zero_excluded_percent, 1)
        check(f'Abstract S669L {nominal}% sign accuracy', accuracy, r.sign_accuracy_percent, 1)

    # Baseline values stated in its Results paragraph and table.
    for name, values in [('S669L', [.33, .31, 1.23, 2.76]), ('S461L', [.43, .40, 1.00, 1.74])]:
        for field, number in zip(['pearson', 'spearman', 'mae', 'mse'], values):
            check(name + ' naive ' + field, number, naive[name][field])
    section = paper.split(r'\label{tab:naive_coverage}')[1].split(r'\end{table}')[0]
    dataset = None
    for line in section.splitlines():
        if '& Standard &' not in line and '& Adaptive &' not in line:
            continue
        cells = line.split('&')
        if cells[0].strip():
            dataset = cells[0].strip()
        method = cells[1].strip().lower()
        for field, cell in zip(['mutation_coverage_percent', 'protein_coverage_percent', 'mean_width'], cells[2:]):
            number = float(re.search(r'\d+\.\d+', cell).group())
            check(dataset + ' naive ' + method + ' ' + field, number, naive[dataset]['intervals'][method][field])
    check('Naive parameter count', 19394, (42*64+64)+(64*64+64)+(192*64+64)+(64*2+2), 0)

    # Comparative values are source transcriptions, not recomputed benchmark predictions.
    sources = OUT / 'external_sources'
    source_manifest = json.loads((sources / 'manifest.json').read_text())
    for filename, metadata in source_manifest['files'].items():
        assert digest(sources / filename) == metadata['sha256']
    tables = {'S669': [excel_values(sources / 'fig3_a.xlsx'), excel_values(sources / 'fig3_b.xlsx')],
              'S461': [excel_values(sources / 'fig3_c.xlsx'), excel_values(sources / 'fig3_d.xlsx')]}

    def normalized(s):
        s = re.sub(r'_dir$', '', s, flags=re.I)
        s = {'CartddgD': 'Cartddg', 'FoldXD': 'FoldX'}.get(s, s)
        s = re.sub(r'[^a-z0-9]', '', s.lower())
        return 'msatransformermean' if s == 'msatransformer' else s

    section = paper.split(r'\label{tab:point}')[1].split(r'\end{table*}')[0]
    dataset = None
    for line in section.splitlines():
        if r'\textbf{S669}' in line:
            dataset = 'S669'
        elif r'\textbf{S461}' in line:
            dataset = 'S461'
        elif '&' in line and dataset and not line.strip().startswith(('Model ', r'\textbf{')):
            cells = line.split('&')
            if len(cells) != 3 or not re.search(r'\d+\.\d+', cells[1]):
                continue
            model = cells[0].strip()
            for j, metric in enumerate(['Pearson', 'MAE']):
                mapping = {normalized(k): v for k, v in tables[dataset][j].items()}
                value = mapping.get(normalized(model))
                if '--' in cells[j + 1]:
                    assert value is None, (dataset, model, metric, value)
                else:
                    if value is None:
                        raise ValueError('Missing external source: ' + dataset + ' ' + model)
                    number = float(re.search(r'\d+\.\d+', cells[j + 1]).group())
                    check(dataset + ' comparison ' + model + ' ' + metric, number, value, 2,
                          'JanusDDG Fig.3 @ ' + source_manifest['commit'])
    check('Theoretical example empirical bound', .725, 1-.2-2*.9*5/120, 3)
    check('Theoretical example corrected bound', .740, 1-.2-2*.9*4/120, 3)
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(OUT / 'numerical_checks.csv', index=False)
    calls.to_csv(OUT / 'recomputed_sign_calls.csv', index=False)
    pd.DataFrame(curve_rows).to_csv(OUT / 'epoch_checks.csv', index=False)
    result = {'main_manuscript_sha256': digest(manuscript), 'main_model': stats, 'naive_model': naive,
              'data': {'S2450_rows': len(train), 'S2450_proteins': int(train.wt_seq.nunique()),
                       'fold_mutations': train.groupby('cvfold').size().tolist(),
                       'fold_proteins': train.groupby('cvfold').wt_seq.nunique().tolist()},
              'selected_epochs': epochs.selected_epoch.tolist(), 'epoch_checks': curve_rows,
              'checks': len(checks), 'mismatches': checks_frame.loc[checks_frame.status != 'PASS'].to_dict('records'),
              'baseline_full_training_code_present_in_repository': (ROOT / 'naive_model/run.py').exists(),
              'checkpoint_verification_file': 'checkpoint_predictions/verification.json',
              'external_comparison_commit': source_manifest['commit'],
              'note': 'Numerical audit only; not a review of every mathematical proof or a new retraining experiment.'}
    (OUT / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('Numerical checks:', len(checks), 'mismatches:', len(result['mismatches']), flush=True)
    print(checks_frame.loc[checks_frame.status != 'PASS'].to_string(index=False), flush=True)
    print('Main model:', json.dumps(stats, indent=2), flush=True)


if __name__ == '__main__':
    main()
