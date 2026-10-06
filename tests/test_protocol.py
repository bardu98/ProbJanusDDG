"""Scientific regression checks for selection, leakage barriers and final evaluation."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent.parent / "pipeline"
sys.path.insert(0, str(HERE))
from protocol import fold_plan, choose_epoch, protein_mse


class ProtocolTests(unittest.TestCase):
    def test_equal_protein_mse(self):
        proteins = np.repeat(['a', 'b', 'c'], [10, 2, 1])
        y = np.repeat([1., 3., 4.], [10, 2, 1])
        score = protein_mse(y, np.zeros(13), proteins)
        self.assertAlmostEqual(score['mse_protein'], (1 + 9 + 16) / 3)
        self.assertAlmostEqual(score['mse_mutation'], (10 + 18 + 16) / 13)

    def test_real_folds_and_label_independence(self):
        data = pd.read_parquet(HERE.parent / 'data/S2450.parquet')
        for plan in fold_plan(data):
            outer = set(plan['outer_rows'])
            self.assertFalse(outer & set(plan['inner_train_rows']))
            self.assertFalse(outer & set(plan['early_stopping_rows']))
            self.assertFalse(outer & set(plan['refit_rows']))
            self.assertEqual(set(plan['refit_rows']), set(plan['inner_train_rows']) | set(plan['early_stopping_rows']))
            val = data.iloc[plan['early_stopping_rows']]
            self.assertTrue(np.isfinite(protein_mse(val.ddG, np.zeros(len(val)), val.wt_seq)['mse_protein']))
            changed = data.copy()
            changed.loc[list(outer), 'ddG'] = 1e8
            self.assertEqual(fold_plan(changed), fold_plan(data))

    def test_epoch_ties_and_missing_curve(self):
        curve = pd.DataFrame(dict(epoch=[3, 2, 1], mse_protein=[2., 2., 3.]))
        self.assertEqual(choose_epoch(curve, 3)['selected_epoch'], 2)
        with self.assertRaises(ValueError):
            choose_epoch(curve.iloc[:2], 3)

    def test_evaluation_artifacts_and_checkpoint_consistency(self):
        from evaluate import generate
        with tempfile.TemporaryDirectory(prefix='cv21_eval_') as tmp:
            out = Path(tmp)
            for name in ('predictions', 'selection'):
                (out / name).mkdir()
            train = pd.DataFrame(dict(cvfold=np.repeat(np.arange(5), 8),
                wt_seq=np.repeat([f'p{i}' for i in range(10)], 4), ddG=np.linspace(-2,2,40)))
            external = pd.DataFrame(dict(wt_seq=['x','x','y','y','z','z'], ddG=np.linspace(-1,1,6)))
            pd.DataFrame(dict(outer_fold=range(5), selected_epoch=[1]*5)).to_csv(out/'selected_epochs.csv', index=False)
            for k in range(5):
                ids = np.flatnonzero(train.cvfold == k)
                np.savez(out/'predictions'/f'oof_f{k}.npz', row_id=ids, y=train.iloc[ids].ddG,
                    mu=train.iloc[ids].ddG-.5, sigma=np.ones(len(ids)), fold_id=np.full(len(ids),k),
                    selected_epoch=1, checkpoint_sha256=f'hash{k}')
                np.savez(out/'predictions'/f'TEST_f{k}.npz', row_id=np.arange(6), y=external.ddG,
                    mu=external.ddG.to_numpy()+k*.01, sigma=np.ones(6), fold_id=np.full(6,k),
                    selected_epoch=1, checkpoint_sha256=f'hash{k}')
                pd.DataFrame(dict(epoch=[1,2], mse_protein=[1.,2.])).to_csv(out/'selection'/f'outer{k}_curve.csv',index=False)
            generate(out, train_data=train, test_data={'TEST':external})
            table = pd.read_csv(out/'metrics.csv')
            self.assertEqual(len(table), 12)
            self.assertTrue((out/'report.md').exists())
            with np.load(out/'predictions/TEST_cvplus.npz') as z:
                bounds = {n:z[n].copy() for n in z.files if n.endswith(('_lower','_upper'))}
            # Test responses can affect reported metrics, never interval endpoints.
            external.ddG += 100
            for k in range(5):
                path = out/'predictions'/f'TEST_f{k}.npz'
                with np.load(path) as z:
                    values = {n:z[n].copy() for n in z.files}
                values['y'] = external.ddG.to_numpy()
                np.savez(path, **values)
            generate(out, train_data=train, test_data={'TEST':external})
            with np.load(out/'predictions/TEST_cvplus.npz') as z:
                for name, expected in bounds.items():
                    np.testing.assert_array_equal(z[name], expected)
            path = out/'predictions/TEST_f0.npz'
            with np.load(path) as z:
                values = {n:z[n].copy() for n in z.files}
            values['checkpoint_sha256'] = 'wrong model'
            np.savez(path, **values)
            with self.assertRaises(AssertionError):
                generate(out, train_data=train, test_data={'TEST':external})


if __name__ == '__main__':
    unittest.main()
