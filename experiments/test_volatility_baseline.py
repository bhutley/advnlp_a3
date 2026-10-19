"""Offline checks for financial timing, purging, and ridge regression."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from volatility_baseline import (bootstrap_mse_difference, chronological_split, company_split,
                                 evaluate, make_labels, ridge_predict, season_split, spearman)


class BaselineTests(unittest.TestCase):
    def test_labels_exclude_call_day_and_first_post_call_return(self):
        dates = pd.bdate_range("2017-01-02", periods=40)
        price = 100 * np.cumprod(1 + np.linspace(-.02, .03, len(dates)))
        with tempfile.TemporaryDirectory() as directory:
            pd.DataFrame({"Date": dates, "Adj Close": price}).to_csv(Path(directory) / "AAA.csv", index=False)
            calls = pd.DataFrame({"call_id": ["one"], "name": ["A"], "call_date": [dates[23]]})
            companies = pd.DataFrame({"name": ["A"], "status": ["ok"], "yahoo_ticker": ["AAA"]})
            result, excluded = make_labels(calls, companies, directory, horizon=7)
        self.assertEqual(excluded, [])
        self.assertEqual(result.iloc[0].anchor_date, dates[24].date().isoformat())
        self.assertEqual(result.iloc[0].label_end, dates[31].date().isoformat())
        returns = pd.Series(price).pct_change().to_numpy()
        self.assertAlmostEqual(result.iloc[0].target, np.log(np.std(returns[25:32], ddof=1)))
        self.assertAlmostEqual(result.iloc[0].past_vol, np.log(np.std(returns[3:23], ddof=1)))

    def test_split_purges_overlapping_outcomes_and_groups_same_dates(self):
        dates = pd.date_range("2017-01-01", periods=30, freq="3D").repeat(2)
        labels = pd.DataFrame({"call_date": dates, "label_end": dates + pd.Timedelta(days=7)})
        train, validation, test = chronological_split(labels)
        self.assertTrue((labels.label_end[train] < labels.call_date[validation].min()).all())
        self.assertTrue((labels.label_end[validation] < labels.call_date[test].min()).all())
        self.assertEqual(set(labels.call_date[train]) & set(labels.call_date[test]), set())

    def test_ridge_recovers_linear_signal_with_constant_column(self):
        x = np.column_stack([np.arange(20), np.ones(20)])
        y = 2 * x[:, 0] + 3
        predictions = ridge_predict(x[:15], y[:15], x[15:], .001)
        np.testing.assert_allclose(predictions, y[15:], atol=.01)

    def test_unpenalized_column_survives_heavy_shrinkage(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(size=(60, 50))
        past = rng.normal(size=60)
        y = 3 * past + 1
        x = np.column_stack([noise, past])
        shrunk = ridge_predict(x[:40], y[:40], x[40:], 1e6)
        free = ridge_predict(x[:40], y[:40], x[40:], 1e6, n_free=1)
        np.testing.assert_allclose(free, y[40:], atol=.05)
        self.assertGreater(np.mean((shrunk - y[40:]) ** 2), 1)
        ols = ridge_predict(past[:40, None], y[:40], past[40:, None], 0., n_free=1)
        np.testing.assert_allclose(ols, y[40:], atol=1e-8)

    def test_price_only_training_runs_without_embedding_dependencies(self):
        dates = pd.date_range("2017-01-01", periods=30, freq="5D")
        labels = pd.DataFrame({"call_date": dates, "label_end": dates + pd.Timedelta(days=7),
                               "past_vol": np.linspace(-5, -3, 30), "target": np.linspace(-5, -3, 30)})
        result = evaluate(labels)
        self.assertEqual(result["past_persistence"]["mse"], 0)
        self.assertTrue(np.isfinite(result["past"]["mse"]))

    def test_embedding_cache_joins_by_id_instead_of_row_order(self):
        dates = pd.date_range("2017-01-01", periods=30, freq="5D")
        ids = np.asarray([f"call-{i}" for i in range(30)])
        signal = np.linspace(-5, -3, 30)
        labels = pd.DataFrame({"call_id": ids, "call_date": dates,
                               "label_end": dates + pd.Timedelta(days=7),
                               "past_vol": signal, "target": signal})
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "features.npz"
            np.savez(cache, call_id=ids[::-1], text=signal[::-1, None],
                     audio=np.ones((30, 2)))
            result = evaluate(labels, cache)
        self.assertLess(result["text"]["mse"], .001)
        self.assertIn("text_audio_past", result)
        self.assertIn("text_audio_past_minus_text_past", result["bootstrap_mse_diff"])

    def test_season_split_uses_fixed_boundaries_and_purges(self):
        dates = pd.to_datetime(["2017-02-01", "2017-06-28", "2017-07-20", "2017-09-28",
                                "2017-10-25", "2017-11-01", "2017-05-01", "2017-08-01"])
        labels = pd.DataFrame({"call_date": dates, "label_end": dates + pd.Timedelta(days=12)})
        train, validation, test = season_split(labels)
        # 06-28 and 09-28 are purged: their outcomes cross the next boundary.
        self.assertEqual(list(train), [True, False, False, False, False, False, True, False])
        self.assertEqual(list(validation), [False, False, True, False, False, False, False, True])
        self.assertEqual(list(test), [False, False, False, False, True, True, False, False])

    def test_refit_uses_validation_for_final_fit(self):
        dates = pd.date_range("2017-01-01", periods=30, freq="5D")
        labels = pd.DataFrame({"call_date": dates, "label_end": dates + pd.Timedelta(days=7),
                               "past_vol": np.linspace(-5, -3, 30), "target": np.linspace(-5, -3, 30)})
        plain = evaluate(labels, split="date", n_boot=0)
        refit = evaluate(labels, split="date", n_boot=0, refit=True)
        self.assertTrue(refit["refit_on_validation"])
        self.assertNotEqual(plain["train_mean"]["mse"], refit["train_mean"]["mse"])

    def test_spearman_is_rank_based(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 1000]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1.0)

    def test_bootstrap_detects_a_better_model_and_resamples_groups(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=200)
        groups = np.repeat(np.arange(40), 5)
        result = bootstrap_mse_difference(y, y + 1, y + .1, groups, n=500)
        self.assertLess(result["ci95"][1], 0)
        self.assertAlmostEqual(result["diff"], .01 - 1)
        self.assertEqual(result["p_b_not_better"], 0)

    def test_company_split_is_disjoint_and_complete(self):
        labels = pd.DataFrame({"name": np.repeat([f"c{i}" for i in range(20)], 3)})
        masks = company_split(labels)
        self.assertTrue((sum(m.astype(int) for m in masks) == 1).all())
        names = [set(labels.name[m]) for m in masks]
        self.assertEqual(names[0] & names[1], set())
        self.assertEqual(names[0] & names[2], set())
        self.assertEqual(names[1] & names[2], set())

    def test_tfidf_benchmark_learns_from_words(self):
        dates = pd.date_range("2017-01-01", periods=40, freq="5D")
        target = np.tile([-5., -3.], 20)
        words = np.where(target > -4, "risk uncertainty decline", "growth record strong")
        labels = pd.DataFrame({"call_id": [f"c{i}" for i in range(40)], "call_date": dates,
                               "label_end": dates + pd.Timedelta(days=7),
                               "past_vol": np.full(40, -4.), "target": target})
        texts = pd.DataFrame({"call_id": labels.call_id, "text": words})
        result = evaluate(labels, texts=texts, n_boot=50)
        self.assertLess(result["tfidf"]["mse"], result["train_mean"]["mse"])
        self.assertIn("tfidf_past", result)


if __name__ == "__main__":
    unittest.main()
