"""Small-sample volatility baseline. Requires numpy and pandas; no network calls.

labels: calls.csv must have call_id,name,call_date (one row per call).
train: features.npz must have call_id,text,audio, with one pooled vector per call.
Optional --texts (call_id,text) adds a TF-IDF benchmark; it needs scikit-learn.
See research.md for the experiment definition and embedding extraction approach.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_labels(calls, companies, prices_dir, horizon=7, past_days=20):
    if horizon < 2 or past_days < 2:
        raise ValueError("Volatility windows need at least two returns")
    if calls.call_id.isna().any() or calls.call_id.duplicated().any():
        raise ValueError("call_id must be nonmissing and unique")
    mapping = companies.set_index("name")
    if not mapping.index.is_unique:
        raise ValueError("Company names must be unique")
    rows = []
    excluded = []
    for call in calls.itertuples(index=False):
        if call.name not in mapping.index or mapping.loc[call.name, "status"] != "ok":
            excluded.append((call.call_id, "no verified price mapping"))
            continue
        ticker = mapping.loc[call.name, "yahoo_ticker"]
        price = pd.read_csv(Path(prices_dir) / f"{ticker}.csv")
        dates = pd.to_datetime(price.Date).dt.normalize()
        if dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError(f"{ticker}: dates must be sorted and unique")
        adjusted = pd.to_numeric(price["Adj Close"], errors="raise")
        if not np.isfinite(adjusted).all() or (adjusted <= 0).any():
            raise ValueError(f"{ticker}: adjusted prices must be finite and positive")
        returns = adjusted.pct_change(fill_method=None).to_numpy()
        call_date = pd.Timestamp(call.call_date).normalize()
        # Only date-level timing is available: anchor at first close STRICTLY
        # after the call date, then predict the following horizon returns.
        # This deliberately excludes the immediate earnings-call reaction.
        anchor = int(dates.searchsorted(call_date, side="right"))
        pre = returns[dates.lt(call_date).to_numpy()]
        pre = pre[np.isfinite(pre)][-past_days:]
        post = returns[anchor + 1:anchor + horizon + 1]
        if len(pre) != past_days or len(post) != horizon or not np.isfinite(post).all():
            excluded.append((call.call_id, "insufficient price window"))
            continue
        rows.append({
            "call_id": call.call_id, "name": call.name, "ticker": ticker,
            "call_date": call_date.date().isoformat(),
            "anchor_date": dates.iloc[anchor].date().isoformat(),
            "label_end": dates.iloc[anchor + horizon].date().isoformat(),
            "past_vol": float(np.log(max(np.std(pre, ddof=1), 1e-8))),
            "target": float(np.log(max(np.std(post, ddof=1), 1e-8))),
            # Reported for inspection only; labels are never clipped.
            "max_abs_return": float(np.abs(np.concatenate([pre, post])).max()),
        })
    if not rows:
        raise ValueError("No usable calls: check company names and price coverage")
    return pd.DataFrame(rows), excluded


def chronological_split(labels):
    dates = pd.to_datetime(labels.call_date)
    unique = sorted(dates.unique())
    if len(unique) < 5:
        raise ValueError("Need at least five distinct call dates")
    return purged_split(labels, unique[int(len(unique) * .6)], unique[int(len(unique) * .8)])


def season_split(labels, validation_start="2017-07-01", test_start="2017-10-01"):
    """Boundaries in the gaps between reporting seasons: calls cluster in
    seasons, so date quantiles can land mid-season and purge most of a split.
    Defaults: train = Jan-Jun calls, validation = Q2 season, test = Q3 season."""
    return purged_split(labels, pd.Timestamp(validation_start), pd.Timestamp(test_start))


def purged_split(labels, validation_start, test_start):
    dates = pd.to_datetime(labels.call_date)
    ends = pd.to_datetime(labels.label_end)
    # Purge examples whose outcomes were not available at the next boundary.
    train = (dates < validation_start) & (ends < validation_start)
    validation = (dates >= validation_start) & (dates < test_start) & (ends < test_start)
    test = dates >= test_start
    masks = [m.to_numpy() for m in (train, validation, test)]
    if any(m.sum() < 2 for m in masks):
        raise ValueError("Too few calls after chronological split and purging")
    return masks


def company_split(labels, seed=0):
    """Company-disjoint 60/20/20 split: a sensitivity check for identity
    memorization, not a forecast of future calls."""
    companies = np.array(sorted(labels.name.unique()))
    np.random.default_rng(seed).shuffle(companies)
    cut1, cut2 = int(len(companies) * .6), int(len(companies) * .8)
    groups = [set(companies[:cut1]), set(companies[cut1:cut2]), set(companies[cut2:])]
    masks = [labels.name.isin(g).to_numpy() for g in groups]
    if any(m.sum() < 2 for m in masks):
        raise ValueError("Too few calls after company split")
    return masks


def spearman(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")  # undefined for a constant prediction, e.g. train_mean
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def bootstrap_mse_difference(y, pred_a, pred_b, groups, n=2000, seed=0):
    """Paired bootstrap of MSE(b) - MSE(a), resampling whole groups (companies)
    so calls from the same company stay together. Negative favours b."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in unique]
    loss = (pred_b - y) ** 2 - (pred_a - y) ** 2
    rng = np.random.default_rng(seed)
    draws = np.empty(n)
    for i in range(n):
        picked = np.concatenate([members[j] for j in rng.integers(0, len(unique), len(unique))])
        draws[i] = loss[picked].mean()
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"diff": float(loss.mean()), "ci95": [float(low), float(high)],
            "p_b_not_better": float(np.mean(draws >= 0))}


def ridge_predict(x_train, y_train, x_eval, alpha, standardize=True, n_free=0):
    """Ridge regression whose last n_free columns (e.g. past volatility) are
    unpenalized, so one strong control is not shrunk like each of hundreds of
    embedding dimensions. Uses Frisch-Waugh: partial [1, free] out of y and the
    penalized columns, fit ridge on the residuals, then recover the free part."""
    free_train = np.column_stack([np.ones(len(x_train)), x_train[:, x_train.shape[1] - n_free:]])
    free_eval = np.column_stack([np.ones(len(x_eval)), x_eval[:, x_eval.shape[1] - n_free:]])
    x_train, x_eval = x_train[:, :x_train.shape[1] - n_free], x_eval[:, :x_eval.shape[1] - n_free]
    project = np.linalg.pinv(free_train)  # least squares onto [1, free]
    if x_train.shape[1] == 0:
        return free_eval @ (project @ y_train)
    # Scaling fits on training only. Dual ridge avoids a large feature matrix solve.
    if standardize:
        mean = x_train.mean(axis=0)
        scale = x_train.std(axis=0)
        scale[scale < 1e-8] = 1
    else:
        mean, scale = 0., 1.
    a = (x_train - mean) / scale
    b = (x_eval - mean) / scale
    a_resid = a - free_train @ (project @ a)
    y_resid = y_train - free_train @ (project @ y_train)
    weights = a_resid.T @ np.linalg.solve(a_resid @ a_resid.T + alpha * np.eye(len(a)), y_resid)
    free_coef = project @ (y_train - a @ weights)
    return b @ weights + free_eval @ free_coef


def tfidf_features(texts, labels, train):
    from sklearn.feature_extraction.text import TfidfVectorizer
    lookup = texts.set_index("call_id").text
    missing = sorted(set(labels.call_id) - set(lookup.index))
    if missing:
        raise ValueError(f"No transcript text for {len(missing)} calls, e.g. {missing[:3]}")
    docs = lookup.loc[labels.call_id].to_numpy()
    # Vocabulary and IDF come from training calls only.
    vectorizer = TfidfVectorizer(min_df=2, max_features=20000, sublinear_tf=True,
                                 ngram_range=(1, 2), stop_words="english")
    vectorizer.fit(docs[train])
    return vectorizer.transform(docs).toarray()


def evaluate(labels, features_path=None, texts=None, split="date", n_boot=2000, refit=False):
    splitter = {"company": company_split, "season": season_split, "date": chronological_split}[split]
    train, validation, test = splitter(labels)
    y = labels.target.to_numpy(dtype=float)
    past = labels.past_vol.to_numpy(dtype=float).reshape(-1, 1)
    variants = {"past": past}
    if features_path:
        with np.load(features_path, allow_pickle=False) as features:
            ids = features["call_id"].astype(str)
            if len(set(ids)) != len(ids):
                raise ValueError("Embedding call_id values must be unique")
            lookup = {v: i for i, v in enumerate(ids)}
            missing = [str(v) for v in labels.call_id if str(v) not in lookup]
            if missing:
                raise ValueError(f"No embeddings for {len(missing)} labelled calls, e.g. {missing[:3]}")
            positions = [lookup[str(v)] for v in labels.call_id]
            text = features["text"][positions]
            audio = features["audio"][positions]
            if text.ndim != 2 or audio.ndim != 2:
                raise ValueError("Expected pooled 2D embedding arrays")
            variants.update({
                "text": text, "audio": audio,
                "text_audio": np.column_stack([text, audio]),
                "text_past": np.column_stack([text, past]),
                "audio_past": np.column_stack([audio, past]),
                "text_audio_past": np.column_stack([text, audio, past]),
            })
    if texts is not None:
        # Standardizing would inflate rare n-grams, so TF-IDF stays unscaled.
        tfidf = tfidf_features(texts, labels, train)
        variants["tfidf"] = tfidf
        variants["tfidf_past"] = np.column_stack([tfidf, past])
    if not np.isfinite(np.column_stack([y, *variants.values()])).all():
        raise ValueError("Labels and features must be finite")
    def metrics(pred):
        return {"mse": float(np.mean((pred - y[test]) ** 2)),
                "mae": float(np.mean(np.abs(pred - y[test]))),
                "spearman": spearman(pred, y[test])}
    # With refit, alpha is still chosen on validation, then every model is refit
    # on train+validation. Purging guarantees those outcomes end before the test
    # period, and all models still share the same training observations.
    fit = train | validation if refit else train
    predictions = {
        "train_mean": np.full(test.sum(), y[fit].mean()),
        "past_persistence": past[test, 0],
    }
    results = {
        "split": split, "refit_on_validation": refit,
        "n_train": int(train.sum()), "n_validation": int(validation.sum()),
        "n_test": int(test.sum()),
        "train_mean": metrics(predictions["train_mean"]),
        "past_persistence": metrics(predictions["past_persistence"]),
    }
    for name, x in variants.items():
        standardize = not name.startswith("tfidf")
        # Past volatility is the last column of every *_past variant and is left
        # unpenalized; "past" alone is then ordinary least squares.
        n_free = 1 if name.endswith("past") else 0
        if name == "past":
            alpha = None
        else:
            alphas = [0.1, 1., 10., 100., 1e3, 1e4, 1e5, 1e6]
            errors = [np.mean((ridge_predict(x[train], y[train], x[validation], a, standardize, n_free)
                               - y[validation]) ** 2) for a in alphas]
            alpha = alphas[int(np.argmin(errors))]
        predictions[name] = ridge_predict(x[fit], y[fit], x[test], alpha or 0., standardize, n_free)
        results[name] = {"alpha": alpha, **metrics(predictions[name])}
    if n_boot:
        groups = labels.name[test] if "name" in labels else np.flatnonzero(test)
        comparisons = [("past", name) for name in predictions if name != "past"]
        for pair in [("text_past", "text_audio_past"), ("tfidf_past", "text_past")]:
            if all(p in predictions for p in pair):
                comparisons.append(pair)
        results["bootstrap_mse_diff"] = {
            f"{b}_minus_{a}": bootstrap_mse_difference(y[test], predictions[a], predictions[b], groups, n_boot)
            for a, b in comparisons}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    label = sub.add_parser("labels")
    label.add_argument("--calls", required=True)
    label.add_argument("--companies", default="acl19_proc/data/company.csv")
    label.add_argument("--prices", default="acl19_proc/data")
    label.add_argument("--horizon", type=int, default=7)
    label.add_argument("--out", required=True)
    train = sub.add_parser("train")
    train.add_argument("--labels", required=True)
    train.add_argument("--features")
    train.add_argument("--texts", help="call_texts.csv for the TF-IDF benchmark")
    train.add_argument("--split", choices=["season", "date", "company"], default="season")
    train.add_argument("--no-refit", dest="refit", action="store_false",
                       help="Fit final models on train only instead of train+validation")
    train.add_argument("--bootstrap", type=int, default=2000)
    train.add_argument("--out", help="Also write the JSON results here")
    args = parser.parse_args()
    if args.command == "labels":
        labels, excluded = make_labels(
            pd.read_csv(args.calls, dtype={"call_id": str}),
            pd.read_csv(args.companies), args.prices, args.horizon,
        )
        labels.to_csv(args.out, index=False)
        pd.DataFrame(excluded, columns=["call_id", "reason"]).to_csv(
            str(args.out) + ".excluded.csv", index=False)
        extreme = labels[labels.max_abs_return > .15].sort_values("max_abs_return", ascending=False)
        extreme.to_csv(str(args.out) + ".extreme.csv", index=False)
        print(f"Wrote {len(labels)} labels; excluded {len(excluded)} calls; "
              f"{len(extreme)} have a daily return above 15% (kept, listed for review)")
    else:
        texts = pd.read_csv(args.texts, dtype={"call_id": str}) if args.texts else None
        results = evaluate(pd.read_csv(args.labels, dtype={"call_id": str}), args.features,
                           texts, args.split, args.bootstrap, args.refit)
        output = json.dumps(results, indent=2)
        print(output)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(output)


if __name__ == "__main__":
    main()
