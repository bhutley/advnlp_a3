"""Run ladder rungs under grouped CV, and compare them.

run_rung returns out-of-fold predictions: one row per call per repeat. Hyperparameters are
tuned by an inner grouped CV on each outer training fold, and the choice is recorded.

compare tests whether one rung beats another on the same calls: per-call losses averaged
over repeats, paired, with a bootstrap that resamples firms (repeat firms aren't
independent observations).
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV

from cv import Split, inner_cv
from data import CallData
from features import assemble, past_column, rung_blocks, task_of
from pipelines import ALPHAS, AUDIO_LAYER_GRID, PCA_KS, build_pipeline, param_grid

OOF_COLUMNS = ["event_id", "target", "rung", "repeat", "fold", "y", "y_hat", "params"]
SCORING = {"regression": "neg_mean_squared_error", "classification": "neg_log_loss"}


def _jsonable(params: dict) -> str:
    return json.dumps({k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()},
                      sort_keys=True)  # fmt: skip


def run_rung(data: CallData, rung: str, target: str, splits: list[Split], inner_seed: int = 0,
             n_jobs: int = -1, permute: int | None = None, alphas=ALPHAS, ks=PCA_KS,
             layers=AUDIO_LAYER_GRID) -> pd.DataFrame:  # fmt: skip
    """Out-of-fold predictions for one rung and target.

    y_hat is the prediction (regression) or P(class 1) (classification). B0 is not fitted:
    vol_pre_n for volatility, the training fold's base rate for direction.
    permute: a seed; if given, y is shuffled across calls once, before any split (a null
    check that every rung should fail).
    """
    task = task_of(target)
    y = data.table[target].to_numpy(dtype=float)
    if permute is not None:
        y = np.random.default_rng(permute).permutation(y)
    blocks = rung_blocks(rung, target)
    X, slices = assemble(data, blocks, target)
    ids = data.table["event_id"].to_numpy()

    frames = []
    for split in splits:
        train, test = split.train, split.test
        if rung == "B0":
            if task == "regression":
                y_hat = data.table[past_column(target)].to_numpy()[test]
            else:
                y_hat = np.full(len(test), y[train].mean())
            params = "{}"
        else:
            search = GridSearchCV(
                build_pipeline(slices, task, audio_dim=data.audio_dim),
                param_grid(slices, task, alphas=alphas, ks=ks, layers=layers),
                scoring=SCORING[task],
                cv=inner_cv(data.groups[train], seed=inner_seed + split.repeat),
                n_jobs=n_jobs,
            )
            fit_y = y[train] if task == "regression" else y[train].astype(int)
            search.fit(X[train], fit_y)
            if task == "regression":
                y_hat = search.predict(X[test])
            else:
                y_hat = search.predict_proba(X[test])[:, 1]
            params = _jsonable(search.best_params_)
        frames.append(pd.DataFrame({
            "event_id": ids[test], "target": target, "rung": rung, "repeat": split.repeat,
            "fold": split.fold, "y": y[test], "y_hat": y_hat, "params": params,
        }))  # fmt: skip
    return pd.concat(frames, ignore_index=True)[OOF_COLUMNS]


EPS = 1e-12  # probability clip for log loss


def _repeat_metrics(df: pd.DataFrame, task: str) -> dict:
    y, p = df["y"].to_numpy(), df["y_hat"].to_numpy()
    if task == "regression":
        return {"mse": float(np.mean((y - p) ** 2))}
    return {
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(np.mean((p >= 0.5) == (y == 1))),
        "log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1])),
    }


def metrics(oof: pd.DataFrame) -> pd.DataFrame:
    """Per target × rung × metric: mean and SD over repeats (each repeat scores all calls)."""
    rows = []
    for (target, rung), df in oof.groupby(["target", "rung"], sort=False):
        per_repeat = pd.DataFrame([_repeat_metrics(r, task_of(target)) for _, r in df.groupby("repeat")])
        for metric in per_repeat:
            values = per_repeat[metric]
            rows.append({"target": target, "rung": rung, "metric": metric, "mean": values.mean(),
                         "sd": values.std(ddof=1) if len(values) > 1 else np.nan,
                         "n_repeats": len(values)})  # fmt: skip
    return pd.DataFrame(rows)


def per_call_loss(oof: pd.DataFrame, task: str) -> pd.Series:
    """Each call's loss averaged over repeats: squared error, or log loss for classification."""
    y, p = oof["y"].to_numpy(), oof["y_hat"].to_numpy()
    if task == "regression":
        loss = (y - p) ** 2
    else:
        p = np.clip(p, EPS, 1 - EPS)
        loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return pd.Series(loss, index=oof["event_id"].to_numpy()).groupby(level=0).mean()


def compare(oof: pd.DataFrame, a: str, b: str, target: str, groups: pd.Series,
            n_boot: int = 10_000, seed: int = 0) -> dict:  # fmt: skip
    """Is rung a better than rung b on target?

    mean_diff = mean over calls of (loss_b − loss_a), so positive means a is better.
    groups maps event_id → firm. The 95% CI and one-sided p (H0: a is not better) come from
    a bootstrap that resamples firms with replacement, keeping each firm's calls together.
    """
    task = task_of(target)
    rows = oof[oof["target"] == target]
    loss_a = per_call_loss(rows[rows["rung"] == a], task)
    loss_b = per_call_loss(rows[rows["rung"] == b], task)
    d = (loss_b - loss_a.loc[loss_b.index]).dropna()
    firm = groups.loc[d.index].to_numpy()
    firms, inverse = np.unique(firm, return_inverse=True)
    sums = np.bincount(inverse, weights=d.to_numpy())
    counts = np.bincount(inverse).astype(float)

    draws = np.random.default_rng(seed).integers(0, len(firms), size=(n_boot, len(firms)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "target": target, "a": a, "b": b, "mean_diff": float(d.mean()),
        "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975)),
        "p": float((np.sum(boot <= 0) + 1) / (n_boot + 1)),
        "n_calls": len(d), "n_firms": len(firms),
    }  # fmt: skip


def holm(pvalues: dict) -> dict:
    """Holm–Bonferroni adjusted p-values (family-wise error control), same keys."""
    keys = sorted(pvalues, key=pvalues.get)
    m, running, adjusted = len(keys), 0.0, {}
    for i, key in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        adjusted[key] = running
    return adjusted
