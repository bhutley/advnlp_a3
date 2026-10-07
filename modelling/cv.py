"""Firm-grouped cross-validation splits.

All of a firm's calls fall on the same side of every split: the same executive's voice
and the firm's own volatility level would otherwise leak from training into test.
"""

from typing import NamedTuple

import numpy as np


class Split(NamedTuple):
    repeat: int
    fold: int
    train: np.ndarray  # row indices
    test: np.ndarray


def _assign_folds(groups: np.ndarray, n_splits: int, rng: np.random.Generator) -> np.ndarray:
    """Fold number per row. Firms are shuffled, then placed largest first into the
    currently smallest fold, which balances fold sizes; ties are broken by the shuffle."""
    firms, sizes = np.unique(groups, return_counts=True)
    if len(firms) < n_splits:
        raise ValueError(f"{len(firms)} groups can't fill {n_splits} folds")
    order = rng.permutation(len(firms))
    order = order[np.argsort(-sizes[order], kind="stable")]
    fold_of_firm, load = {}, np.zeros(n_splits, dtype=int)
    for i in order:
        fold = int(np.argmin(load))
        fold_of_firm[firms[i]] = fold
        load[fold] += sizes[i]
    return np.array([fold_of_firm[g] for g in groups])


def _splits_from_folds(folds: np.ndarray, n_splits: int):
    rows = np.arange(len(folds))
    return [(rows[folds != k], rows[folds == k]) for k in range(n_splits)]


def outer_splits(groups, n_splits: int = 5, n_repeats: int = 10, seed: int = 0) -> list[Split]:
    """n_repeats × n_splits grouped splits; each repeat reshuffles which firms share a fold."""
    groups = np.asarray(groups)
    splits = []
    for repeat in range(n_repeats):
        folds = _assign_folds(groups, n_splits, np.random.default_rng([seed, repeat]))
        for fold, (train, test) in enumerate(_splits_from_folds(folds, n_splits)):
            splits.append(Split(repeat, fold, train, test))
    return splits


def inner_cv(groups_train, n_splits: int = 4, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped (train, test) splits over the training rows, as indices local to them.

    Passed to GridSearchCV as cv=, so tuning never sees the outer test fold.
    """
    groups_train = np.asarray(groups_train)
    folds = _assign_folds(groups_train, n_splits, np.random.default_rng([seed, 10_000]))
    return _splits_from_folds(folds, n_splits)


def ungrouped(groups) -> np.ndarray:
    """One group per row, so outer_splits and inner_cv ignore firms: the naive split
    (round 2, follow-up C). Used only to show what grouping prevents."""
    return np.array([f"row{i}" for i in range(len(groups))])
