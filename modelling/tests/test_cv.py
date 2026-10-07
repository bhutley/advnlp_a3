import numpy as np
import pytest

from cv import inner_cv, outer_splits

# The modelling sample's firm structure: 87 firms with 1 call, 83 with 2, 45 with 3, 25 with 4.
REAL_SIZES = [1] * 87 + [2] * 83 + [3] * 45 + [4] * 25


def groups_from_sizes(sizes):
    return np.repeat([f"F{i}" for i in range(len(sizes))], sizes)


GROUPS = groups_from_sizes(REAL_SIZES)


def test_real_structure_size():
    assert len(GROUPS) == 488 and len(set(GROUPS)) == 240


def test_outer_splits_shape():
    splits = outer_splits(GROUPS, n_splits=5, n_repeats=3, seed=0)
    assert len(splits) == 15
    assert [(s.repeat, s.fold) for s in splits[:6]] == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0)]


def test_no_firm_on_both_sides():
    for s in outer_splits(GROUPS, n_repeats=3):
        assert not set(GROUPS[s.train]) & set(GROUPS[s.test])
        assert len(s.train) + len(s.test) == len(GROUPS)


def test_every_row_tested_once_per_repeat():
    splits = outer_splits(GROUPS, n_repeats=3)
    for r in range(3):
        tested = np.concatenate([s.test for s in splits if s.repeat == r])
        assert sorted(tested) == list(range(len(GROUPS)))


def test_folds_balanced():
    for s in outer_splits(GROUPS, n_repeats=3):
        assert abs(len(s.test) - len(GROUPS) / 5) <= 0.1 * len(GROUPS) / 5


def test_repeats_differ_and_seed_reproduces():
    a = outer_splits(GROUPS, n_repeats=2, seed=0)
    b = outer_splits(GROUPS, n_repeats=2, seed=0)
    c = outer_splits(GROUPS, n_repeats=2, seed=1)
    assert all(np.array_equal(x.test, y.test) for x, y in zip(a, b))
    assert not np.array_equal(a[0].test, a[5].test)  # repeat 0 vs repeat 1, fold 0
    assert not all(np.array_equal(x.test, y.test) for x, y in zip(a, c))


def test_too_few_groups():
    with pytest.raises(ValueError):
        outer_splits(np.array(["A", "A", "B"]), n_splits=5)


def test_inner_cv_grouped_and_local():
    outer = outer_splits(GROUPS, n_repeats=1)[0]
    g_train = GROUPS[outer.train]

    inner = inner_cv(g_train, n_splits=4, seed=0)

    assert len(inner) == 4
    covered = np.concatenate([te for _, te in inner])
    assert sorted(covered) == list(range(len(g_train)))  # local indices into the training rows
    for tr, te in inner:
        assert not set(g_train[tr]) & set(g_train[te])
        assert len(tr) + len(te) == len(g_train)


# --- ungrouped (round 2, follow-up C) ---


def test_ungrouped_gives_each_row_its_own_group():
    from cv import ungrouped

    g = ungrouped(GROUPS)
    assert len(g) == len(GROUPS) and len(set(g)) == len(GROUPS)


def test_ungrouped_splits_mix_firms():
    from cv import ungrouped

    splits = outer_splits(ungrouped(GROUPS), n_repeats=2)
    for r in range(2):
        tested = np.concatenate([s.test for s in splits if s.repeat == r])
        assert sorted(tested) == list(range(len(GROUPS)))
    # Unlike grouped CV, some firm now appears on both sides of a split.
    assert any(set(GROUPS[s.train]) & set(GROUPS[s.test]) for s in splits)
