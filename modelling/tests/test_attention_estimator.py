import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from attention import AttentionRegressor, SentenceStore, make_store
from cv import inner_cv, outer_splits

DIM = 6


def planted_store(n_calls=240, n_sent=10, spread=False, seed=0):
    """One marked sentence per call (text dim 0 = +3) carries the signal z in text dim 1.

    spread=True: the signal is the mean of text dim 1 over all sentences, so uniform
    pooling is the right model and attention has nothing to find.
    """
    rng = np.random.default_rng(seed)
    text, audio, sent_idx, y = [], [], [], []
    for _ in range(n_calls):
        t = rng.normal(size=(n_sent, DIM)).astype(np.float32)
        a = rng.normal(size=(n_sent, DIM)).astype(np.float32)
        marked = rng.integers(n_sent)
        t[marked, 0] += 3.0
        y.append(t[:, 1].mean() if spread else t[marked, 1])
        text.append(t)
        audio.append(a)
        sent_idx.append(np.arange(n_sent) * 2)  # sent_idx ≠ position
    small = rng.normal(size=(n_calls, 2)).astype(np.float32)
    y = np.array(y) + rng.normal(0, 0.05, n_calls)
    groups = np.repeat(np.arange(n_calls // 2), 2)
    return make_store(text, audio, sent_idx, small), y, groups


def X_of(idx):
    return np.asarray(idx).reshape(-1, 1)


def fit_eval(store, y, groups, learn_attention, wd=1e-3):
    split = outer_splits(groups, n_splits=4, n_repeats=1)[0]
    est = AttentionRegressor(store, learn_attention=learn_attention, weight_decay=wd,
                             pca_dim=DIM, n_seeds=2, max_iter=100)  # fmt: skip
    est.fit(X_of(split.train), y[split.train])
    return float(np.mean((est.predict(X_of(split.test)) - y[split.test]) ** 2)), est, split


def test_clone_shares_the_store():
    store, _, _ = planted_store(n_calls=10)
    est = AttentionRegressor(store)
    assert clone(est).store is store


def test_planted_single_sentence_attention_wins():
    store, y, groups = planted_store()
    att, est, split = fit_eval(store, y, groups, learn_attention=True)
    uni, _, _ = fit_eval(store, y, groups, learn_attention=False)
    assert att < 0.8 * uni


def test_spread_signal_attention_does_not_win():
    store, y, groups = planted_store(spread=True)
    att, _, _ = fit_eval(store, y, groups, learn_attention=True)
    uni, _, _ = fit_eval(store, y, groups, learn_attention=False)
    assert att >= 0.95 * uni


def test_top_sentences_finds_the_marked_one():
    store, y, groups = planted_store()
    _, est, split = fit_eval(store, y, groups, learn_attention=True)

    top = est.top_sentences(X_of(split.test), k=3)

    assert set(top.columns) == {"call", "modality", "rank", "sent_idx", "weight"}
    first = top[(top["modality"] == "text") & (top["rank"] == 1)].set_index("call")["sent_idx"]
    marked = {c: int(store.sent_idx[c][np.argmax(store.text[c][:, 0])]) for c in split.test}
    hits = np.mean([first[c] == marked[c] for c in split.test])
    assert hits > 0.7


def test_seed_determinism():
    store, y, groups = planted_store(n_calls=60)
    a = AttentionRegressor(store, pca_dim=DIM, n_seeds=2, max_iter=30).fit(X_of(range(40)), y[:40])
    b = AttentionRegressor(store, pca_dim=DIM, n_seeds=2, max_iter=30).fit(X_of(range(40)), y[:40])
    np.testing.assert_array_equal(a.predict(X_of(range(40, 60))), b.predict(X_of(range(40, 60))))


def test_reduction_fitted_on_training_sentences_only():
    store, y, _ = planted_store(n_calls=60, n_sent=10)
    train = np.arange(0, 60, 2)
    est = AttentionRegressor(store, pca_dim=4, n_seeds=1, max_iter=10).fit(X_of(train), y[train])
    assert est.text_scaler_.n_samples_seen_ == 10 * len(train)
    assert est.audio_scaler_.n_samples_seen_ == 10 * len(train)
    assert est.text_pca_.n_components_ == 4


def test_reduction_cached_per_training_set():
    store, y, _ = planted_store(n_calls=40)
    train = np.arange(30)
    a = AttentionRegressor(store, weight_decay=1e-3, pca_dim=4, n_seeds=1, max_iter=5).fit(X_of(train), y[train])
    b = AttentionRegressor(store, weight_decay=1e-1, pca_dim=4, n_seeds=1, max_iter=5).fit(X_of(train), y[train])
    assert a.text_pca_ is b.text_pca_  # same training calls → one PCA fit
    c = AttentionRegressor(store, pca_dim=4, n_seeds=1, max_iter=5).fit(X_of(train[:20]), y[:20])
    assert c.text_pca_ is not a.text_pca_


def test_grid_search_with_inner_cv():
    store, y, groups = planted_store(n_calls=80)
    split = outer_splits(groups, n_splits=4, n_repeats=1)[0]
    search = GridSearchCV(AttentionRegressor(store, pca_dim=DIM, n_seeds=1, max_iter=20),
                          {"weight_decay": [1e-3, 1e-1]}, scoring="neg_mean_squared_error",
                          cv=inner_cv(groups[split.train]), n_jobs=1)  # fmt: skip
    search.fit(X_of(split.train), y[split.train])
    assert search.best_params_["weight_decay"] in (1e-3, 1e-1)
    assert np.isfinite(search.predict(X_of(split.test))).all()
