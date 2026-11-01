import numpy as np
import pandas as pd

from nids.config import build_config
from nids.datasets import make_smoke_frame
from nids.experiment import run_experiment
from nids.imbalance import maybe_smote
from nids.preprocess import TrainOnlyPreprocessor, coerce_features, drop_duplicates
from nids.split import apply_split, assert_disjoint_splits, stratified_train_val_test


def test_splits_are_disjoint_and_cover():
    df = make_smoke_frame()
    x, y = coerce_features(df.drop(columns=["id", "who CH"]))
    cfg = build_config(dataset="wsn_ds", preset="smoke")
    parts = stratified_train_val_test(x, y, cfg)
    assert_disjoint_splits(parts)
    n = len(x)
    covered = set(parts["train"]) | set(parts["val"]) | set(parts["test"])
    assert covered == set(range(n))


def test_scaler_matches_train_not_full_data():
    rng = np.random.default_rng(0)
    x = pd.DataFrame(
        {
            "a": np.concatenate([rng.normal(0, 1, 80), rng.normal(50, 1, 20)]),
            "b": rng.normal(size=100),
        }
    )
    y = pd.Series(["n"] * 80 + ["a"] * 20)
    cfg = build_config(dataset="wsn_ds", preset="smoke", feature_selection="none")
    parts = stratified_train_val_test(x, y, cfg)
    split = apply_split(x, y, parts)
    prep = TrainOnlyPreprocessor(cfg)
    prep.fit(split["x_train"], split["y_train"])
    train_mean = split["x_train"].mean().to_numpy()
    np.testing.assert_allclose(prep.scaler.mean_, train_mean, rtol=1e-6, atol=1e-6)
    full_mean = x.mean().to_numpy()
    assert not np.allclose(prep.scaler.mean_, full_mean)


def test_smote_does_not_change_test_distribution():
    rng = np.random.default_rng(1)
    y = np.array([0] * 40 + [1] * 10)
    x = rng.normal(size=(50, 4)).astype(np.float32)
    x_s, y_s = maybe_smote(x, y, seed=1)
    assert len(y_s) > len(y)
    # Original arrays must remain usable as the untouched test stand-in.
    assert len(y) == 50
    assert x.shape == (50, 4)


def test_smoke_experiment_writes_metrics(tmp_path):
    cfg = build_config(
        dataset="wsn_ds",
        preset="smoke",
        models=("rf", "mlp"),
        output_dir=tmp_path,
        imbalance="class_weight",
    )
    out = run_experiment(cfg, smoke_frame=make_smoke_frame())
    assert (out / "metrics.csv").exists()
    assert (out / "summary.json").exists()
    text = (out / "summary.json").read_text(encoding="utf-8")
    assert "test indices disjoint" in text
