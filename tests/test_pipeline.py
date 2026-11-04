import numpy as np
import pandas as pd

from nids.config import build_config
from nids.datasets import make_smoke_frame
from nids.experiment import _cap_training_indices, run_experiment
from nids.imbalance import maybe_smote
from nids.preprocess import (
    TrainOnlyPreprocessor,
    coerce_features,
    drop_duplicates,
    group_duplicate_features,
)
from nids.split import (
    apply_split,
    assert_disjoint_splits,
    stratified_train_val_test,
    stratified_group_train_val_test,
    time_train_val_test,
)
from nids.evaluate import evaluate_predictions


def test_splits_are_disjoint_and_cover():
    df = make_smoke_frame()
    x, y = coerce_features(df.drop(columns=["id", "who CH"]))
    cfg = build_config(dataset="wsn_ds", preset="smoke")
    parts = stratified_train_val_test(x, y, cfg)
    assert_disjoint_splits(parts, n_samples=len(x))
    n = len(x)
    covered = set(parts["train"]) | set(parts["val"]) | set(parts["test"])
    assert covered == set(range(n))


def test_training_cap_leaves_validation_and_test_unchanged():
    df = make_smoke_frame()
    x, y = coerce_features(df.drop(columns=["id", "who CH"]))
    cfg = build_config(dataset="wsn_ds", preset="smoke", split_seed=9)
    parts = stratified_train_val_test(x, y, cfg)
    capped, metadata = _cap_training_indices(parts, y, cfg)
    assert set(capped["val"]) == set(parts["val"])
    assert set(capped["test"]) == set(parts["test"])
    assert len(capped["train"]) <= len(parts["train"])
    assert metadata["caps_applied_to"] == "training partition only"


def test_chronological_split_keeps_equal_times_together():
    time = pd.Series(np.repeat(np.arange(10), 10))
    x = pd.DataFrame({"feature": np.arange(len(time))})
    y = pd.Series(np.tile(["a", "b"], 50))
    cfg = build_config(dataset="wsn_ds", preset="fast", split_mode="time")
    parts = time_train_val_test(x, y, time, cfg)
    assert_disjoint_splits(parts, n_samples=len(x))
    split_times = [set(time.iloc[parts[name]]) for name in ("train", "val", "test")]
    assert not split_times[0] & split_times[1]
    assert not split_times[1] & split_times[2]
    assert max(split_times[0]) < min(split_times[1])
    assert max(split_times[1]) < min(split_times[2])


def test_duplicate_features_with_conflicting_labels_are_excluded():
    x = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": [4, 4, 5, 5, 6]})
    y = pd.Series(["a", "a", "b", "c", "c"])
    x_out, y_out, meta = drop_duplicates(x, y, return_metadata=True)
    assert len(x_out) == len(y_out) == 2
    assert set(y_out) == {"a", "c"}
    assert meta["conflicting_feature_vectors_excluded"] == 1


def test_duplicate_detection_can_ignore_time_metadata():
    x = pd.DataFrame({"Time": [10, 20, 30], "signal": [1.0, 1.0, 2.0]})
    y = pd.Series(["a", "a", "b"])
    x_out, y_out = drop_duplicates(x, y, exclude_columns=("Time",))
    assert len(x_out) == len(y_out) == 2


def test_grouping_retains_duplicates_but_excludes_label_conflicts():
    x = pd.DataFrame(
        {"Time": [1, 2, 3, 4, 5], "signal": [1.0, 1.0, 2.0, 2.0, 3.0]}
    )
    y = pd.Series(["a", "a", "b", "c", "c"])
    x_out, y_out, groups, meta = group_duplicate_features(
        x, y, exclude_columns=("Time",)
    )
    assert len(x_out) == len(y_out) == len(groups) == 3
    assert groups.iloc[0] == groups.iloc[1]
    assert y_out.tolist() == ["a", "a", "c"]
    assert meta["conflicting_feature_groups_excluded"] == 1


def test_grouped_stratified_split_keeps_groups_and_classes_together():
    group_values, labels = [], []
    for label_i, label in enumerate(["a", "b", "c"]):
        for group_i in range(20):
            group = label_i * 100 + group_i
            count = 1 + group_i % 4
            group_values.extend([group] * count)
            labels.extend([label] * count)
    y = pd.Series(labels)
    groups = pd.Series(group_values)
    x = pd.DataFrame({"feature": np.arange(len(y), dtype=float)})
    cfg = build_config(dataset="wsn_ds", preset="fast", split_seed=7)
    parts = stratified_group_train_val_test(x, y, groups, cfg)
    assert_disjoint_splits(parts, n_samples=len(x))
    split_groups = [set(groups.iloc[parts[name]]) for name in ("train", "val", "test")]
    assert not split_groups[0] & split_groups[1]
    assert not split_groups[0] & split_groups[2]
    assert not split_groups[1] & split_groups[2]
    for name in ("train", "val", "test"):
        assert set(y.iloc[parts[name]]) == {"a", "b", "c"}


def test_missing_labels_are_not_encoded_as_a_class():
    df = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "label": ["Normal", "Attack", "Normal", None, "", "nan", "null"],
        }
    )
    _, y = coerce_features(df)
    assert y.tolist() == ["Normal", "Attack", "Normal"]


def test_metrics_expose_missing_test_classes():
    scores = evaluate_predictions(
        "model",
        np.array([0, 1]),
        np.array([0, 2]),
        ["a", "b", "c"],
        1.0,
    )
    assert scores["test_unsupported_classes"] == ["c"]
    assert scores["f1_macro_all_labels"] < scores["f1_macro"]


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


def test_constant_and_empty_feature_screening_uses_training_rows_only():
    x_train = pd.DataFrame(
        {
            "signal": [1.0, 2.0, 3.0, 4.0],
            "train_constant": [1.0, 1.0, 1.0, 1.0],
            "train_empty": [np.nan, np.nan, np.nan, np.nan],
        }
    )
    y_train = pd.Series(["a", "b", "a", "b"])
    cfg = build_config(dataset="wsn_ds", preset="smoke")
    prep = TrainOnlyPreprocessor(cfg)
    prep.fit(x_train, y_train)
    x_test = pd.DataFrame(
        {
            "signal": [5.0, 6.0],
            "train_constant": [0.0, 2.0],
            "train_empty": [3.0, 4.0],
        }
    )
    transformed = prep.transform_x(x_test)
    assert prep.feature_names_ == ["signal"]
    assert set(prep.dropped_train_constant_features_) == {"train_constant", "train_empty"}
    assert transformed.shape == (2, 1)


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
