from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from nids.config import ExperimentConfig


def coerce_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    label_text = df["label"].astype(str).str.strip().str.lower()
    valid_label = df["label"].notna() & ~label_text.isin(
        {"", "nan", "none", "null", "<na>"}
    )
    df = df.loc[valid_label].copy()
    y = df["label"].astype(str).str.strip()
    x = df.drop(columns=["label"])
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    # Drop non-numeric leftovers.
    x = x.select_dtypes(include=[np.number])
    mask = x.notna().any(axis=1) & y.notna()
    return x.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


def drop_duplicates(
    x: pd.DataFrame,
    y: pd.Series,
    return_metadata: bool = False,
    exclude_columns: tuple[str, ...] = (),
):
    """Keep one example per feature vector; exclude vectors with conflicting labels.

    Deduplicating on features alone prevents an identical input from appearing in
    separate splits merely because two copies carry different labels.
    """
    dedupe_features = x.drop(columns=list(exclude_columns), errors="ignore")
    if dedupe_features.shape[1] == 0:
        raise ValueError("Cannot deduplicate rows after excluding every feature.")
    row_hash = pd.util.hash_pandas_object(dedupe_features, index=False)
    by_hash = pd.DataFrame({"row_hash": row_hash.to_numpy(), "label": y.to_numpy()})
    label_counts = by_hash.groupby("row_hash", sort=False)["label"].nunique()
    conflicting_hashes = set(label_counts[label_counts > 1].index.tolist())
    unique_mask = ~row_hash.duplicated(keep="first")
    conflict_mask = row_hash.isin(conflicting_hashes)
    keep = unique_mask & ~conflict_mask
    x_out = x.loc[keep].reset_index(drop=True)
    y_out = y.loc[keep].reset_index(drop=True)
    metadata = {
        "rows_before": int(len(x)),
        "unique_feature_vectors": int(row_hash.nunique()),
        "duplicate_feature_rows_removed": int(
            ((~unique_mask) & ~conflict_mask).sum()
        ),
        "conflicting_feature_vectors_excluded": int(len(conflicting_hashes)),
        "conflicting_rows_excluded": int(conflict_mask.sum()),
        "dedupe_excluded_columns": list(exclude_columns),
        "total_rows_removed": int(len(x) - len(x_out)),
        "rows_after": int(len(x_out)),
    }
    if return_metadata:
        return x_out, y_out, metadata
    return x_out, y_out


def group_duplicate_features(
    x: pd.DataFrame,
    y: pd.Series,
    exclude_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """Retain repeated samples but return groups for leakage-safe splitting.

    Feature vectors associated with more than one label are ambiguous and are
    excluded. Remaining identical feature vectors receive the same group ID.
    """
    features = x.drop(columns=list(exclude_columns), errors="ignore")
    if features.shape[1] == 0:
        raise ValueError("Cannot group rows after excluding every feature.")
    row_hash = pd.util.hash_pandas_object(features, index=False)
    by_hash = pd.DataFrame({"row_hash": row_hash.to_numpy(), "label": y.to_numpy()})
    label_counts = by_hash.groupby("row_hash", sort=False)["label"].nunique()
    conflicting_hashes = set(label_counts[label_counts > 1].index.tolist())
    keep = ~row_hash.isin(conflicting_hashes)
    kept_hashes = row_hash.loc[keep].reset_index(drop=True)
    group_ids, unique_hashes = pd.factorize(kept_hashes, sort=False)
    x_out = x.loc[keep].reset_index(drop=True)
    y_out = y.loc[keep].reset_index(drop=True)
    group_out = pd.Series(group_ids, name="feature_group")
    metadata = {
        "rows_before": int(len(x)),
        "unique_feature_groups_before_conflict_removal": int(row_hash.nunique()),
        "duplicate_feature_rows_retained": int(len(x) - int(row_hash.nunique())),
        "conflicting_feature_groups_excluded": int(len(conflicting_hashes)),
        "conflicting_rows_excluded": int((~keep).sum()),
        "unique_feature_groups_after_cleaning": int(len(unique_hashes)),
        "rows_after": int(len(x_out)),
        "grouping_excluded_columns": list(exclude_columns),
    }
    return x_out, y_out, group_out, metadata


class TrainOnlyPreprocessor:
    """Impute, scale, and optionally select features using train data only."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.selector: SelectKBest | None = None
        self.feature_names_: list[str] = []
        self.dropped_train_constant_features_: list[str] = []
        self.selected_names_: list[str] = []
        self.label_encoder = LabelEncoder()

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        keep = x_train.notna().any(axis=0) & (x_train.nunique(dropna=True) > 1)
        self.feature_names_ = list(x_train.columns[keep])
        self.dropped_train_constant_features_ = [
            col for col in x_train.columns if col not in self.feature_names_
        ]
        if not self.feature_names_:
            raise ValueError("No non-constant numeric features remain in the training data.")
        x_train = x_train.reindex(columns=self.feature_names_)
        x_imp = self.imputer.fit_transform(x_train)
        x_scaled = self.scaler.fit_transform(x_imp)
        self.label_encoder.fit(y_train)
        y_enc = self.label_encoder.transform(y_train)
        if self.cfg.feature_selection == "mutual_info":
            k = min(self.cfg.k_features, x_scaled.shape[1])
            self.selector = SelectKBest(score_func=mutual_info_classif, k=k)
            self.selector.fit(x_scaled, y_enc)
            mask = self.selector.get_support()
            self.selected_names_ = [
                name for name, keep in zip(self.feature_names_, mask) if keep
            ]
        else:
            self.selected_names_ = self.feature_names_

    def transform_x(self, x: pd.DataFrame) -> np.ndarray:
        x = x.reindex(columns=self.feature_names_)
        x_imp = self.imputer.transform(x)
        x_scaled = self.scaler.transform(x_imp)
        if self.selector is not None:
            x_scaled = self.selector.transform(x_scaled)
        return x_scaled.astype(np.float32)

    def transform_y(self, y: pd.Series) -> np.ndarray:
        return self.label_encoder.transform(y)

    def class_names(self) -> list[str]:
        return list(self.label_encoder.classes_)
