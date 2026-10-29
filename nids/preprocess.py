from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from nids.config import ExperimentConfig


def coerce_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df["label"].astype(str)
    x = df.drop(columns=["label"])
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    # Drop non-numeric leftovers.
    x = x.select_dtypes(include=[np.number])
    keep = x.notna().any(axis=0) & (x.nunique(dropna=True) > 1)
    x = x.loc[:, keep]
    mask = x.notna().any(axis=1) & y.notna()
    return x.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


def drop_duplicates(x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    combined = x.copy()
    combined["__label__"] = y.to_numpy()
    combined = combined.drop_duplicates()
    y_out = combined.pop("__label__")
    return combined.reset_index(drop=True), y_out.reset_index(drop=True)


class TrainOnlyPreprocessor:
    """Impute, scale, and optionally select features using train data only."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.selector: SelectKBest | None = None
        self.feature_names_: list[str] = []
        self.selected_names_: list[str] = []
        self.label_encoder = LabelEncoder()

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.feature_names_ = list(x_train.columns)
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
