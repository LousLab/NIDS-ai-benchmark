from __future__ import annotations

import time
from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from nids.config import ExperimentConfig
from nids.imbalance import class_weight_dict, sample_weight_vector


def _fit_predict_sklearn(model, x_train, y_train, x_test, fit_kwargs=None):
    fit_kwargs = fit_kwargs or {}
    start = time.perf_counter()
    model.fit(x_train, y_train, **fit_kwargs)
    elapsed = time.perf_counter() - start
    pred = model.predict(x_test)
    return pred, elapsed, model


def train_rf(cfg: ExperimentConfig, arrays: dict, use_weights: bool):
    model = RandomForestClassifier(
        n_estimators=cfg.rf_estimators,
        random_state=cfg.seed,
        n_jobs=-1,
        class_weight="balanced" if use_weights else None,
        min_samples_leaf=2,
    )
    return _fit_predict_sklearn(
        model, arrays["x_train"], arrays["y_train"], arrays["x_test"]
    )


def train_xgb(cfg: ExperimentConfig, arrays: dict, use_weights: bool):
    from xgboost import XGBClassifier

    n_classes = int(len(np.unique(arrays["y_train"])))
    model = XGBClassifier(
        n_estimators=cfg.xgb_estimators,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=cfg.seed,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        verbosity=0,
    )
    fit_kwargs = {}
    if use_weights:
        fit_kwargs["sample_weight"] = sample_weight_vector(arrays["y_train"])
    return _fit_predict_sklearn(
        model, arrays["x_train"], arrays["y_train"], arrays["x_test"], fit_kwargs
    )


def train_mlp(cfg: ExperimentConfig, arrays: dict, use_weights: bool):
    """Use Keras when installed, otherwise a fixed-epoch sklearn MLP."""
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=cfg.mlp_epochs,
            early_stopping=False,
            random_state=cfg.seed,
        )
        fit_kwargs = {}
        if use_weights:
            fit_kwargs["sample_weight"] = sample_weight_vector(arrays["y_train"])
        return _fit_predict_sklearn(
            model, arrays["x_train"], arrays["y_train"], arrays["x_test"], fit_kwargs
        )
    return _train_keras_mlp(cfg, arrays, use_weights)


def _train_keras_mlp(cfg: ExperimentConfig, arrays: dict, use_weights: bool):
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input

    tf.keras.utils.set_random_seed(cfg.seed)
    n_classes = int(len(np.unique(arrays["y_train"])))
    model = Sequential(
        [
            Input(shape=(arrays["x_train"].shape[1],)),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(n_classes if n_classes > 2 else 1, activation="softmax" if n_classes > 2 else "sigmoid"),
        ]
    )
    loss = "sparse_categorical_crossentropy" if n_classes > 2 else "binary_crossentropy"
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    class_weight = class_weight_dict(arrays["y_train"]) if use_weights else None
    start = time.perf_counter()
    model.fit(
        arrays["x_train"],
        arrays["y_train"],
        epochs=cfg.mlp_epochs,
        batch_size=128,
        class_weight=class_weight,
        verbose=0,
    )
    elapsed = time.perf_counter() - start
    if n_classes > 2:
        pred = np.argmax(model.predict(arrays["x_test"], verbose=0), axis=1)
    else:
        proba = model.predict(arrays["x_test"], verbose=0).reshape(-1)
        pred = (proba >= 0.5).astype(int)
    return pred, elapsed, model


def train_tabnet(cfg: ExperimentConfig, arrays: dict, use_weights: bool):
    from pytorch_tabnet.tab_model import TabNetClassifier

    model = TabNetClassifier(
        n_d=16,
        n_a=16,
        n_steps=3,
        gamma=1.3,
        lambda_sparse=1e-4,
        optimizer_params={"lr": 2e-2},
        verbose=0,
        seed=cfg.seed,
    )
    start = time.perf_counter()
    model.fit(
        X_train=arrays["x_train"],
        y_train=arrays["y_train"],
        max_epochs=cfg.tabnet_epochs,
        patience=0,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        weights=1 if use_weights else 0,
    )
    elapsed = time.perf_counter() - start
    pred = np.asarray(model.predict(arrays["x_test"])).reshape(-1)
    return pred, elapsed, model


TRAINERS: dict[str, Callable] = {
    "rf": train_rf,
    "xgb": train_xgb,
    "mlp": train_mlp,
    "tabnet": train_tabnet,
}
