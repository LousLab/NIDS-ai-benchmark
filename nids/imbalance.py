from __future__ import annotations

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def class_weight_dict(y: np.ndarray) -> dict[int, float]:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def sample_weight_vector(y: np.ndarray) -> np.ndarray:
    weights = class_weight_dict(y)
    return np.asarray([weights[int(v)] for v in y], dtype=np.float32)


def maybe_smote(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Oversample the training matrix only. Never call this on val/test."""
    from imblearn.over_sampling import SMOTE

    counts = np.bincount(y)
    minority = int(counts[counts > 0].min())
    if minority < 2:
        raise ValueError("SMOTE requires at least two training samples in every class.")
    k = min(5, minority - 1)
    sampler = SMOTE(random_state=seed, k_neighbors=k)
    return sampler.fit_resample(x, y)
