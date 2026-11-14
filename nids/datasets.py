from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nids.config import ExperimentConfig
from nids.paths import cicids2018_dir, wsn_ds_csv


WSN_IDENTIFIER_COLS = {"id", "who ch", "who_ch"}
CICIDS_IDENTIFIER_COLS = {
    "timestamp",
    "flow id",
    "src ip",
    "src port",
    "dst ip",
}
CICIDS_KEEP_NOTE = (
    "Dst Port is kept because it is a standard CICFlowMeter feature, but it can "
    "correlate with service type rather than attack mechanics."
)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [" ".join(str(c).strip().split()) for c in df.columns]
    return df


def _column_key(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def detect_label_column(columns: list[str], dataset: str) -> str:
    keys = {_column_key(c): c for c in columns}
    if dataset == "wsn_ds":
        for candidate in ("attack type", "attack_type", "label"):
            if candidate in keys:
                return keys[candidate]
        raise ValueError(
            "WSN-DS label column not found. Expected 'Attack type'. "
            f"Columns: {columns}"
        )
    for candidate in ("label", "attack type", "attack_type"):
        if candidate in keys:
            col = keys[candidate]
            if candidate != "label":
                raise ValueError(
                    "CICIDS2018 files use a 'Label' column. The legacy scripts "
                    f"looked for '{col}', which belongs to WSN-DS."
                )
            return col
    raise ValueError(f"CICIDS2018 label column 'Label' not found. Columns: {columns}")


def identifier_columns(columns: list[str], dataset: str) -> list[str]:
    banned = WSN_IDENTIFIER_COLS if dataset == "wsn_ds" else CICIDS_IDENTIFIER_COLS
    dropped = [c for c in columns if _column_key(c) in banned]
    return dropped


def _to_binary_labels(y: pd.Series, dataset: str) -> pd.Series:
    if dataset == "wsn_ds":
        return np.where(y.astype(str).str.strip().str.lower() == "normal", "Normal", "Attack")
    return np.where(y.astype(str).str.strip().str.lower() == "benign", "Benign", "Attack")


def _finalize_frame(df: pd.DataFrame, dataset: str, cfg: ExperimentConfig) -> pd.DataFrame:
    df = _normalize_columns(df)
    label_col = detect_label_column(df.columns.tolist(), dataset)
    label_text = df[label_col].astype(str).str.strip().str.lower()
    valid_label = df[label_col].notna() & ~label_text.isin({"", "nan", "none", "null", "<na>"})
    df = df.loc[valid_label].copy()
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[label_col].str.lower() != "label"]
    drop_cols = identifier_columns(df.columns.tolist(), dataset)
    df = df.drop(columns=drop_cols, errors="ignore")
    if cfg.task == "binary":
        df[label_col] = _to_binary_labels(df[label_col], dataset)
    df = df.rename(columns={label_col: "label"})
    return df.reset_index(drop=True)


def load_wsn_ds(cfg: ExperimentConfig) -> tuple[pd.DataFrame, dict]:
    path = wsn_ds_csv()
    raw = pd.read_csv(path, low_memory=False)
    source_rows = int(len(raw))
    df = _finalize_frame(raw, "wsn_ds", cfg)
    meta = {
        "path": str(path),
        "source_rows": source_rows,
        "rows_after_label_cleaning": int(len(df)),
        "identifier_policy": sorted(WSN_IDENTIFIER_COLS),
        "sampling": "Full cleaned dataset loaded; configured row caps are applied to the training partition after splitting.",
    }
    meta["working_rows"] = int(len(df))
    meta["class_counts"] = df["label"].value_counts().to_dict()
    return df, meta


def _cicids_csv_files(
    root: Path, cfg: ExperimentConfig
) -> tuple[list[Path], list[dict[str, str | float]]]:
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CICIDS2018 CSV files under {root}")
    kept = []
    skipped = []
    for file in files:
        size_mb = file.stat().st_size / (1024 * 1024)
        if size_mb > cfg.skip_large_cicids_files_mb:
            skipped.append(
                {"file": file.relative_to(root).as_posix(), "size_mb": round(size_mb, 1)}
            )
            continue
        kept.append(file)
    if not kept:
        raise RuntimeError("All CICIDS2018 CSVs were skipped by the size filter.")
    return kept, skipped


def _reservoir_by_class(
    chunks,
    max_per_class: int,
    seed: int,
) -> pd.DataFrame:
    """Approximate per-class reservoir: keep a random cap from each chunk, then recap.

    Exact one-pass reservoir over millions of rows is too slow in Python. This keeps
    later chunks in play without iterating row by row.
    """
    reservoirs: dict[str, pd.DataFrame] = {}
    for chunk_i, chunk in enumerate(chunks):
        if "label" not in chunk.columns:
            continue
        for label, part in chunk.groupby("label", sort=False):
            take_n = min(len(part), max_per_class)
            if take_n <= 0:
                continue
            sample = part.sample(n=take_n, random_state=seed + chunk_i)
            if label not in reservoirs:
                reservoirs[label] = sample.reset_index(drop=True)
            else:
                merged = pd.concat([reservoirs[label], sample], ignore_index=True)
                if len(merged) > max_per_class:
                    merged = merged.sample(n=max_per_class, random_state=seed + chunk_i)
                reservoirs[label] = merged.reset_index(drop=True)
    return pd.concat(reservoirs.values(), ignore_index=True) if reservoirs else pd.DataFrame()


def load_cicids2018(cfg: ExperimentConfig) -> tuple[pd.DataFrame, dict]:
    root = cicids2018_dir()
    files, skipped = _cicids_csv_files(root, cfg)
    max_per_class = cfg.max_per_class or 15_000
    parts = []
    files_used = []
    file_notes = []
    # Keep the sampled working set fixed when model seeds are repeated.
    per_file_seed = cfg.split_seed
    for file in files:
        usecols_note = None
        header = pd.read_csv(file, nrows=0)
        header = _normalize_columns(header)
        try:
            raw_label = detect_label_column(header.columns.tolist(), "cicids2018")
        except ValueError as exc:
            file_notes.append(
                {"file": file.relative_to(root).as_posix(), "error": str(exc)}
            )
            continue
        chunk_iter = pd.read_csv(file, chunksize=75_000, low_memory=False)
        cleaned_chunks = (
            _finalize_frame(chunk, "cicids2018", cfg) for chunk in chunk_iter
        )
        sampled = _reservoir_by_class(cleaned_chunks, max_per_class, per_file_seed)
        file_notes.append(
            {
                "file": file.relative_to(root).as_posix(),
                "rows_sampled": int(len(sampled)),
                "classes": sampled["label"].value_counts().to_dict() if len(sampled) else {},
                "usecols_note": usecols_note,
            }
        )
        if len(sampled):
            parts.append(sampled)
            files_used.append(file.relative_to(root).as_posix())
        per_file_seed += 1
    if not parts:
        raise RuntimeError("Failed to sample any CICIDS2018 rows.")
    df = pd.concat(parts, ignore_index=True)
    df = _cap_classes(df, cfg, seed=cfg.split_seed)
    meta = {
        "path": str(root),
        "files_used": files_used,
        "files_skipped": skipped,
        "file_notes": file_notes,
        "identifier_policy": sorted(CICIDS_IDENTIFIER_COLS),
        "dst_port_note": CICIDS_KEEP_NOTE,
        "working_rows": int(len(df)),
        "class_counts": df["label"].value_counts().to_dict(),
        "sampling_seed": int(cfg.split_seed),
        "sampling": (
            "Per-file class-capped reservoir sampling, then a global per-class cap. "
            "Sampling happens before the split and over-represents rare attacks. "
            "CICIDS2018 metrics are exploratory and are not prevalence-representative; "
            "do not interpret accuracy or precision as deployment estimates."
        ),
        "test_prevalence_note": "Pre-split class-capped sample; not raw traffic prevalence.",
    }
    return df, meta


def _cap_classes(
    df: pd.DataFrame, cfg: ExperimentConfig, seed: int | None = None
) -> pd.DataFrame:
    seed = cfg.seed if seed is None else seed
    if cfg.max_per_class is not None:
        pieces = []
        for _, group in df.groupby("label", sort=False):
            n = min(len(group), cfg.max_per_class)
            pieces.append(group.sample(n=n, random_state=seed))
        df = pd.concat(pieces, ignore_index=True)
    if cfg.max_total_rows is not None and len(df) > cfg.max_total_rows:
        df = df.sample(n=cfg.max_total_rows, random_state=seed).reset_index(drop=True)
    return df


def load_dataset(cfg: ExperimentConfig) -> tuple[pd.DataFrame, dict]:
    if cfg.dataset == "wsn_ds":
        return load_wsn_ds(cfg)
    if cfg.dataset == "cicids2018":
        return load_cicids2018(cfg)
    raise ValueError(cfg.dataset)


def make_smoke_frame(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 360
    x = rng.normal(size=(n, 8))
    y = np.array(["Normal"] * 200 + ["Flooding"] * 80 + ["Grayhole"] * 80)
    rng.shuffle(y)
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(8)])
    df["id"] = np.arange(n)
    df["who CH"] = rng.integers(0, 12, size=n)
    df["label"] = y
    return df
