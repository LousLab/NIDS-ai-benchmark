from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.is_file() or path.is_dir():
            return path
    return None


def wsn_ds_csv() -> Path:
    env = os.environ.get("WSN_DS_PATH")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            PROJECT_ROOT / "data" / "raw" / "WSN-DS.csv",
            Path.home() / "OneDrive" / "Desktop" / "WSNDS-dataset" / "archive" / "WSN-DS.csv",
            Path.home() / "Downloads" / "WSN-DS.csv",
        ]
    )
    found = _first_existing(candidates)
    if found is None:
        raise FileNotFoundError(
            "WSN-DS.csv not found. Set WSN_DS_PATH or copy the file to data/raw/WSN-DS.csv."
        )
    return found


def cicids2018_dir() -> Path:
    env = os.environ.get("CICIDS2018_DIR")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            PROJECT_ROOT / "data" / "raw" / "cicids2018",
            Path.home()
            / ".cache"
            / "kagglehub"
            / "datasets"
            / "towhidultonmoy"
            / "cicids2018"
            / "versions"
            / "1",
        ]
    )
    found = _first_existing(candidates)
    if found is None:
        raise FileNotFoundError(
            "CICIDS2018 directory not found. Set CICIDS2018_DIR or place CSVs in "
            "data/raw/cicids2018/."
        )
    return found
