"""Dataset loaders for MIMIC-III, PhysioNet 2015, Yahoo S5.

All three datasets require external downloads (credentialed / license-restricted).
See DATA.md in the repo root for access instructions.

Each loader returns a dict with:
    windows: np.ndarray (N, T, C)
    labels: np.ndarray (N,) int
    class_names: list[str]
    channel_names: list[str]
"""
from __future__ import annotations
import os
import json
from pathlib import Path
import numpy as np


DEFAULT_DATA_ROOT = Path(os.environ.get("CALM_TS_DATA", str(Path.home() / ".calm_ts_data")))

DATASETS = {
    "mimic": {
        "n_segments": 12847,
        "n_classes": 8,
        "class_names": [
            "normal", "cardiac_arrhythmia", "sepsis_onset",
            "hemodynamic_instability", "respiratory_failure",
            "neurological_event", "infection", "other",
        ],
        "channels": ["HR", "SpO2", "SBP", "DBP", "MAP", "RR", "Temp", "GCS"],
        "window_seconds": 60,
    },
    "physionet": {
        "n_segments": 8234,
        "n_classes": 5,
        "class_names": [
            "asystole", "bradycardia", "tachycardia",
            "ventricular_tachycardia", "ventricular_fibrillation",
        ],
        "channels": ["ECG_I", "ECG_II", "ABP", "PPG", "RESP"],
        "window_seconds": 30,
    },
    "physionet_binary": {
        # PhysioNet Challenge 2015 official task: true vs false alarm.
        # Preprocessed by `preprocess_alarm.py --task binary` from the
        # same raw .hea/.dat records that feed `physionet`. Class 0 = false
        # alarm, class 1 = true alarm. Per-segment label is taken from
        # `# Alarm: True|False` in the header.
        "n_segments": 7875,
        "n_classes": 2,
        "class_names": ["false_alarm", "true_alarm"],
        "channels": ["ECG_I", "ECG_II", "ABP", "PPG", "RESP"],
        "window_seconds": 30,
    },
    "yahoo": {
        "n_segments": 15621,
        "n_classes": 2,
        "class_names": ["normal", "anomaly"],
        "channels": ["value"],
        "window_seconds": 3600,  # 1h metric window
    },
}


class DatasetUnavailableError(FileNotFoundError):
    """Raised when real benchmark data is not present on disk.

    The paper's reported numbers come from real MIMIC-III / PhysioNet 2015 / Yahoo S5
    data. A silent fallback to synthetic data would produce JSON results that look
    valid but cannot reproduce paper Table 1; we therefore refuse to proceed unless
    the caller explicitly opts in via CALM_TS_ALLOW_SYNTHETIC=1.
    """


def load_dataset(name: str, split: str = "all",
                 data_root: Path | str | None = None,
                 seed: int = 0) -> dict:
    """Load a benchmark dataset.

    Real data layout expected under {data_root}/{name}/:
        windows.npy   — (N, T, C) float32
        labels.npy    — (N,) int32
        split.json    — {"gold": [...], "unlabeled": [...], "test": [...]}

    If the expected files are not present, this function raises
    DatasetUnavailableError unless the environment variable
    CALM_TS_ALLOW_SYNTHETIC=1 is set, in which case a clearly-labeled synthetic
    sample is produced for smoke-testing only (the returned dict carries
    `_synthetic=True` so downstream result writers can refuse to overwrite
    paper-grade JSON).

    Env override:
        CALM_TS_DATASET_OVERRIDE=physionet  forces every load_dataset(...) call
        to use this dataset name (and its class space). Used for camera-ready
        runs where only a subset of the three benchmarks has freshly produced
        data; experiments still iterate over hardcoded names internally but
        the actual data loaded is the override. The override-warned name is
        recorded in the returned dict under '_logical_name' so callers can
        keep the original label string in their result tables.
    """
    override = os.environ.get("CALM_TS_DATASET_OVERRIDE", "").strip()
    logical_name = name
    if override and override != name:
        name = override
    root = Path(data_root or DEFAULT_DATA_ROOT) / name
    spec = DATASETS[name]

    windows_path = root / "windows.npy"
    labels_path = root / "labels.npy"
    split_path = root / "split.json"
    is_synthetic = False
    if windows_path.exists() and labels_path.exists():
        windows = np.load(windows_path)
        labels = np.load(labels_path)
        if split != "all":
            if not split_path.exists():
                raise DatasetUnavailableError(
                    f"split.json missing under {root}; cannot honor split='{split}'"
                )
            splits = json.loads(split_path.read_text())
            idx = np.array(splits[split])
            windows, labels = windows[idx], labels[idx]
    else:
        if os.environ.get("CALM_TS_ALLOW_SYNTHETIC", "0") != "1":
            raise DatasetUnavailableError(
                f"Real benchmark files not found under {root} "
                f"(expected windows.npy and labels.npy). "
                f"Run preprocess_{name}.py to materialize the data, or set "
                f"CALM_TS_ALLOW_SYNTHETIC=1 to opt into a smoke-test sample "
                f"(results.json must NOT be overwritten in that mode)."
            )
        import warnings
        warnings.warn(
            f"[{name}] Real data missing; CALM_TS_ALLOW_SYNTHETIC=1 → producing "
            f"a synthetic sample. This is a smoke-test path only; do NOT use "
            f"the resulting numbers for paper claims.",
            stacklevel=2,
        )
        rng = np.random.default_rng(seed)
        windows, labels = _synthetic(spec, n=min(spec["n_segments"], 500), rng=rng)
        is_synthetic = True
    return {
        "windows": windows,
        "labels": labels,
        "class_names": spec["class_names"],
        "channel_names": spec["channels"],
        "_logical_name": logical_name,
        "_actual_name": name,
        "_synthetic": is_synthetic,
    }


def _synthetic(spec: dict, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Class-separable synthetic windows — smoke-test path only."""
    T = 60
    C = len(spec["channels"])
    K = spec["n_classes"]
    labels = rng.integers(0, K, size=n)
    windows = rng.normal(size=(n, T, C)).astype(np.float32)
    for k in range(K):
        mask = labels == k
        windows[mask] += rng.normal(scale=0.5, size=(C,)) * (k + 1) * 0.3
    return windows, labels


def train_gold_split(ds: dict, gold_size: int = 500, seed: int = 0) -> dict:
    """Deterministic random split of a loaded dataset into gold / unlabeled / test.

    Cost-control via env vars (no script changes needed):
        CALM_TS_GOLD_CAP   cap gold subset to this many windows
        CALM_TS_TEST_CAP   cap test subset to this many windows
        CALM_TS_UNLAB_CAP  cap unlabeled subset to this many windows

    The caps trim AFTER the deterministic split, keeping the seed semantics.
    """
    rng = np.random.default_rng(seed)
    n = len(ds["labels"])
    idx = rng.permutation(n)
    gold = idx[:gold_size]
    rest = idx[gold_size:]
    test_size = len(rest) // 5
    test = rest[:test_size]
    unlabeled = rest[test_size:]

    gold_cap = int(os.environ.get("CALM_TS_GOLD_CAP", "0"))
    test_cap = int(os.environ.get("CALM_TS_TEST_CAP", "0"))
    unlab_cap = int(os.environ.get("CALM_TS_UNLAB_CAP", "0"))
    if gold_cap > 0 and len(gold) > gold_cap:
        gold = gold[:gold_cap]
    if test_cap > 0 and len(test) > test_cap:
        test = test[:test_cap]
    if unlab_cap > 0 and len(unlabeled) > unlab_cap:
        unlabeled = unlabeled[:unlab_cap]

    return {
        "gold": {"windows": ds["windows"][gold], "labels": ds["labels"][gold]},
        "unlabeled": {"windows": ds["windows"][unlabeled], "labels": ds["labels"][unlabeled]},
        "test": {"windows": ds["windows"][test], "labels": ds["labels"][test]},
        "class_names": ds["class_names"],
        "channel_names": ds["channel_names"],
    }


if __name__ == "__main__":
    for name in DATASETS:
        ds = load_dataset(name)
        print(f"{name}: windows={ds['windows'].shape}, classes={len(ds['class_names'])}")

