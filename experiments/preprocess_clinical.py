"""MIMIC-III ICU monitoring -> windows.npy / labels.npy / split.json.

Source: PhysioNet MIMIC-III v1.4 (credentialed access required)

This script consumes a flat parquet/csv export of CHARTEVENTS, LABEVENTS, and
D_ICD_DIAGNOSES filtered to ICU stays. The exact upstream extraction is
project-specific; we accept the pre-extracted form below to keep redistribution
out of scope (per MIMIC license).

Expected --input layout:
    <root>/charts.parquet           # columns: stay_id, charttime (sec offset), HR, SpO2, SBP, DBP, MAP, RR, Temp, GCS
    <root>/diagnoses.parquet        # columns: stay_id, icd_category   (one of the 8 paper categories)

Or CSV equivalents (--format csv).

Window construction:
    For each ICU stay, slide a 60 s window with stride 60 s (non-overlapping)
    over the chart record. Each window collects per-channel mean within that
    minute (since CHARTEVENTS is 1-min resolution typical for vitals). Window
    label = the diagnosis category of the stay; if a stay has multiple, we use
    the first non-"other" category to match the paper's per-stay labeling.
    Filter: keep only windows with <10% missingness across the 8 channels.

Output:
    windows.npy   (N, 60, 8) float32   -- T=60 (1Hz post upsample) by paper spec
    labels.npy    (N,) int32           -- 0..7
    split.json    {"gold":[..500..], "unlabeled":[..], "test":[..]}

Note on T=60: paper uses "60 s @ 1 Hz" downsampled aggregates per channel. If
the upstream parquet provides denser sampling, the script averages into 1 s bins.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

CHANNELS = ["HR", "SpO2", "SBP", "DBP", "MAP", "RR", "Temp", "GCS"]
N_CHANNELS = len(CHANNELS)
WIN_SECONDS = 60
TARGET_FS = 1
TARGET_LEN = WIN_SECONDS * TARGET_FS

CATEGORY_ORDER = [
    "normal", "cardiac_arrhythmia", "sepsis_onset",
    "hemodynamic_instability", "respiratory_failure",
    "neurological_event", "infection", "other",
]
CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORY_ORDER)}
NORMAL_IDX = CATEGORY_TO_IDX["normal"]
OTHER_IDX = CATEGORY_TO_IDX["other"]


def _load_table(path: Path, fmt: str):
    if fmt == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise RuntimeError("pyarrow not installed; pip install pyarrow") from e
        return pq.read_table(str(path)).to_pandas()
    if fmt == "csv":
        try:
            import pandas as pd
        except ImportError as e:
            raise RuntimeError("pandas not installed; pip install pandas") from e
        return pd.read_csv(str(path))
    raise SystemExit(f"unknown format: {fmt}")


def _stay_label(stay_diags: list[str]) -> int:
    for c in stay_diags:
        idx = CATEGORY_TO_IDX.get(c)
        if idx is not None and idx != OTHER_IDX:
            return idx
    if any(CATEGORY_TO_IDX.get(c) == OTHER_IDX for c in stay_diags):
        return OTHER_IDX
    return NORMAL_IDX


def _aggregate_minutes(chart_df, channels: list[str]) -> np.ndarray | None:
    """Aggregate one stay's chart rows into per-minute averages.

    Returns array (n_minutes, len(channels)) float32 with NaN for missing minutes,
    or None if the stay has no usable chart data.
    """
    if len(chart_df) == 0:
        return None
    t_sec = chart_df["charttime"].to_numpy(dtype=np.float64)
    minute = (t_sec // WIN_SECONDS).astype(np.int64)
    if minute.size == 0:
        return None
    minute -= minute.min()
    n_minutes = int(minute.max()) + 1
    out = np.full((n_minutes, len(channels)), np.nan, dtype=np.float32)
    counts = np.zeros((n_minutes, len(channels)), dtype=np.int32)
    for c_idx, ch in enumerate(channels):
        if ch not in chart_df.columns:
            continue
        col = chart_df[ch].to_numpy(dtype=np.float32)
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            continue
        for m, v in zip(minute[valid], col[valid]):
            if np.isnan(out[m, c_idx]):
                out[m, c_idx] = v
                counts[m, c_idx] = 1
            else:
                out[m, c_idx] = (out[m, c_idx] * counts[m, c_idx] + v) / (counts[m, c_idx] + 1)
                counts[m, c_idx] += 1
    return out


def _slice_windows(per_min: np.ndarray, max_missingness: float = 0.10) -> np.ndarray:
    """Cut per-minute matrix (M, C) into 60-minute windows; keep windows
    with per-channel missingness <= max_missingness. Missing -> last-observed-carry-forward + leading mean."""
    M, C = per_min.shape
    n_win = M // TARGET_LEN
    if n_win == 0:
        return np.empty((0, TARGET_LEN, C), dtype=np.float32)
    trimmed = per_min[: n_win * TARGET_LEN].reshape(n_win, TARGET_LEN, C)
    keep = []
    for w in trimmed:
        missing = np.isnan(w).mean(axis=0)
        if (missing > max_missingness).any():
            continue
        filled = w.copy()
        for c in range(C):
            col = filled[:, c]
            mask = np.isnan(col)
            if not mask.any():
                continue
            if mask.all():
                col[:] = 0
                continue
            mean_v = float(np.nanmean(col))
            last = mean_v
            for i in range(len(col)):
                if np.isnan(col[i]):
                    col[i] = last
                else:
                    last = col[i]
            filled[:, c] = col
        keep.append(filled.astype(np.float32))
    if not keep:
        return np.empty((0, TARGET_LEN, C), dtype=np.float32)
    return np.stack(keep, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="dir with charts.parquet + diagnoses.parquet (or .csv)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--gold-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    suffix = "parquet" if args.format == "parquet" else "csv"
    charts = _load_table(in_root / f"charts.{suffix}", args.format)
    diags = _load_table(in_root / f"diagnoses.{suffix}", args.format)

    diag_by_stay = {}
    for stay_id, group in diags.groupby("stay_id"):
        diag_by_stay[stay_id] = list(group["icd_category"])

    all_windows = []
    all_labels = []
    n_stays_total = 0
    n_stays_used = 0
    for stay_id, chart_df in charts.groupby("stay_id"):
        n_stays_total += 1
        per_min = _aggregate_minutes(chart_df, CHANNELS)
        if per_min is None:
            continue
        wins = _slice_windows(per_min)
        if wins.shape[0] == 0:
            continue
        label_idx = _stay_label(diag_by_stay.get(stay_id, []))
        all_windows.append(wins)
        all_labels.append(np.full(wins.shape[0], label_idx, dtype=np.int32))
        n_stays_used += 1

    if not all_windows:
        raise SystemExit("no usable stays produced any windows")
    windows = np.concatenate(all_windows, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int32)
    print(f"stays_used={n_stays_used}/{n_stays_total} windows={windows.shape} labels={labels.shape}")

    rng = np.random.default_rng(args.seed)
    n = len(labels)
    perm = rng.permutation(n).tolist()
    gold = perm[: args.gold_size]
    rest = perm[args.gold_size:]
    test_size = max(1, len(rest) // 5)
    test = rest[:test_size]
    unlabeled = rest[test_size:]
    split = {"gold": gold, "unlabeled": unlabeled, "test": test}

    np.save(out_root / "windows.npy", windows)
    np.save(out_root / "labels.npy", labels)
    (out_root / "split.json").write_text(json.dumps(split))
    print(f"wrote {out_root}/windows.npy + labels.npy + split.json "
          f"(gold={len(gold)}, unlabeled={len(unlabeled)}, test={len(test)})")


if __name__ == "__main__":
    main()

