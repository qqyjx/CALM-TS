"""Yahoo S5 anomaly benchmark -> windows.npy / labels.npy / split.json.

Source: Yahoo Webscope S5 (academic-only, requires DSA).

The S5 archive contains four sub-benchmarks A1/A2/A3/A4. Each sub-benchmark
is a directory of CSV files; every CSV is a univariate time series with at
least these columns:
    timestamp, value, is_anomaly       (A1, A2 use these names)
    timestamps, value, anomaly         (A3, A4 sometimes use plural / 'anomaly')

The script tolerates both spellings.

Window construction:
    Concatenate every series within a sub-benchmark (preserving series order)
    and slice into 3600-step non-overlapping windows. A window is labeled
    anomaly (1) if any timestamp in it is flagged anomalous, else normal (0).

Output:
    windows.npy   (N, 3600, 1) float32
    labels.npy    (N,) int32 in {0, 1}
    split.json    {"gold":[..], "unlabeled":[..], "test":[..]}
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

WIN_LEN = 3600
N_CHANNELS = 1


def _read_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (values, is_anomaly) for one CSV; tolerant to A1-A4 column variants."""
    values = []
    flags = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)
        cols = {c.lower(): c for c in reader.fieldnames}
        val_col = cols.get("value") or cols.get("val")
        anom_col = (cols.get("is_anomaly") or cols.get("anomaly")
                    or cols.get("isanomaly"))
        if val_col is None:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)
        for row in reader:
            try:
                v = float(row[val_col])
            except (TypeError, ValueError):
                continue
            if anom_col and row.get(anom_col) not in (None, ""):
                try:
                    a = int(float(row[anom_col]))
                except (TypeError, ValueError):
                    a = 0
            else:
                a = 0
            values.append(v)
            flags.append(1 if a else 0)
    return np.asarray(values, dtype=np.float32), np.asarray(flags, dtype=np.int32)


def _slice_windows(values: np.ndarray, flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_win = len(values) // WIN_LEN
    if n_win == 0:
        return (np.empty((0, WIN_LEN, N_CHANNELS), dtype=np.float32),
                np.empty(0, dtype=np.int32))
    v = values[: n_win * WIN_LEN].reshape(n_win, WIN_LEN, 1).astype(np.float32)
    f = flags[: n_win * WIN_LEN].reshape(n_win, WIN_LEN)
    win_labels = (f.any(axis=1)).astype(np.int32)
    return v, win_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="root of S5 extracted archive (contains A1Benchmark/ A2Benchmark/ ...)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--gold-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(in_root.rglob("*.csv"))
    if not csv_files:
        raise SystemExit(f"no .csv files under {in_root}")
    print(f"found {len(csv_files)} csv files under {in_root}")

    all_windows = []
    all_labels = []
    n_series = 0
    for csv_path in csv_files:
        vals, flags = _read_csv(csv_path)
        if vals.size < WIN_LEN:
            continue
        wins, lbls = _slice_windows(vals, flags)
        if wins.shape[0] == 0:
            continue
        all_windows.append(wins)
        all_labels.append(lbls)
        n_series += 1

    if not all_windows:
        raise SystemExit("no series produced any 3600-step windows")
    windows = np.concatenate(all_windows, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int32)
    n_anom = int(labels.sum())
    print(f"series_used={n_series} windows={windows.shape} labels={labels.shape} "
          f"(anomaly={n_anom}/{len(labels)})")

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

