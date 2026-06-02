"""PhysioNet Challenge 2015 -> windows.npy / labels.npy / split.json.

Source: https://physionet.org/content/challenge-2015/1.0.0/

Layout expected under --input:
    <root>/training/a103l.hea + a103l.mat
    <root>/training/b126s.hea + b126s.mat
    ...
or any directory tree containing *.hea / *.mat record pairs (we glob recursively).

Record naming (PhysioBank convention for this challenge):
    First letter of record id encodes alarm type:
        a -> asystole
        b -> bradycardia
        t -> tachycardia
        v -> ventricular_tachycardia
        f -> ventricular_fibrillation/flutter (merged)

Each record provides 5 min 30 s (~330 s) of multi-channel signals at native fs.
The alarm event sounds at second 300; we cut non-overlapping 30 s windows over
the entire record and assign every window the record's alarm-type label, which
matches the 5-class formulation used in the paper (Table 1).

Output written to --output:
    windows.npy   (N, 7500, 5) float32  -- 30 s @ 250 Hz, 5 channels
    labels.npy    (N,) int32            -- 0..4
    split.json    {"gold":[..500..], "unlabeled":[..], "test":[..]}

Channel selection (deterministic ordering):
    ECG_I  <- 'I' or first lead in {'II','V'} that exists
    ECG_II <- 'II' if available else 'V'
    ABP    <- 'ABP'
    PPG    <- 'PLETH' (PhysioBank uses PLETH for photoplethysmogram == PPG)
    RESP   <- 'RESP'
Missing channels are zero-filled (downstream stats handle constant arrays).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

ALARM_PREFIX_TO_LABEL = {
    "a": 0,  # asystole
    "b": 1,  # bradycardia
    "t": 2,  # tachycardia
    "v": 3,  # ventricular_tachycardia
    "f": 4,  # ventricular_fibrillation/flutter (merged)
}


def _parse_alarm_truth(hea_path: Path) -> int | None:
    """Parse '#True alarm' / '#False alarm' from .hea comments.

    Returns 1 for True (life-threatening), 0 for False, None if not found.
    Challenge-2015 official task is this binary classification.
    """
    try:
        text = hea_path.read_text()
    except Exception:
        return None
    lower = text.lower()
    if "#true alarm" in lower:
        return 1
    if "#false alarm" in lower:
        return 0
    return None

TARGET_FS = 250
WIN_SECONDS = 30
TARGET_LEN = TARGET_FS * WIN_SECONDS  # 7500
TARGET_CHANNELS = ["ECG_I", "ECG_II", "ABP", "PPG", "RESP"]
N_TARGET_CHANNELS = len(TARGET_CHANNELS)

CHANNEL_ALIASES = {
    "ECG_I":  ["I", "i", "ECGI", "ECG-I"],
    "ECG_II": ["II", "ii", "V", "v", "ECGII", "ECG-II"],
    "ABP":    ["ABP", "abp"],
    "PPG":    ["PLETH", "pleth", "PPG", "ppg"],
    "RESP":   ["RESP", "resp", "RESPIRATION"],
}


def _resample_linear(x: np.ndarray, src_fs: float, tgt_fs: float, tgt_len: int) -> np.ndarray:
    """Cheap linear resampler good enough for downstream summary statistics."""
    if src_fs == tgt_fs and len(x) == tgt_len:
        return x.astype(np.float32, copy=False)
    src_t = np.arange(len(x), dtype=np.float64) / src_fs
    tgt_t = np.arange(tgt_len, dtype=np.float64) / tgt_fs
    if tgt_t[-1] > src_t[-1]:
        tgt_t = np.clip(tgt_t, 0, src_t[-1])
    return np.interp(tgt_t, src_t, x).astype(np.float32)


def _pick_channel(sig_names: list[str], target: str) -> int | None:
    aliases = CHANNEL_ALIASES[target]
    upper = [s.upper() for s in sig_names]
    for a in aliases:
        au = a.upper()
        if au in upper:
            return upper.index(au)
    return None


def _label_from_record_id(rec_id: str) -> int | None:
    if not rec_id:
        return None
    return ALARM_PREFIX_TO_LABEL.get(rec_id[0].lower())


def _build_one_record(rec_path: Path) -> tuple[np.ndarray, int] | None:
    """Return (signals, src_fs) where signals is (T_total, 5) at TARGET_FS, or None on failure."""
    try:
        import wfdb
    except ImportError as e:
        raise RuntimeError("wfdb not installed; pip install wfdb") from e
    try:
        rec = wfdb.rdrecord(str(rec_path.with_suffix("")))
    except Exception:
        return None
    sigs = rec.p_signal  # (T, n_sig)
    src_fs = float(rec.fs)
    sig_names = list(rec.sig_name)
    if sigs is None or sigs.size == 0:
        return None
    total_seconds = sigs.shape[0] / src_fs
    if total_seconds < WIN_SECONDS:
        return None

    out_total_len = int(round(total_seconds * TARGET_FS))
    out = np.zeros((out_total_len, N_TARGET_CHANNELS), dtype=np.float32)
    for ch_idx, ch_name in enumerate(TARGET_CHANNELS):
        src_idx = _pick_channel(sig_names, ch_name)
        if src_idx is None:
            continue
        x = sigs[:, src_idx]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        out[:, ch_idx] = _resample_linear(x, src_fs, TARGET_FS, out_total_len)
    return out, int(src_fs)


def _slice_windows(signals: np.ndarray) -> np.ndarray:
    """Cut into non-overlapping 30 s windows. signals: (T_total, C)."""
    T = signals.shape[0]
    n_win = T // TARGET_LEN
    if n_win == 0:
        return np.empty((0, TARGET_LEN, signals.shape[1]), dtype=np.float32)
    trimmed = signals[: n_win * TARGET_LEN]
    return trimmed.reshape(n_win, TARGET_LEN, signals.shape[1]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="root dir containing PhysioBank .hea/.mat records")
    ap.add_argument("--output", required=True, help="output dir for windows.npy/labels.npy/split.json")
    ap.add_argument("--gold-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", choices=["binary", "multiclass"], default="multiclass",
                    help="multiclass (default): 5-class alarm-type label as in paper Table 1; "
                         "binary: PhysioNet Challenge 2015 official true/false alarm task "
                         "(supplementary, useful for cross-checking against the challenge protocol)")
    args = ap.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    hea_files = sorted(in_root.rglob("*.hea"))
    if not hea_files:
        raise SystemExit(f"no .hea files under {in_root}")
    print(f"found {len(hea_files)} record headers under {in_root}")

    all_windows = []
    all_labels = []
    skipped_bad = 0
    skipped_label = 0
    for hea in hea_files:
        rec_id = hea.stem
        if args.task == "binary":
            label = _parse_alarm_truth(hea)
        else:
            label = _label_from_record_id(rec_id)
        if label is None:
            skipped_label += 1
            continue
        built = _build_one_record(hea)
        if built is None:
            skipped_bad += 1
            continue
        signals, _ = built
        wins = _slice_windows(signals)
        if wins.shape[0] == 0:
            skipped_bad += 1
            continue
        all_windows.append(wins)
        all_labels.append(np.full(wins.shape[0], label, dtype=np.int32))

    if not all_windows:
        raise SystemExit("no usable records produced any windows")
    windows = np.concatenate(all_windows, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int32)
    print(f"windows={windows.shape} labels={labels.shape} "
          f"skipped_bad={skipped_bad} skipped_label={skipped_label}")

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

