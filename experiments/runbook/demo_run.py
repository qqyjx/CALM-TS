"""Cost-bounded camera-ready demonstration on PhysioNet Challenge 2015.

Goal: prove the camera-ready pipeline executes end-to-end on freshly produced
PhysioNet data. NOT meant to reproduce paper Table 1 numbers — that needs the
full 7,875-window run with 5 seeds.

Subsamples gold and test sets, runs 1 seed, runs CALM-TS plus four
LLM-based baselines. Writes a separate `results/demo_run.json`
to keep paper numbers untouched.

Token budget (gpt-4o-mini, $0.15/1M in + $0.6/1M out):
    100 gold + 200 test windows, ~12 views per window for CALM-TS/SC,
    1 view for Full-LLM/Conformal-LLM. Roughly 6,500 LLM calls,
    ~$0.50 total.

Usage:
    export OPENAI_API_KEY=...
    export OPENAI_BASE_URL=https://yunwu.ai/v1
    export MODEL=gpt-4o-mini
    python experiments/runbook/demo_run.py
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from lib.calm_ts_core import (  # noqa: E402
    CalmTSPipeline, TemporalSummarizer, MultiViewLabeler,
    Calibrator, SelectiveGate, CostAwareVerifier, FeaturePredictor,
)
from lib.baselines import BASELINES  # noqa: E402
from lib.data_loaders import load_dataset, train_gold_split  # noqa: E402
from lib.evaluation import compute_metrics  # noqa: E402


def subsample_split(split: dict, n_gold: int, n_test: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = dict(split)
    g = split["gold"]
    t = split["test"]
    n_gold = min(n_gold, len(g["labels"]))
    n_test = min(n_test, len(t["labels"]))
    gi = rng.permutation(len(g["labels"]))[:n_gold]
    ti = rng.permutation(len(t["labels"]))[:n_test]
    out["gold"] = {"windows": g["windows"][gi], "labels": g["labels"][gi]}
    out["test"] = {"windows": t["windows"][ti], "labels": t["labels"][ti]}
    return out


def run_calm_ts(split: dict, summarizer: TemporalSummarizer, predictor,
                alpha: float = 0.05) -> dict:
    # Paper-consistent: keep LLM as the labeler in CalmTSPipeline (Stage 2
    # outputs class labels, not agree/disagree). The implementation
    # enhancement is purely on the prompt input -- summarizer attaches an
    # 'ml_hint' line carrying the predictor's top-2 class posteriors so the
    # LLM has additional grounding. The multi-view, calibration and
    # selective-gate stages are byte-identical to the review submission.
    pipe = CalmTSPipeline(
        summarizer=summarizer,
        multi_view=MultiViewLabeler(),
        calibrator=Calibrator(),
        gate=SelectiveGate(alpha=alpha),
        verifier=CostAwareVerifier(budget_pct=0.18),
    )
    print(f"  CALM-TS: labelling {len(split['gold']['windows'])} held-out gold windows (calibration set) ...", flush=True)
    t0 = time.time()
    gold = pipe.run_labeling(
        split["gold"]["windows"], split["channel_names"], split["class_names"]
    )
    print(f"  CALM-TS: gold done ({time.time()-t0:.0f}s), labelling {len(split['test']['windows'])} test windows ...", flush=True)
    correct = np.array([o["pred"] == y for o, y in zip(gold, split["gold"]["labels"])])
    X = np.array([o["feats"] for o in gold])
    pipe.calibrator.fit(X, correct)
    scores = pipe.calibrator.predict_proba(X)
    pipe.gate.fit(scores, correct)

    test = pipe.run_labeling(
        split["test"]["windows"], split["channel_names"], split["class_names"]
    )
    X_test = np.array([o["feats"] for o in test])
    s_test = pipe.calibrator.predict_proba(X_test)
    accept = pipe.gate.accept(s_test)
    pred = np.array([o["pred"] for o in test])
    return compute_metrics(pred, split["test"]["labels"], accept)


def run_baseline(name: str, split: dict, summarizer: TemporalSummarizer) -> dict:
    cls = BASELINES[name]
    baseline = cls(
        labeler=MultiViewLabeler(),
        summarizer=summarizer,
    )
    pred, conf, accept = baseline.label(
        split["test"]["windows"], split["gold"],
        split["channel_names"], split["class_names"],
    )
    return compute_metrics(pred, split["test"]["labels"], accept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gold",  type=int, default=100)
    ap.add_argument("--n-test",  type=int, default=200)
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument("--alpha",   type=float, default=0.05)
    ap.add_argument("--methods", nargs="+",
                    default=["CALM-TS", "Full-LLM", "Self-Consistency", "Conformal-LLM"])
    ap.add_argument("--out", default=str(REPO_ROOT / "results/demo_run.json"))
    args = ap.parse_args()

    print(f"Model: {os.environ.get('MODEL', '(unset; defaults to gpt-4-0613)')}")
    print(f"Base URL: {os.environ.get('OPENAI_BASE_URL', '(default openai.com)')}")

    ds = load_dataset("physionet")
    print(f"physionet shape: windows={ds['windows'].shape}")
    full_split = train_gold_split(ds, gold_size=500, seed=args.seed)
    split = subsample_split(full_split, args.n_gold, args.n_test, args.seed)
    print(f"subsampled: gold={len(split['gold']['labels'])} test={len(split['test']['labels'])}")

    # Hybrid pipeline implementation note:
    # - Stage 1 (FeaturePredictor) needs labelled training data.
    # - Stage 3 (Calibrator) needs (feature_vector, ML-correct?) pairs where
    #   the ML predictor has NOT seen the window during training -- otherwise
    #   the in-sample predictions are 100% correct and the calibrator has no
    #   negative-class signal to learn from.
    # We therefore split the gold set 50/50: the first half trains the
    # predictor, the second half is used as held-out gold for LLM verification
    # plus calibrator + gate fitting.
    summarizer = TemporalSummarizer(n_stats=12, n_freq_bands=5, n_samples=32)
    summarizer.default_class_names = split["class_names"]
    n_gold_total = len(split["gold"]["labels"])
    n_pred_train = max(1, n_gold_total // 2)
    pred_train_windows = split["gold"]["windows"][:n_pred_train]
    pred_train_labels = split["gold"]["labels"][:n_pred_train]
    cal_windows = split["gold"]["windows"][n_pred_train:]
    cal_labels = split["gold"]["labels"][n_pred_train:]
    print(f"gold split: predictor-train={n_pred_train} calibrator-train={len(cal_labels)}")

    print("training feature predictor on first half of gold ...")
    feat_train = summarizer.extract_features(pred_train_windows, split["channel_names"])
    predictor = FeaturePredictor(n_classes=len(split["class_names"]))
    predictor.fit(feat_train, pred_train_labels)
    summarizer.ml_hint_predictor = predictor

    # Sanity: predictor accuracy on held-out gold and on test (full-sample
    # PhysioNet predictor accuracy was ~90% in a separate evaluation; expect
    # lower here because predictor only sees half the gold).
    feat_cal_holdout = summarizer.extract_features(cal_windows, split["channel_names"])
    ml_acc_holdout = (predictor.predict(feat_cal_holdout) == cal_labels).mean()
    feat_test = summarizer.extract_features(split["test"]["windows"], split["channel_names"])
    ml_acc_test = (predictor.predict(feat_test) == split["test"]["labels"]).mean()
    print(f"feature-predictor accuracy: holdout-gold={ml_acc_holdout:.3f} test={ml_acc_test:.3f}")

    # Replace gold split inside `split` with held-out half so run_calm_ts
    # uses LLM labelling on data the predictor has not seen -- needed so the
    # calibrator sees a realistic mix of correct and incorrect LLM labels
    # (predictor's training data would lead to 100%-correct labels and a
    # degenerate calibration signal).
    split["gold"] = {"windows": cal_windows, "labels": cal_labels}

    out: dict = {
        "_meta": {
            "intent": "camera-ready demonstration of pipeline on freshly produced PhysioNet data",
            "model": os.environ.get("MODEL", "gpt-4-0613"),
            "base_url": os.environ.get("OPENAI_BASE_URL", "default"),
            "n_gold": args.n_gold,
            "n_test": args.n_test,
            "seed": args.seed,
            "alpha": args.alpha,
            "methods": args.methods,
            "warning": "subsampled run for cost control; NOT a reproduction of paper Table 1 numbers",
        },
        "results": {},
    }
    t0 = time.time()
    out["_meta"]["feature_predictor_test_accuracy"] = float(ml_acc_test)
    out["_meta"]["feature_predictor_holdout_gold_accuracy"] = float(ml_acc_holdout)
    out["_meta"]["n_predictor_train"] = int(n_pred_train)
    out["_meta"]["n_calibrator_train"] = int(len(cal_labels))
    for m in args.methods:
        t_m = time.time()
        if m == "CALM-TS":
            r = run_calm_ts(split, summarizer, predictor, args.alpha)
        else:
            if m not in BASELINES:
                print(f"unknown method {m}; skipping")
                continue
            r = run_baseline(m, split, summarizer)
        elapsed = time.time() - t_m
        out["results"][m] = {**r, "wall_seconds": round(elapsed, 1)}
        print(f"[{m}] risk={r['risk']:.3f} cov={r['coverage']:.2f} f1={r['f1']:.3f} ({elapsed:.0f}s)")
    out["_meta"]["total_wall_seconds"] = round(time.time() - t0, 1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

