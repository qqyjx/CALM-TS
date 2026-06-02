"""Evaluation metrics for CALM-TS and baselines.

The paper defines four headline metrics (Section 5.1):

    Risk    = error rate among accepted labels (target ≤ α)
    Cov     = fraction of unlabeled instances accepted automatically
    Cost    = total annotation expenditure (LLM inference + verification),
              normalised so Full-LLM = 1.00
    F1      = macro-averaged F1 of a LightGBM classifier trained on the
              ACCEPTED labels and evaluated on a held-out test set

The F1 metric in particular is *not* macro-F1 of LLM output vs. ground truth
on accepted samples — it measures the quality of a downstream classifier
that consumes the noisy automatic labels. This module therefore exposes
`downstream_f1(...)` which trains LightGBM (or a sklearn fallback) on
`(features, llm_label)` over accepted samples and reports macro-F1 on
test-set ground truth, matching the paper definition.
"""
from __future__ import annotations
import logging
import numpy as np
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- core metrics


def _train_downstream(X_train: np.ndarray, y_train: np.ndarray,
                      n_classes: int):
    """Train a LightGBM (or sklearn fallback) classifier on accepted labels.

    Returns a fitted estimator with `.predict(X)` available.
    """
    if len(np.unique(y_train)) < 2:
        # degenerate single-class accept set -> majority predictor
        class _Const:
            def __init__(self, v): self.v = v
            def predict(self, X): return np.full(len(X), self.v, dtype=int)
        return _Const(int(y_train[0]) if len(y_train) else 0)
    try:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=300, num_leaves=63, learning_rate=0.05,
            objective="multiclass", num_class=n_classes, verbose=-1,
            min_child_samples=5,
        )
        clf.fit(X_train, y_train)
        return clf
    except Exception as e:
        logger.warning("LightGBM unavailable (%s); falling back to RandomForest", e)
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
        clf.fit(X_train, y_train)
        return clf


def downstream_f1(features: np.ndarray, llm_pred: np.ndarray, accept: np.ndarray,
                  test_features: np.ndarray, test_true: np.ndarray,
                  n_classes: int) -> float:
    """Macro-averaged F1 of a LightGBM classifier trained on accepted labels.

    Args:
        features:      (N, F) features for the unlabeled pool
        llm_pred:      (N,)   LLM-predicted labels for the same pool
        accept:        (N,)   bool gate decisions
        test_features: (M, F) features of the held-out test set
        test_true:     (M,)   ground-truth labels for the test set
        n_classes:     K
    """
    mask = accept.astype(bool)
    if mask.sum() == 0:
        return 0.0
    clf = _train_downstream(features[mask], llm_pred[mask], n_classes)
    preds = clf.predict(test_features)
    return float(f1_score(test_true, preds, average="macro", zero_division=0))


# ---------------------------------------------------------------- cost model


def normalized_cost(accept: np.ndarray,
                    multi_view_token_overhead: float = 12.0,
                    c_llm_per_verify: float = 0.1,
                    verify_unit_cost: float = 1.0) -> float:
    """Cost normalised so Full-LLM (no rejection, single-view) = 1.0.

    Cost decomposition (Section 5.1, Appendix H):
        per_sample_LLM       = M × c_LLM   (M=12 multi-view, c_LLM ≈ 0.1·c_verify)
        per_rejected_sample  = c_verify
        Full-LLM baseline    = N × c_verify   (1 LLM call ≈ verify cost units)

    Returns total_cost / (N × verify_unit_cost). For the paper defaults
    (M=12, c_LLM ≈ 0.1·c_verify) this collapses to:
        cost = (1 - cov) + 12 × 0.1 × cov ≈ (1 - cov) + 1.2·cov   on Full-LLM scale.
    The constant scaling is normalised internally by dividing through Full-LLM.
    """
    N = len(accept)
    if N == 0:
        return 0.0
    n_accepted = int(accept.sum())
    n_rejected = N - n_accepted
    llm_cost = N * multi_view_token_overhead * c_llm_per_verify * verify_unit_cost
    verify_cost = n_rejected * verify_unit_cost
    full_llm_cost = N * verify_unit_cost
    total = llm_cost + verify_cost
    return float(total / max(full_llm_cost, 1e-9))


def compute_metrics(pred: np.ndarray, true: np.ndarray, accept: np.ndarray,
                    *,
                    features: np.ndarray | None = None,
                    test_features: np.ndarray | None = None,
                    test_true: np.ndarray | None = None,
                    n_classes: int | None = None,
                    multi_view_token_overhead: float = 12.0,
                    c_llm_per_verify: float = 0.1,
                    accept_set_f1: bool = True) -> dict:
    """Compute risk / coverage / cost / F1.

    Default (feature-aware) mode: if `features`, `test_features`, and `test_true`
    are provided, F1 is the LightGBM downstream macro-F1 (paper definition).
    Backward-compatible mode (default): if those are absent, F1 is macro-F1
    of (pred[accepted], true[accepted]) — set `accept_set_f1=False` to skip
    the F1 computation entirely.

    Cost is always returned now (no more None) — its formula is described in
    `normalized_cost`.
    """
    m = accept.astype(bool)
    n_accepted = int(m.sum())
    out: dict = {"n_accepted": n_accepted}
    if n_accepted == 0:
        out.update({"risk": 0.0, "coverage": 0.0, "f1": 0.0,
                    "cost": normalized_cost(accept, multi_view_token_overhead, c_llm_per_verify)})
        return out
    out["risk"] = float((pred[m] != true[m]).mean())
    out["coverage"] = float(m.mean())
    out["cost"] = normalized_cost(accept, multi_view_token_overhead, c_llm_per_verify)
    if features is not None and test_features is not None and test_true is not None:
        K = n_classes or int(max(true.max(), pred.max())) + 1
        out["f1"] = downstream_f1(features, pred, accept, test_features, test_true, K)
    elif accept_set_f1:
        out["f1"] = float(f1_score(true[m], pred[m], average="macro", zero_division=0))
    else:
        out["f1"] = 0.0  # explicit: F1 not computed in this mode
    return out


def per_seed_compliance(risks: list[float], alpha: float = 0.05) -> str:
    compliant = sum(1 for r in risks if r <= alpha)
    return f"{compliant}/{len(risks)}"


def f1_on_full_coverage(pred_after_verify: np.ndarray, true: np.ndarray) -> float:
    """For identical-coverage comparison: F1 on all samples after verifier
    relabels gated ones."""
    return float(f1_score(true, pred_after_verify, average="macro", zero_division=0))


# ---------------------------------------------------------------- significance


def paired_wilcoxon(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> dict:
    """Paired Wilcoxon signed-rank test: H0 = a and b have same distribution.

    Falls back to a simple sign test if scipy is unavailable. Returns
    {"statistic": W, "p_value": p, "n_pairs": n}.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = a - b
    valid = diffs[diffs != 0]
    n = int(len(valid))
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_pairs": 0}
    try:
        from scipy.stats import wilcoxon  # type: ignore
        w = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return {"statistic": float(w.statistic), "p_value": float(w.pvalue), "n_pairs": n}
    except Exception:
        # sign test fallback
        from math import comb
        k = int((valid > 0).sum())
        # two-sided p-value under Binomial(n, 0.5)
        from itertools import islice
        def _binom_p(k, n):
            return sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
        p_low = _binom_p(min(k, n - k), n)
        return {"statistic": float(k), "p_value": float(min(1.0, 2 * p_low)), "n_pairs": n}


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values.

    Given m raw p-values, returns the adjusted p-values in the original order
    such that one rejects H0_i at level α iff adj_p_i ≤ α controls the FWER.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj_p = (m - rank) * p[idx]
        running_max = max(running_max, adj_p)
        adj[idx] = min(1.0, running_max)
    return adj.tolist()


def pairwise_significance(per_seed: dict[str, list[float]],
                          target: str = "CALM-TS") -> dict:
    """Pairwise Wilcoxon + Holm-Bonferroni correction across baselines.

    Args:
        per_seed: mapping method_name -> list of per-seed metric values
                  (e.g. risk-violation indicator, or risk values)
        target:   the method to compare every other method against

    Returns:
        dict with raw_p, adjusted_p, n_pairs per other method.
    """
    if target not in per_seed:
        raise KeyError(f"target {target!r} not in per_seed keys {list(per_seed)}")
    others = [k for k in per_seed if k != target]
    raw = []
    pairs = []
    for k in others:
        res = paired_wilcoxon(per_seed[target], per_seed[k])
        raw.append(res["p_value"])
        pairs.append((k, res))
    adj = holm_bonferroni(raw)
    return {
        k: {"raw_p": pairs[i][1]["p_value"],
            "adjusted_p": adj[i],
            "n_pairs": pairs[i][1]["n_pairs"]}
        for i, k in enumerate(others)
    }

