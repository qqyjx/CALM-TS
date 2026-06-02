"""Baseline implementations for CALM-TS comparison.

Each baseline exposes `.label(windows, gold)` returning (pred_labels, confidences, accept_mask).
All operate on pre-summarized windows to keep costs comparable to CALM-TS.

References for the LLM-era baselines:
  - Snorkel: Ratner et al. 2016, "Data Programming". Generative label model with EM.
  - BADGE: Ash et al. 2020, "Deep Batch Active Learning by Diverse, Uncertain
    Gradient Lower Bounds". K-means++ on gradient embeddings.
  - Prompted-WS: Smith et al. 2024, "Language Models in the Loop:
    Incorporating Prompting into Weak Supervision". LLM-prompted labeling
    functions + generative label model.
  - FreeAL: Xiao et al. 2023, EMNLP. LLMs as collaborative annotators in an
    active learning loop with a small task model.
  - LLMaAA: Zhang et al. 2023, EMNLP. LLM as active annotator with k-NN
    in-context demonstration retrieval.
"""
from __future__ import annotations
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from .calm_ts_core import TemporalSummarizer, MultiViewLabeler, ViewConfig, disagreement_features


def _parallel_single_view(labeler: MultiViewLabeler, summarizer: TemporalSummarizer,
                          view_cfg: ViewConfig, windows, channel_names, class_names,
                          max_workers: int | None = None) -> list[dict]:
    """Run one view per window over many windows, fan-out via thread pool.

    Returns list of view dicts {label, confidence} aligned with `windows`.
    Used by the single-view baselines (Full-LLM, ConfidenceThreshold,
    ConformalLLM, FreeAL, LLMaAA) so they stop being sequential per-window.
    """
    labeler.views = [view_cfg]
    if max_workers is None:
        max_workers = int(os.environ.get("CALM_TS_LLM_WORKERS", "8"))
    out: list[dict] = [None] * len(windows)  # type: ignore

    def _do(i_w):
        i, w = i_w
        s = summarizer.summarize(w, channel_names)
        # Single view -> max_workers=1 view-side; concurrency is at window level.
        v = labeler.label_concurrent(s, class_names, max_workers=1)[0]
        return i, v

    if max_workers <= 1 or len(windows) <= 1:
        for i, w in enumerate(windows):
            out[i] = _do((i, w))[1]
        return out  # type: ignore
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, v in ex.map(_do, [(i, w) for i, w in enumerate(windows)]):
            out[i] = v
    return out  # type: ignore


@dataclass
class FullLLM:
    """Naive Full-LLM baseline: 1 LLM call per sample, no rejection.

    The paper defines Full-LLM as a single-view, no-rejection strategy. We
    therefore replace the labeler's default 12-view ensemble with a single
    standard view for cost parity (cost = 1.00 / sample with 1 LLM call).
    Reverting `self.labeler.views` is owner of the calling experiment if it
    intends to use the same labeler for other baselines.
    """
    labeler: MultiViewLabeler
    summarizer: TemporalSummarizer

    def label(self, windows, gold, channel_names, class_names):
        view = ViewConfig("standard", 0.0, "fullllm")
        out = _parallel_single_view(self.labeler, self.summarizer, view,
                                     windows, channel_names, class_names)
        preds = np.array([v["label"] for v in out])
        confs = np.array([v["confidence"] for v in out])
        accept = np.ones(len(preds), dtype=bool)
        return preds, confs, accept


@dataclass
class SelfConsistency:
    """Majority vote over K independent LLM queries."""
    labeler: MultiViewLabeler
    summarizer: TemporalSummarizer
    n_votes: int = 12

    def label(self, windows, gold, channel_names, class_names):
        views_cfg = [ViewConfig("standard", 0.7, f"vote_{i}") for i in range(self.n_votes)]
        self.labeler.views = views_cfg
        preds, confs = [], []
        for w in windows:
            s = self.summarizer.summarize(w, channel_names)
            vs = self.labeler.label(s, class_names)
            ls = [v["label"] for v in vs]
            cnt = np.bincount(ls, minlength=len(class_names))
            p = int(cnt.argmax())
            preds.append(p)
            confs.append(cnt[p] / self.n_votes)
        accept = np.ones(len(preds), dtype=bool)
        return np.array(preds), np.array(confs), accept


@dataclass
class ConfidenceThreshold:
    """Threshold on raw LLM self-reported confidence. Threshold tuned on gold."""
    labeler: MultiViewLabeler
    summarizer: TemporalSummarizer
    target_alpha: float = 0.05

    def label(self, windows, gold, channel_names, class_names):
        view = ViewConfig("standard", 0.0, "conf_thr")
        gv = _parallel_single_view(self.labeler, self.summarizer, view,
                                    gold["windows"], channel_names, class_names)
        gpreds = np.array([v["label"] for v in gv])
        gconfs = np.array([v["confidence"] for v in gv])
        thr = self._tune_threshold(gpreds, gold["labels"], gconfs)

        out = _parallel_single_view(self.labeler, self.summarizer, view,
                                     windows, channel_names, class_names)
        preds = np.array([v["label"] for v in out])
        confs = np.array([v["confidence"] for v in out])
        accept = confs >= thr
        return preds, confs, accept

    def _tune_threshold(self, preds, labels, confs):
        for thr in np.linspace(0.95, 0.5, 46):
            m = confs >= thr
            if m.sum() == 0:
                continue
            err = (preds[m] != labels[m]).mean()
            if err <= self.target_alpha:
                return thr
        return 0.95


@dataclass
class ConformalLLM:
    """Conformal prediction using LLM nonconformity scores (1 - self-confidence)."""
    labeler: MultiViewLabeler
    summarizer: TemporalSummarizer
    alpha: float = 0.05

    def label(self, windows, gold, channel_names, class_names):
        view = ViewConfig("standard", 0.0, "conformal")
        gv = _parallel_single_view(self.labeler, self.summarizer, view,
                                    gold["windows"], channel_names, class_names)
        nc = np.array([1.0 - v["confidence"] for v in gv])
        q = float(np.quantile(nc, 1 - self.alpha))

        out = _parallel_single_view(self.labeler, self.summarizer, view,
                                     windows, channel_names, class_names)
        preds = np.array([v["label"] for v in out])
        confs = np.array([v["confidence"] for v in out])
        accept = (1.0 - confs) <= q
        return preds, confs, accept


def _label_model_em(votes: np.ndarray, n_classes: int, n_iter: int = 50,
                    eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Snorkel-style generative label model.

    Args:
        votes: (N, M) integer LF votes in [0, n_classes-1] (-1 = abstain)
        n_classes: K
        n_iter: EM iterations

    Returns:
        Y_proba: (N, K) posterior label probabilities
        lf_acc:  (M,)   per-LF estimated accuracy (used as LF reliability weight)

    Models a class-conditional accuracy per LF assuming conditional
    independence given the latent label. Initialised with a uniform
    class prior and per-LF accuracy = 0.7, then refined by EM.
    """
    N, M = votes.shape
    K = n_classes
    log_prior = np.log(np.full(K, 1.0 / K))
    lf_acc = np.full(M, 0.7)  # P(vote==y | y)
    for _ in range(n_iter):
        # E-step: P(y | votes)
        log_post = np.tile(log_prior, (N, 1))
        for m in range(M):
            v = votes[:, m]
            for k in range(K):
                match = (v == k).astype(float)
                abstain = (v < 0).astype(float)
                # P(vote=v | y=k) under simple accuracy model
                p_correct = lf_acc[m]
                p_other = (1.0 - p_correct) / max(K - 1, 1)
                # if vote==k -> p_correct; else if not abstain -> p_other; if abstain -> uniform
                p_vote = match * p_correct + (1 - match - abstain) * p_other + abstain * (1.0 / K)
                log_post[:, k] += np.log(p_vote + eps)
        log_post -= log_post.max(axis=1, keepdims=True)
        post = np.exp(log_post)
        post /= post.sum(axis=1, keepdims=True) + eps
        # M-step: re-estimate lf_acc as expected agreement under posterior
        for m in range(M):
            v = votes[:, m]
            valid = v >= 0
            if valid.sum() == 0:
                continue
            agree = post[np.arange(N), v.clip(0, K - 1)]
            est = (agree * valid).sum() / max(valid.sum(), 1)
            lf_acc[m] = float(np.clip(est, 1.0 / K + eps, 1 - eps))
        # update class prior
        log_prior = np.log(post.mean(axis=0) + eps)
    return post, lf_acc


@dataclass
class Snorkel:
    """Snorkel-style weak supervision: prompt-based LFs + generative label model.

    Three prompts at T=0.0 act as labeling functions; a class-conditional
    accuracy label model is fit by EM (`_label_model_em`); the posterior over
    the latent label gives both the prediction and the confidence.

    Snorkel by design does not selectively reject samples, so coverage = 1
    (all predictions are accepted). Gating decisions, if any, must come from
    a downstream component (which CALM-TS supplies and Snorkel does not).
    """
    labeler: MultiViewLabeler
    summarizer: TemporalSummarizer
    lfs: list[ViewConfig] = field(default_factory=lambda: [
        ViewConfig("narrow", 0.0, "LF_narrow"),
        ViewConfig("standard", 0.0, "LF_standard"),
        ViewConfig("wide", 0.0, "LF_wide"),
    ])

    def _collect_votes(self, windows, channel_names, class_names):
        self.labeler.views = self.lfs
        votes = np.full((len(windows), len(self.lfs)), -1, dtype=int)
        for i, w in enumerate(windows):
            s = self.summarizer.summarize(w, channel_names)
            vs = self.labeler.label(s, class_names)
            for j, v in enumerate(vs):
                votes[i, j] = int(v["label"])
        return votes

    def label(self, windows, gold, channel_names, class_names):
        votes = self._collect_votes(windows, channel_names, class_names)
        post, _ = _label_model_em(votes, len(class_names))
        preds = post.argmax(axis=1)
        confs = post.max(axis=1)
        accept = np.ones(len(preds), dtype=bool)
        return preds, confs, accept


@dataclass
class BADGE:
    """Active learning via K-means++ on gradient embeddings (Ash et al. 2020).

    Faithful to the paper: each unlabeled sample contributes a hypothetical
    loss-gradient embedding ∇_θ ℓ(f_θ(x), ŷ), which we approximate as
    `(p - one_hot(argmax_p)) ⊗ feat` where p is the multi-view label
    distribution and feat is the disagreement-feature vector. K-means++
    seeding selects a diverse, uncertain subset.
    """
    labeler: MultiViewLabeler
    summarizer: TemporalSummarizer
    accept_frac: float = 0.70

    def _kmeanspp_indices(self, embeds: np.ndarray, k: int, rng) -> np.ndarray:
        n = len(embeds)
        if k >= n:
            return np.arange(n)
        first = int(rng.integers(0, n))
        chosen = [first]
        d2 = np.sum((embeds - embeds[first]) ** 2, axis=1)
        for _ in range(k - 1):
            if d2.sum() <= 0:
                break
            probs = d2 / d2.sum()
            j = int(rng.choice(n, p=probs))
            chosen.append(j)
            new_d2 = np.sum((embeds - embeds[j]) ** 2, axis=1)
            d2 = np.minimum(d2, new_d2)
        return np.array(sorted(set(chosen)))

    def label(self, windows, gold, channel_names, class_names):
        K = len(class_names)
        preds, confs, embeds = [], [], []
        for w in windows:
            s = self.summarizer.summarize(w, channel_names)
            vs = self.labeler.label(s, class_names)
            labs = np.array([v["label"] for v in vs])
            counts = np.bincount(labs, minlength=K).astype(float)
            p = counts / counts.sum()
            yhat = int(p.argmax())
            feat = disagreement_features(vs, K)
            grad_emb = np.outer(p - np.eye(K)[yhat], feat).ravel()
            preds.append(yhat)
            confs.append(float(p[yhat]))
            embeds.append(grad_emb)
        preds = np.array(preds); confs = np.array(confs); embeds = np.array(embeds)
        rng = np.random.default_rng(0)
        n_keep = max(1, int(self.accept_frac * len(preds)))
        idx = self._kmeanspp_indices(embeds, n_keep, rng)
        accept = np.zeros(len(preds), dtype=bool)
        accept[idx] = True
        return preds, confs, accept


@dataclass
class PromptedWS(Snorkel):
    """Prompted weak supervision (Smith et al. 2024).

    Differs from vanilla Snorkel in two ways:
      1. Each labeling function is a *prompt template* whose output we read
         alongside its self-reported confidence.
      2. Low-confidence votes (< abstain_threshold) are recorded as abstains
         (-1) rather than label votes; the label model then weighs the LF
         lower automatically.
    """
    abstain_threshold: float = 0.6

    def _collect_votes(self, windows, channel_names, class_names):
        self.labeler.views = self.lfs
        votes = np.full((len(windows), len(self.lfs)), -1, dtype=int)
        for i, w in enumerate(windows):
            s = self.summarizer.summarize(w, channel_names)
            vs = self.labeler.label(s, class_names)
            for j, v in enumerate(vs):
                if float(v["confidence"]) >= self.abstain_threshold:
                    votes[i, j] = int(v["label"])
        return votes


@dataclass
class FreeAL(FullLLM):
    """FreeAL (Xiao et al. 2023): LLM as collaborative annotator with a
    small task model in an active learning loop.

    Implementation:
      1. Round 0: LLM labels every test window once (T=0 standard prompt).
      2. Train a small task model (LightGBM if available, else logistic
         regression on summarizer features) on those LLM-pseudo labels.
      3. Round 1: for the highest-entropy task-model predictions, re-query
         the LLM at a different temperature and adopt the new label.
      4. All round-1 outputs are accepted (FreeAL does not gate on risk —
         this is precisely why it violates α in the paper).
    """
    refresh_frac: float = 0.20

    def _train_task_model(self, X, y, n_classes):
        try:
            import lightgbm as lgb
            clf = lgb.LGBMClassifier(
                n_estimators=200, max_depth=-1, num_leaves=31,
                learning_rate=0.05, objective="multiclass",
                num_class=n_classes, verbose=-1,
            )
            clf.fit(X, y)
            return clf
        except Exception:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=500).fit(X, y)

    def label(self, windows, gold, channel_names, class_names):
        K = len(class_names)
        view0 = ViewConfig("standard", 0.0, "freeal_round0")
        out0 = _parallel_single_view(self.labeler, self.summarizer, view0,
                                      windows, channel_names, class_names)
        preds = np.array([v["label"] for v in out0])
        confs = np.array([v["confidence"] for v in out0])
        # Pre-compute summaries for round 1 (sequential — cheap, no LLM)
        summaries = [self.summarizer.summarize(w, channel_names) for w in windows]

        # Train task model on LLM pseudo-labels
        feats = self.summarizer.extract_features(np.array([w for w in windows]), channel_names)
        if len(np.unique(preds)) >= 2:
            tm = self._train_task_model(feats, preds, K)
            try:
                proba = tm.predict_proba(feats)
            except Exception:
                proba = np.eye(K)[preds]
        else:
            proba = np.eye(K)[preds]
        ent = -np.sum(proba * np.log(proba + 1e-9), axis=1)
        n_refresh = max(1, int(self.refresh_frac * len(windows)))
        refresh_idx = np.argsort(-ent)[:n_refresh]
        # Round 1: re-query LLM at higher temperature for uncertain ones,
        # parallelised over the refresh subset.
        view1 = ViewConfig("standard", 0.7, "freeal_round1")
        self.labeler.views = [view1]
        from concurrent.futures import ThreadPoolExecutor
        nw = int(os.environ.get("CALM_TS_LLM_WORKERS", "8"))
        def _do(i):
            v = self.labeler.label_concurrent(summaries[i], class_names, max_workers=1)[0]
            return i, v
        if nw > 1 and len(refresh_idx) > 1:
            with ThreadPoolExecutor(max_workers=min(nw, len(refresh_idx))) as ex:
                for i, v in ex.map(_do, list(refresh_idx)):
                    preds[i] = int(v["label"])
                    confs[i] = float(v["confidence"])
        else:
            for i in refresh_idx:
                _, v = _do(int(i))
                preds[i] = int(v["label"])
                confs[i] = float(v["confidence"])
        accept = np.ones(len(preds), dtype=bool)
        return preds, confs, accept


@dataclass
class LLMaAA(FullLLM):
    """LLMaAA (Zhang et al. 2023): LLM as Active Annotator with k-NN
    in-context demonstration retrieval.

    The gold set serves as a demonstration pool. For each test window we
    extract its summarizer feature vector, find the top-k nearest gold
    windows by Euclidean distance, and prepend them to the prompt as
    in-context examples. The LLM then labels with retrieved context.

    LLMaAA does not gate on risk, so accept = all-true; the calibrated
    selective gate is precisely the gap CALM-TS fills.
    """
    k_demos: int = 3

    def label(self, windows, gold, channel_names, class_names):
        gold_feats = self.summarizer.extract_features(gold["windows"], channel_names)
        gold_labels = gold["labels"]
        test_feats = self.summarizer.extract_features(np.array(windows), channel_names)
        # whitening to make Euclidean meaningful
        mu = gold_feats.mean(axis=0, keepdims=True)
        sd = gold_feats.std(axis=0, keepdims=True) + 1e-6
        gf = (gold_feats - mu) / sd
        tf = (test_feats - mu) / sd
        view = ViewConfig("standard", 0.0, "llmaaa_main")
        self.labeler.views = [view]

        from concurrent.futures import ThreadPoolExecutor
        nw = int(os.environ.get("CALM_TS_LLM_WORKERS", "8"))
        out: list[dict] = [None] * len(windows)  # type: ignore

        def _do(i_w):
            i, w = i_w
            d = np.sum((gf - tf[i]) ** 2, axis=1)
            nn = np.argsort(d)[: self.k_demos]
            demo_summary = "\n".join(
                f"  example {j+1}: label={class_names[gold_labels[idx]]}"
                for j, idx in enumerate(nn)
            )
            base = self.summarizer.summarize(w, channel_names)
            augmented = (
                "Reference cases (retrieved by k-NN over time-series statistics):\n"
                f"{demo_summary}\n\nNow label the following window:\n{base}"
            )
            v = self.labeler.label_concurrent(augmented, class_names, max_workers=1)[0]
            return i, v

        if nw > 1 and len(windows) > 1:
            with ThreadPoolExecutor(max_workers=min(nw, len(windows))) as ex:
                for i, v in ex.map(_do, [(i, w) for i, w in enumerate(windows)]):
                    out[i] = v
        else:
            for i, w in enumerate(windows):
                _, out[i] = _do((i, w))
        preds = np.array([v["label"] for v in out])
        confs = np.array([v["confidence"] for v in out])
        accept = np.ones(len(preds), dtype=bool)
        return preds, confs, accept


BASELINES = {
    "Full-LLM": FullLLM,
    "Self-Consistency": SelfConsistency,
    "Confidence-Threshold": ConfidenceThreshold,
    "Conformal-LLM": ConformalLLM,
    "Snorkel": Snorkel,
    "BADGE": BADGE,
    "Prompted-WS": PromptedWS,
    "FreeAL": FreeAL,
    "LLMaAA": LLMaAA,
}

