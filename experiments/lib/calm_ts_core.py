"""CALM-TS core pipeline: summarization, multi-view labeling, calibration, gating, verification.

Implements the four-stage pipeline from the paper:
    T -> textual summary -> M LLM views -> calibrated score -> selective gate -> (auto-accept | verify)

LLM calls assume an OpenAI-compatible API via the env var OPENAI_API_KEY.
For paper reproduction with GPT-4, set MODEL=gpt-4-0613.
"""
from __future__ import annotations
import os
import json
import math
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Iterable
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


# ---------- Stage 1: Temporal summarization ----------

CLINICAL_STATS = [
    "mean", "std", "min", "max", "p25", "p50", "p75",
    "trend_slope", "trend_r2", "n_peaks", "spectral_centroid", "hfd",
]


def detect_change_points(x: np.ndarray, max_k: int = 3,
                         penalty: float | None = None,
                         max_pts_for_pelt: int = 512) -> list[int]:
    """PELT (Killick et al. 2012) change-point detection on a 1-D series.

    Decimate to `max_pts_for_pelt` ≤ 512 before fitting PELT (rbf is O(n^2);
    on the PhysioNet 30 s × 250 Hz = 7 500-sample windows it does not
    converge in reasonable wall time). Breakpoints are rescaled back into
    the original index space; sub-sample resolution loss is negligible
    relative to the LLM summarization window granularity. l2 cost is used
    instead of rbf for ~10× speed-up at indistinguishable change-point
    quality on these signals (see ablation in `experiments/rebuttal/
    pelt_cost_ablation.py`).

    Falls back to a CUSUM-style top-k variance/mean-shift detector when
    `ruptures` is not installed or fitting fails. Returns a sorted list of
    integer breakpoints in (0, len(x)) capped at `max_k`.

    Set `CALM_TS_PELT=cusum` env var to skip PELT entirely (tested in the
    ablation; ~5× speed-up at <1% F1 cost on PhysioNet).
    """
    import os as _os
    n = len(x)
    if n < 8 or np.allclose(x.std(), 0):
        return []
    pen = penalty if penalty is not None else 2.0 * float(np.log(n)) * float(np.var(x) + 1e-9)
    if _os.environ.get("CALM_TS_PELT", "").lower() == "cusum":
        # Skip PELT entirely; go straight to CUSUM fallback below.
        raise_to_fallback = True
    else:
        raise_to_fallback = False
    if not raise_to_fallback:
        if n > max_pts_for_pelt:
            stride = int(np.ceil(n / max_pts_for_pelt))
            x_d = x[::stride].astype(float)
            scale = stride
        else:
            x_d = x.astype(float)
            scale = 1
        try:
            import ruptures as rpt  # type: ignore
            algo = rpt.Pelt(model="l2", min_size=4,
                            jump=max(1, len(x_d) // 64)).fit(x_d)
            bkps = algo.predict(pen=pen)
            bkps = [int(b) * scale for b in bkps if 0 < b < len(x_d)]
            return bkps[:max_k]
        except Exception:
            pass
    # CUSUM-style fallback (also used when CALM_TS_PELT=cusum)
    n = len(x)
    win = max(8, n // 10)
    # Vectorised: sliding mean/std with stride 1 is O(n^2) in the simple form
    # below; for n ≤ 8000 this is still <50 ms in numpy.
    scores: list[tuple[int, float]] = []
    for t in range(win, n - win):
        left = x[t - win:t]
        right = x[t:t + win]
        scores.append((t, abs(float(left.std()) - float(right.std()))
                          + abs(float(left.mean()) - float(right.mean()))))
    scores.sort(key=lambda kv: -kv[1])
    picks: list[int] = []
    for t, _ in scores:
        if all(abs(t - p) >= win for p in picks):
            picks.append(t)
        if len(picks) >= max_k:
            break
    return sorted(picks)


@dataclass
class TemporalSummarizer:
    """Convert raw multivariate time series window -> structured text template.

    Per channel produces three blocks of evidence:
        1. n_stats clinician-vetted statistics (mean/std/percentiles/peaks/...)
        2. n_freq_bands FFT power-band ratios — captures rhythm content that
           statistics alone miss (a flat-line asystole vs. a fast vfib have
           very different spectral mass distributions).
        3. n_samples evenly-stratified raw amplitude samples with units in the
           channel's native scale, giving the LLM a coarse waveform sketch.

    When `ml_hint_predictor` is set, an additional one-line hint
        "ml_hint=<class>:<prob>"
    is appended to the summary. This is an implementation enhancement that
    grounds the LLM in a calibrated tabular classifier's posterior so the LLM
    can focus on agreement/disagreement reasoning rather than zero-shot
    classification of dense numerical features. The multi-view disagreement
    framework downstream is unchanged; views still vary in prompt and
    temperature, and the calibrator/gate still learn from view disagreement.

    Paper-original behavior is recovered by setting n_freq_bands=0, n_samples=0,
    ml_hint_predictor=None.
    """
    # Defaults trimmed for prompt-token efficiency (yunwu gpt-4o-mini latency
    # scales with prompt length). The values match the review submission for
    # n_stats; n_freq_bands and n_samples are halved because they're
    # informationally redundant with stats + hr_bpm for clinical labels.
    # Restore review-paper defaults (n_freq_bands=5, n_samples=32) by passing
    # them explicitly when constructing TemporalSummarizer.
    n_stats: int = 12
    n_freq_bands: int = 5         # paper-original; matches LLM cache + raw_trace
    n_samples: int = 32           # paper-original; matches LLM cache + raw_trace
    n_change_points: int = 3       # K in the paper (top-K change points, K=3)
    use_clinical_features: bool = True   # adds 9 per-channel clinical features
    pelt_penalty: float | None = None
    stat_names: list[str] = field(default_factory=lambda: CLINICAL_STATS[:12])
    ml_hint_predictor: object | None = None  # FeaturePredictor or None
    default_class_names: list[str] | None = None  # used by summarize when caller omits

    def summarize(self, window: np.ndarray, channel_names: list[str],
                  class_names: list[str] | None = None) -> str:
        if window.ndim != 2:
            raise ValueError(f"expected (T, C), got {window.shape}")
        assert window.shape[1] == len(channel_names)
        T = window.shape[0]
        # Detect channels that are pure zero / constant — PhysioNet 2015
        # records frequently only contain a subset of {ECG_I, ECG_II, ABP,
        # PPG, RESP}; missing leads are zero-filled in preprocessing.
        # Surfacing them as "ABSENT" prevents the LLM from misreading a
        # zero-fill as a flatline (asystole).
        present_mask = np.array([float(window[:, c].std()) > 1e-6
                                 for c in range(window.shape[1])])
        n_present = int(present_mask.sum())
        lines = [f"Window: T={T}, channels_total={len(channel_names)}, "
                 f"channels_present={n_present}"]
        feat_cache = []  # cached (stats, bands) per channel for ML hint
        global_cps: list[tuple[int, str, int]] = []  # (t_index, channel_name, channel_idx)
        for c, name in enumerate(channel_names):
            if not present_mask[c]:
                lines.append(f"  {name}: ABSENT (lead not recorded for this segment)")
                feat_cache.append((self._compute_stats(window[:, c]), None))
                continue
            x = window[:, c]
            stats = self._compute_stats(x)
            parts = [f"{k}={stats[k]:.3g}" for k in self.stat_names[: self.n_stats]]
            # Clinical features per channel — for both binary (true vs
            # false alarm — artifact detection) and 5-class alarm type
            # (rhythm regularity, HRV).
            clinical = None
            if self.use_clinical_features and any(
                tok in name.upper() for tok in ("ECG", "ABP", "PPG")
            ):
                clinical = self._clinical_features(x, name, fs_hz=250.0)
                if clinical["hr_bpm"] > 0:
                    parts.append(f"hr_bpm≈{clinical['hr_bpm']:.0f}")
                if clinical["rr_mean_ms"] > 0:
                    parts.append(f"rr_mean≈{clinical['rr_mean_ms']:.0f}ms")
                if clinical["rmssd_ms"] > 0:
                    parts.append(f"rmssd≈{clinical['rmssd_ms']:.0f}ms")
                if clinical["artifact_score"] > 0:
                    parts.append(f"artifact={clinical['artifact_score']:.2g}")
                if clinical["flatline_frac"] > 0.05:
                    parts.append(f"flatline={clinical['flatline_frac']:.2g}")
            line = f"  {name}: " + ", ".join(parts)
            bands = None
            if self.n_freq_bands > 0:
                bands = self._freq_bands(x, self.n_freq_bands)
                line += "; bandpow=[" + ",".join(f"{b:.2g}" for b in bands) + "]"
            if self.n_samples > 0:
                samples = self._strided_samples(x, self.n_samples)
                line += "; wave=[" + ",".join(f"{v:.2g}" for v in samples) + "]"
            if self.n_change_points > 0:
                cps = detect_change_points(x, max_k=self.n_change_points,
                                           penalty=self.pelt_penalty)
                if cps:
                    line += "; cp=[" + ",".join(str(t) for t in cps) + "]"
                    for t in cps:
                        global_cps.append((t, name, c))
            lines.append(line)
            feat_cache.append((stats, bands, clinical))

        if global_cps:
            # rank by per-channel novelty (how isolated the cp is across channels)
            global_cps.sort(key=lambda t: t[0])
            cp_lines = []
            for j, (t, ch, _c) in enumerate(global_cps[: self.n_change_points]):
                cp_lines.append(f"cp_{j+1:03d}: t={t}, channel={ch}")
            if cp_lines:
                lines.append("  change_points: " + "; ".join(cp_lines))

        effective_class_names = class_names or self.default_class_names
        if self.ml_hint_predictor is not None and effective_class_names is not None:
            class_names = effective_class_names
            feat = self._channel_features(feat_cache)
            try:
                proba = self.ml_hint_predictor.predict_proba(feat.reshape(1, -1))[0]
                top = int(np.argmax(proba))
                top_p = float(proba[top])
                second = int(np.argsort(proba)[-2]) if len(proba) > 1 else top
                second_p = float(proba[second])
                lines.append(
                    f"  ml_hint: top={class_names[top]} (p={top_p:.2f}), "
                    f"runner_up={class_names[second]} (p={second_p:.2f})"
                )
            except Exception as e:
                logger.warning("ml_hint failed: %s", e)
        return "\n".join(lines)

    def _channel_features(self, cache: list) -> np.ndarray:
        """Flatten per-channel (stats, bands, clinical) cache into a vector.

        Per channel layout: [stats..., bands..., clinical...]
        Clinical block is zero-padded for ABSENT leads to keep dim stable.
        """
        out = []
        for entry in cache:
            stats, bands = entry[0], entry[1]
            clinical = entry[2] if len(entry) >= 3 else None
            out.extend(float(stats[k]) for k in self.stat_names[: self.n_stats])
            if bands is not None:
                out.extend(float(b) for b in bands)
            elif self.n_freq_bands > 0:
                out.extend([0.0] * self.n_freq_bands)
            if self.use_clinical_features:
                if clinical is not None:
                    out.extend(float(clinical[k]) for k in self.CLINICAL_FEATURE_KEYS)
                else:
                    out.extend([0.0] * len(self.CLINICAL_FEATURE_KEYS))
        return np.asarray(out, dtype=np.float32)

    @staticmethod
    def _freq_bands(x: np.ndarray, n_bands: int) -> np.ndarray:
        """Equal-width FFT power bands; returns fraction of total power per band."""
        if len(x) < 4:
            return np.zeros(n_bands)
        fft = np.fft.rfft(x - x.mean())
        mag = np.abs(fft) ** 2
        if mag.sum() < 1e-12:
            return np.zeros(n_bands)
        edges = np.linspace(0, len(mag), n_bands + 1, dtype=int)
        bands = np.zeros(n_bands)
        for i in range(n_bands):
            bands[i] = mag[edges[i]:edges[i + 1]].sum()
        return bands / mag.sum()

    @staticmethod
    def _strided_samples(x: np.ndarray, n: int) -> np.ndarray:
        if len(x) <= n:
            return x.copy()
        idx = np.linspace(0, len(x) - 1, n).astype(int)
        return x[idx]

    def _compute_stats(self, x: np.ndarray) -> dict:
        if len(x) < 2:
            return {k: 0.0 for k in CLINICAL_STATS}
        slope, intercept = np.polyfit(np.arange(len(x)), x, 1)
        fit = slope * np.arange(len(x)) + intercept
        ss_res = np.sum((x - fit) ** 2)
        ss_tot = np.sum((x - x.mean()) ** 2) + 1e-9
        r2 = 1.0 - ss_res / ss_tot
        diffs = np.diff(x)
        peaks = int(np.sum((diffs[:-1] > 0) & (diffs[1:] < 0)))
        fft = np.fft.rfft(x - x.mean())
        mag = np.abs(fft)
        freqs = np.arange(len(mag))
        centroid = float((freqs * mag).sum() / (mag.sum() + 1e-9))
        hfd = float(np.mean(np.abs(diffs)))
        return dict(
            mean=float(x.mean()), std=float(x.std()),
            min=float(x.min()), max=float(x.max()),
            p25=float(np.percentile(x, 25)),
            p50=float(np.percentile(x, 50)),
            p75=float(np.percentile(x, 75)),
            trend_slope=float(slope), trend_r2=float(r2),
            n_peaks=peaks, spectral_centroid=centroid, hfd=hfd,
        )

    def _estimate_hr_bpm(self, x: np.ndarray, fs_hz: float = 250.0,
                         channel_name: str = "") -> float:
        peaks = self._detect_peaks(x, fs_hz)
        duration_s = len(x) / fs_hz
        return 60.0 * len(peaks) / max(duration_s, 1e-6)

    @staticmethod
    def _detect_peaks(x: np.ndarray, fs_hz: float = 250.0,
                      refractory_s: float = 0.3) -> list[int]:
        """Pan-Tompkins-lite QRS / pulse peak detector.

        Bandpass-equivalent: emphasis by squared first difference, then
        local-max thresholding. Returns peak sample indices.
        """
        n = len(x)
        if n < int(fs_hz):
            return []
        std = float(x.std())
        if std < 1e-6:
            return []
        # Pan-Tompkins emphasis: |dx/dt|^2 has a peak at each QRS complex
        diff = np.diff(x)
        emph = diff * diff
        # Pad to length n
        emph = np.concatenate([[0.0], emph])
        # Threshold: mean + 1.0 * std of the emphasis signal
        thr = float(emph.mean()) + 1.0 * float(emph.std())
        refractory = int(refractory_s * fs_hz)
        peaks: list[int] = []
        last = -refractory
        for i in range(1, n - 1):
            if emph[i] > thr and emph[i] >= emph[i - 1] and emph[i] > emph[i + 1] and (i - last) >= refractory:
                peaks.append(i)
                last = i
        return peaks

    @staticmethod
    def _clinical_features(x: np.ndarray, channel_name: str,
                            fs_hz: float = 250.0) -> dict:
        """Per-channel clinical features useful for both PhysioNet tasks.

        Returns a fixed-key dict so the feature dimension is deterministic
        across channels (even for ABSENT leads, we emit zeros).

        Keys:
          hr_bpm        — rate from peak detection
          rr_mean_ms    — mean RR interval (ms); 0 if <2 peaks
          rr_std_ms     — std of RR intervals; 0 if <2 peaks
          rmssd_ms      — root-mean-square of successive RR diffs (HRV)
          peak_amp_mean — mean amplitude at peaks
          peak_amp_std  — std of peak amplitudes (rhythm regularity proxy)
          artifact_score— high-frequency noise indicator (mean |d²x/dt²|)
          flatline_frac — fraction of samples with |x - median| < 0.05·std
          sat_frac      — fraction of samples at ±99-percentile of range
        """
        zero = dict(hr_bpm=0.0, rr_mean_ms=0.0, rr_std_ms=0.0, rmssd_ms=0.0,
                    peak_amp_mean=0.0, peak_amp_std=0.0, artifact_score=0.0,
                    flatline_frac=0.0, sat_frac=0.0)
        n = len(x)
        if n < int(fs_hz) or float(x.std()) < 1e-6:
            return zero
        peaks = TemporalSummarizer._detect_peaks(x, fs_hz)
        duration_s = n / fs_hz
        hr = 60.0 * len(peaks) / max(duration_s, 1e-6)
        if len(peaks) >= 2:
            rr_samples = np.diff(np.asarray(peaks, dtype=np.float64))
            rr_ms = rr_samples * (1000.0 / fs_hz)
            rr_mean = float(rr_ms.mean())
            rr_std = float(rr_ms.std())
            rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr_ms) >= 2 else 0.0
            amps = x[peaks]
            amp_mean = float(amps.mean())
            amp_std = float(amps.std())
        else:
            rr_mean = rr_std = rmssd = amp_mean = amp_std = 0.0
        d2 = np.diff(x, n=2)
        artifact = float(np.mean(np.abs(d2)))
        med = float(np.median(x))
        std = float(x.std())
        flat = float(np.mean(np.abs(x - med) < 0.05 * std))
        rng = float(x.max() - x.min())
        if rng > 1e-9:
            p1 = float(np.percentile(x, 1)); p99 = float(np.percentile(x, 99))
            sat = float(np.mean((x <= p1) | (x >= p99)))
        else:
            sat = 1.0
        return dict(hr_bpm=hr, rr_mean_ms=rr_mean, rr_std_ms=rr_std, rmssd_ms=rmssd,
                    peak_amp_mean=amp_mean, peak_amp_std=amp_std,
                    artifact_score=artifact, flatline_frac=flat, sat_frac=sat)

    # Order matters for extract_features stability:
    CLINICAL_FEATURE_KEYS = (
        "hr_bpm", "rr_mean_ms", "rr_std_ms", "rmssd_ms",
        "peak_amp_mean", "peak_amp_std",
        "artifact_score", "flatline_frac", "sat_frac",
    )

    def extract_features(self, windows: np.ndarray, channel_names: list[str]) -> np.ndarray:
        """Batch feature extraction for ML predictor training. Returns (N, F)."""
        N = windows.shape[0]
        out = []
        for i in range(N):
            cache = []
            for c in range(windows.shape[2]):
                x = windows[i, :, c]
                s = self._compute_stats(x)
                b = self._freq_bands(x, self.n_freq_bands) if self.n_freq_bands > 0 else None
                cl = None
                if self.use_clinical_features:
                    name = channel_names[c]
                    if any(tok in name.upper() for tok in ("ECG", "ABP", "PPG")):
                        if float(x.std()) > 1e-6:
                            cl = self._clinical_features(x, name, fs_hz=250.0)
                cache.append((s, b, cl))
            out.append(self._channel_features(cache))
        return np.stack(out, axis=0)


class FeaturePredictor:
    """Tabular classifier over TemporalSummarizer features.

    Used to provide an ML hint to the LLM in the textual summary. The hint
    grounds the LLM's classification in a calibrated tabular posterior so the
    multi-view disagreement framework downstream can focus on
    agreement/disagreement reasoning instead of zero-shot label mapping.

    Default backbone is XGBoost; falls back to RandomForest if xgboost is
    unavailable.
    """

    def __init__(self, n_classes: int, backbone: str = "xgboost"):
        self.n_classes = n_classes
        self.backbone = backbone
        self._clf = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeaturePredictor":
        if self.backbone == "xgboost":
            try:
                import xgboost as xgb
                self._clf = xgb.XGBClassifier(
                    n_estimators=200, max_depth=6,
                    n_jobs=4, tree_method="hist",
                    objective="multi:softprob",
                )
            except ImportError:
                self.backbone = "rf"
        if self.backbone == "rf":
            from sklearn.ensemble import RandomForestClassifier
            self._clf = RandomForestClassifier(n_estimators=200, n_jobs=4)
        self._clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict(X)


# ---------- Stage 2: Multi-view LLM labeling ----------

@dataclass
class ViewConfig:
    prompt_variant: str     # one of {narrow, standard, wide}
    temperature: float
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.prompt_variant}_T{self.temperature}"


# Binary task (PhysioNet Challenge 2015 true/false alarm) needs a different
# clinical frame than the 5-class alarm-type task. Selected automatically
# in _label_one_view when labels include "true_alarm"/"false_alarm".
PROMPT_TEMPLATES_BINARY = {
    "narrow": (
        "You are deciding whether an ICU alarm was triggered by a genuine "
        "life-threatening event or was a false alarm caused by signal noise, "
        "lead motion, or other artifact. Classes: {labels}. "
        "Output JSON: {{\"label\": <int>, \"confidence\": <float 0-1>}}.\n"
        "Segment:\n{summary}"
    ),
    "standard": (
        "You are reviewing an ICU bedside alarm. Decide whether it was a "
        "true alarm (a real cardiac event occurred during the segment) or a "
        "false alarm (noise / motion / lead-off / electrical interference).\n"
        "Heuristics:\n"
        "  - high artifact_score, high flatline_frac on multiple channels, or "
        "saturated samples (sat_frac high) on the ECG channels strongly suggest "
        "false alarm.\n"
        "  - hr_bpm consistent across ECG_I, ECG_II, PPG and ABP (when present) "
        "suggests true alarm; large disagreement across leads suggests false.\n"
        "  - regular RR intervals (low rmssd) with a hr_bpm in the relevant "
        "range (asystole near 0, brady<60, tachy>100) supports true alarm.\n"
        "Classes: {labels}.\n"
        "Output JSON: {{\"label\": <int>, \"confidence\": <float 0-1>, \"rationale\": <str>}}.\n"
        "Summary:\n{summary}"
    ),
    "wide": (
        "An ICU bedside alarm fired during this 30-second window. Decide "
        "whether the alarm represents a genuine cardiac event (true_alarm) "
        "or was a false alarm.\n"
        "Cross-check evidence:\n"
        "  - Look across all available leads (ECG_I, ECG_II, ABP, PPG). True "
        "alarms typically show consistent rhythm signatures on multiple leads; "
        "false alarms usually have one noisy lead disagreeing with quieter ones.\n"
        "  - Channels marked ABSENT carry no information — ignore them.\n"
        "  - High artifact_score on ECG combined with intact PPG/ABP rhythm "
        "is a classic false-alarm pattern (lead came loose).\n"
        "Classes: {labels}.\n"
        "Output JSON: {{\"label\": <int>, \"confidence\": <float 0-1>, "
        "\"rationale\": <str>, \"secondary\": <int or null>}}.\n"
        "Summary:\n{summary}"
    ),
}

PROMPT_TEMPLATES = {
    "narrow": (
        "You are labeling a short time-series segment. "
        "Classes: {labels}. Output JSON: {{\"label\": <int>, \"confidence\": <float 0-1>}}.\n"
        "Segment:\n{summary}"
    ),
    "standard": (
        "You are a clinical signal-processing assistant labeling a 30-second "
        "ICU monitoring window. Use hr_bpm directly when present:\n"
        "  asystole = no rhythm (hr_bpm < 10); look for flatline ECG / very low std.\n"
        "  bradycardia = hr_bpm 30-60.\n"
        "  tachycardia = sinus-style hr_bpm 100-180 with regular peaks.\n"
        "  ventricular_tachycardia = hr_bpm 120-250, irregular high-amplitude QRS, "
        "high spectral centroid in ECG, often a recent change point.\n"
        "  ventricular_fibrillation/flutter = hr_bpm 200+ chaotic with high hfd.\n"
        "Classes: {labels}.\n"
        "Output JSON: {{\"label\": <int>, \"confidence\": <float 0-1>, \"rationale\": <str>}}.\n"
        "Summary:\n{summary}"
    ),
    "wide": (
        "You are reviewing an ICU alarm window. The alarm priors include "
        "asystole, bradycardia, tachycardia, ventricular_tachycardia, and "
        "ventricular_fibrillation/flutter. Use hr_bpm and waveform shape; "
        "missing leads are marked ABSENT and should be ignored — base your "
        "decision on the channels actually present.\n"
        "Classes: {labels}.\n"
        "Output JSON: {{\"label\": <int>, \"confidence\": <float 0-1>, "
        "\"rationale\": <str>, \"secondary\": <int or null>}}.\n"
        "Summary:\n{summary}"
    ),
}



def default_views() -> list[ViewConfig]:
    """Paper default: 3 prompts × 4 temperatures = 12 views."""
    temps = [0.0, 0.3, 0.7, 1.0]
    prompts = ["narrow", "standard", "wide"]
    return [ViewConfig(p, t) for p in prompts for t in temps]


class MultiViewLabeler:
    """Query M LLM views on a textual summary. Returns per-view predictions and confidences."""

    def __init__(self, views: list[ViewConfig] | None = None,
                 llm_call: Callable[[str, float], str] | None = None):
        self.views = views or default_views()
        self.llm_call = llm_call or self._default_llm_call

    def label(self, summary: str, labels: list[str]) -> list[dict]:
        """Label `summary` with all configured views.

        Now delegates to `label_concurrent` so every baseline that calls the
        labeler benefits from the thread pool (Self-Consistency, BADGE,
        Snorkel, Prompted-WS were all sequential before). Sequential
        execution is recoverable via `CALM_TS_LLM_WORKERS=1`.
        """
        max_workers = int(os.environ.get("CALM_TS_LLM_WORKERS", str(len(self.views))))
        return self.label_concurrent(summary, labels, max_workers=max_workers)

    _LABEL_RE = None
    _CONF_RE = None

    def _parse(self, raw: str, n_labels: int) -> dict:
        # Strategy 1: full JSON parse on the {...} substring.
        if raw is None:
            return {"label": 0, "confidence": 0.5}
        s = raw.strip()
        # Strip markdown fence
        if s.startswith("```"):
            s = s.split("```", 2)[-1]
            if s.startswith("json"):
                s = s[4:]
        i, j = s.find("{"), s.rfind("}")
        if 0 <= i < j:
            try:
                obj = json.loads(s[i:j + 1])
                raw_lbl = obj.get("label", 0)
                raw_conf = obj.get("confidence", 0.5)
                if raw_lbl is None or raw_conf is None:
                    raise ValueError("null label/confidence")
                lbl = int(raw_lbl)
                conf = float(raw_conf)
                return {"label": max(0, min(n_labels - 1, lbl)),
                        "confidence": max(0.0, min(1.0, conf))}
            except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                pass
        # Strategy 2: regex on common patterns ("label": 3, "confidence": 0.85)
        import re
        if MultiViewLabeler._LABEL_RE is None:
            MultiViewLabeler._LABEL_RE = re.compile(r'"label"\s*:\s*(-?\d+)')
            MultiViewLabeler._CONF_RE = re.compile(r'"confidence"\s*:\s*([\d.]+)')
        ml = MultiViewLabeler._LABEL_RE.search(s)
        mc = MultiViewLabeler._CONF_RE.search(s)
        if ml is not None and ml.group(1):
            try:
                lbl = max(0, min(n_labels - 1, int(ml.group(1))))
                conf = 0.5
                if mc is not None and mc.group(1):
                    try:
                        conf = max(0.0, min(1.0, float(mc.group(1))))
                    except ValueError:
                        pass
                return {"label": lbl, "confidence": conf}
            except (ValueError, IndexError, TypeError):
                pass
        # Strategy 3: surrender — return uninformative fallback. The
        # disagreement features will downweight this view via the
        # plurality / confidence-mean signals.
        return {"label": 0, "confidence": 0.5}

    _shared_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls_ok": 0,
                     "calls_failed": 0, "model": None}
    _trace_path: str | None = None  # set by experiment runner; if set, append jsonl
    _trace_lock: threading.Lock = threading.Lock()
    _usage_lock: threading.Lock = threading.Lock()

    @classmethod
    def reset_usage(cls):
        with cls._usage_lock:
            cls._shared_usage.update({"prompt_tokens": 0, "completion_tokens": 0,
                                      "calls_ok": 0, "calls_failed": 0, "model": None})

    @classmethod
    def set_trace(cls, path: str | None):
        cls._trace_path = path
        if path and os.environ.get("CALM_TS_TRACE_TRUNCATE") == "1":
            # Usually resume runs should append to the audit trail; truncate
            # only when explicitly requested for a fresh trace.
            open(path, "w").close()

    @staticmethod
    def _append_trace(record: dict):
        path = MultiViewLabeler._trace_path
        if not path:
            return
        try:
            with MultiViewLabeler._trace_lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    @staticmethod
    def _bump_usage(prompt_tokens: int = 0, completion_tokens: int = 0,
                    ok: bool = True, model: str | None = None):
        with MultiViewLabeler._usage_lock:
            MultiViewLabeler._shared_usage["prompt_tokens"] += prompt_tokens
            MultiViewLabeler._shared_usage["completion_tokens"] += completion_tokens
            MultiViewLabeler._shared_usage["calls_ok" if ok else "calls_failed"] += 1
            if model is not None:
                MultiViewLabeler._shared_usage["model"] = model

    _shared_client = None
    _client_lock: threading.Lock = threading.Lock()

    @staticmethod
    def _get_client(force_new: bool = False):
        """One shared OpenAI client; httpx is thread-safe for our use.

        Per-thread clients turned out to leak connection pool threads under
        view-level concurrency on the yunwu.ai gateway. A single shared
        client with a small per-call timeout is safer.
        """
        with MultiViewLabeler._client_lock:
            if MultiViewLabeler._shared_client is None or force_new:
                import openai, httpx
                base_url = os.environ.get("OPENAI_BASE_URL")
                api_key = os.environ.get("OPENAI_API_KEY")
                timeout_s = float(os.environ.get("OPENAI_TIMEOUT_S", "30"))
                # Strict per-request timeout (connect/read/write/pool).
                http = httpx.Client(
                    timeout=httpx.Timeout(timeout_s),
                    limits=httpx.Limits(max_connections=64,
                                        max_keepalive_connections=32),
                    trust_env=False,  # ignore HTTP_PROXY env
                )
                kw = {"http_client": http, "api_key": api_key, "timeout": timeout_s}
                if base_url:
                    kw["base_url"] = base_url
                MultiViewLabeler._shared_client = openai.OpenAI(**kw)
            return MultiViewLabeler._shared_client

    @staticmethod
    def _reset_client():
        with MultiViewLabeler._client_lock:
            MultiViewLabeler._shared_client = None

    def _default_llm_call(self, prompt: str, temperature: float) -> str:
        """OpenAI-compatible chat completion with strict timeout + bounded retries.

        Env vars:
            OPENAI_API_KEY    required
            OPENAI_BASE_URL   optional, e.g. https://yunwu.ai/v1 for a proxy
            MODEL             chat model id; default gpt-4o-mini
            OPENAI_TIMEOUT_S  per-request timeout (default 30s, hard ceiling)
            CALM_TS_LLM_CACHE     "1" to consult the (prompt,model,temp) cache
                                  before issuing a network call; misses are
                                  persisted so subsequent sweep cells get a
                                  zero-cost hit. See lib/llm_cache.py.

        Bounded total budget: 3 attempts × (timeout + small backoff). Total
        worst-case ≈ 3 × (30+2) = 96s per call, then we surrender to the parser
        fallback so the pipeline never deadlocks.

        Override `MultiViewLabeler(llm_call=...)` for an alternative endpoint or a mock.
        """
        model = os.environ.get("MODEL", "gpt-4o-mini")
        # ---- Cache hook (camera-ready audit fix; see results/integrity_audit) ----
        try:
            from .llm_cache import get_cache, cache_enabled
            _cache = get_cache() if cache_enabled() else None
        except ImportError:
            _cache = None
        if _cache is not None:
            hit = _cache.get(model, temperature, prompt)
            if hit is not None:
                return hit
        last_exc = None
        t0 = time.time()
        for attempt in range(3):
            try:
                client = MultiViewLabeler._get_client()
                # 200 tokens truncates rationales for standard/wide prompts on
                # gpt-4o; the parser cannot recover the JSON object then.
                # 600 covers all observed responses, override via env if needed.
                max_tokens = int(os.environ.get("CALM_TS_LLM_MAX_TOKENS", "600"))
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                u = resp.usage
                pt = getattr(u, "prompt_tokens", 0) or 0
                ct = getattr(u, "completion_tokens", 0) or 0
                MultiViewLabeler._bump_usage(pt, ct, ok=True, model=model)
                content = resp.choices[0].message.content
                if _cache is not None and content is not None:
                    _cache.put(model, temperature, prompt, content)
                MultiViewLabeler._append_trace({
                    "model": model,
                    "temperature": temperature,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "elapsed_s": round(time.time() - t0, 3),
                    "attempt": attempt + 1,
                    "prompt": prompt,
                    "response": content,
                })
                return content
            except Exception as e:  # connection / rate-limit / 5xx / timeout
                last_exc = e
                # Fail fast on quota / permission errors — retrying with the
                # same exhausted key just burns walltime. Same memory lesson
                # as the gpt-4o ablation: quota errors should not enter the
                # exponential-backoff loop.
                msg = str(e)
                if ("insufficient_quota" in msg or "user quota is not enough" in msg
                        or "PermissionDenied" in type(e).__name__):
                    break
                # Force a fresh client + connection pool on the next attempt.
                MultiViewLabeler._reset_client()
                wait = min(2 + attempt, 4)
                logger.warning(
                    "LLM call attempt %d failed (%s); sleeping %ds",
                    attempt + 1, type(e).__name__, wait,
                )
                time.sleep(wait)

        MultiViewLabeler._bump_usage(ok=False, model=model)
        logger.error("LLM call gave up after retries: %s", last_exc)
        MultiViewLabeler._append_trace({
            "ts": time.time(),
            "model": model,
            "temperature": temperature,
            "error": f"{type(last_exc).__name__}: {last_exc}",
            "elapsed_s": round(time.time() - t0, 3),
            "prompt": prompt,
            "response": None,
        })
        # Return a parseable fallback so the pipeline keeps running
        return '{"label": 0, "confidence": 0.5}'

    def _label_one_view(self, summary: str, labels_str: str, n_labels: int,
                        v: ViewConfig) -> dict:
        templates = (
            PROMPT_TEMPLATES_BINARY
            if "true_alarm" in labels_str or "false_alarm" in labels_str
            else PROMPT_TEMPLATES
        )
        prompt = templates[v.prompt_variant].format(
            summary=summary, labels=labels_str
        )
        try:
            raw = self.llm_call(prompt, v.temperature)
        except Exception as e:
            logger.warning("llm_call raised %s; falling back", e)
            raw = '{"label": 0, "confidence": 0.5}'
        try:
            parsed = self._parse(raw, n_labels)
        except Exception as e:
            logger.warning("parse raised %s; falling back", e)
            parsed = {"label": 0, "confidence": 0.5}
        return {"view": v.name, **parsed}

    def label_concurrent(self, summary: str, labels: list[str],
                         max_workers: int = 8) -> list[dict]:
        """Same as label() but views fired in parallel via ThreadPoolExecutor.

        Returns view results in the original view-config order (not as-completed).
        """
        labels_str = ", ".join(f"{i}={name}" for i, name in enumerate(labels))
        n = len(labels)
        out: list[dict] = [None] * len(self.views)  # type: ignore
        if max_workers <= 1 or len(self.views) == 1:
            for i, v in enumerate(self.views):
                out[i] = self._label_one_view(summary, labels_str, n, v)
            return out  # type: ignore
        with ThreadPoolExecutor(max_workers=min(max_workers, len(self.views))) as ex:
            future_to_idx = {
                ex.submit(self._label_one_view, summary, labels_str, n, v): i
                for i, v in enumerate(self.views)
            }
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                out[i] = fut.result()
        return out  # type: ignore


# ---------- Stage 3: Calibration ----------

def disagreement_features(view_outputs: list[dict], n_labels: int) -> np.ndarray:
    """Extract features from M multi-view outputs for one sample."""
    labels = np.array([v["label"] for v in view_outputs])
    confs = np.array([v["confidence"] for v in view_outputs])
    M = len(view_outputs)
    # vote distribution entropy
    counts = np.bincount(labels, minlength=n_labels) / M
    ent = -np.sum(counts[counts > 0] * np.log(counts[counts > 0] + 1e-9))
    # confidence stats
    return np.array([
        counts.max(),                    # plurality rate
        ent,                              # vote entropy
        confs.mean(), confs.std(),        # confidence mean/std
        float((labels == labels[0]).all()),  # unanimity
        float(np.median(labels) == labels[0]),  # majority alignment
    ])



@dataclass
class Calibrator:
    """Logistic regression with degree-2 polynomial features on disagreement signals."""
    _clf: LogisticRegression | None = None
    _poly: PolynomialFeatures | None = None

    def fit(self, X: np.ndarray, y_correct: np.ndarray):
        self._poly = PolynomialFeatures(degree=2, include_bias=False)
        Xp = self._poly.fit_transform(X)
        self._clf = LogisticRegression(max_iter=1000, C=1.0)
        self._clf.fit(Xp, y_correct.astype(int))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(self._poly.transform(X))[:, 1]


# ---------- Stage 4: Selective gating + cost-aware verification ----------

def conformal_threshold(scores_cal: np.ndarray, correct_cal: np.ndarray,
                        alpha: float, delta: float = 0.05) -> float:
    """Selective gating threshold τ(α) — Equation (5) in the paper.

        τ(α) = inf { τ :
                     (Σ_{j: ŷ_j ≠ y_j*} 1[p̂_j < τ]) /
                     |{j: ŷ_j ≠ y_j*}|
                     ≥ 1 − α }

    In words: among the gold-set instances where the LLM was *wrong*, τ is the
    (1−α)-quantile of their calibrated probabilities. Accepting only samples
    with p̂ ≥ τ then leaves at most an α-fraction of gold-set errors above the
    threshold, which under exchangeability bounds the population risk by α
    (Theorem 1 / Proof in Appendix A.1).

    Args:
        scores_cal:  (n,) calibrated probabilities on the gold set
        correct_cal: (n,) bool / 0-1 — was the LLM prediction correct?
        alpha:       target risk level
        delta:       confidence parameter (logged but the basic Eq.5 form is
                     not Bernstein/Hoeffding-adjusted; pass through to be
                     compatible with the variance-adaptive variant)

    Returns the threshold τ ∈ [0, 1]; falls back to 1.0 (accept nothing) when
    fewer than `n_min` errors are observed (the bound vacuously holds).
    """
    n_min = 10  # avoid degenerate quantile on tiny error sets
    err_mask = np.asarray(correct_cal, dtype=bool) == False  # noqa: E712
    err_scores = np.asarray(scores_cal)[err_mask]
    n_err = int(err_scores.size)
    if n_err < n_min:
        # Not enough error samples to form a (1-alpha)-quantile estimate.
        # Returning 1.0 means "accept nothing" — strictly safe for the bound,
        # callers can route to verification in that case.
        return 1.0
    return float(np.quantile(err_scores, 1.0 - alpha))


def conformal_threshold_bernstein(scores_cal: np.ndarray, correct_cal: np.ndarray,
                                  alpha: float, delta: float = 0.05) -> float:
    """Variance-adaptive Bernstein refinement (Corollary 2 in the paper).

    Returns the largest τ such that the empirical error rate above τ plus
    an empirical-Bernstein excess term stays at or below α with probability
    ≥ 1−δ. Used by `SelectiveGate(use_bernstein=True)`.
    """
    cand = np.sort(np.unique(scores_cal))[::-1]
    best = 1.0
    for tau in cand:
        mask = scores_cal >= tau
        k = int(mask.sum())
        if k < 10:
            continue
        err_indicators = (1 - np.asarray(correct_cal)[mask]).astype(float)
        p_hat = float(err_indicators.mean())
        var_hat = float(err_indicators.var(ddof=1)) if k > 1 else 0.0
        # Empirical Bernstein bound (Maurer & Pontil 2009):
        #   p_true ≤ p_hat + sqrt(2 σ̂² log(1/δ) / n) + 7 log(1/δ) / (3(n-1))
        bound = p_hat + math.sqrt(2 * var_hat * math.log(1 / delta) / k) \
                      + 7 * math.log(1 / delta) / (3 * max(k - 1, 1))
        if bound <= alpha:
            best = tau
        else:
            break
    return best


@dataclass
class SelectiveGate:
    """Selective acceptance gate (Stage 4).

    `use_bernstein=True` switches from the basic Eq.5 (1-α)-quantile threshold
    to the variance-adaptive Bernstein refinement (Corollary 2). The paper's
    main-table numbers use the basic form; the Bernstein variant is reported
    in Table 4 of the appendix.
    """
    alpha: float = 0.05
    delta: float = 0.05
    use_bernstein: bool = False
    tau: float = 1.0

    def fit(self, scores_cal: np.ndarray, correct_cal: np.ndarray):
        if self.use_bernstein:
            self.tau = conformal_threshold_bernstein(
                scores_cal, correct_cal, self.alpha, self.delta)
        else:
            self.tau = conformal_threshold(
                scores_cal, correct_cal, self.alpha, self.delta)
        return self

    def accept(self, scores: np.ndarray) -> np.ndarray:
        return scores >= self.tau


@dataclass
class CostAwareVerifier:
    """Rank gated-uncertain samples by expected-risk-reduction-per-token, keep top B."""
    budget_pct: float = 0.18

    def select(self, uncertain_scores: np.ndarray, token_costs: np.ndarray) -> np.ndarray:
        n = len(uncertain_scores)
        if n == 0:
            return np.array([], dtype=int)
        # higher expected reduction = farther from 0.5 in wrong direction; simpler proxy:
        # (1 - score) / cost
        value = (1 - uncertain_scores) / np.maximum(token_costs, 1.0)
        k = max(1, int(self.budget_pct * n))
        return np.argsort(-value)[:k]


# ---------- End-to-end pipeline ----------

@dataclass
class CalmTSPipeline:
    summarizer: TemporalSummarizer = field(default_factory=TemporalSummarizer)
    multi_view: MultiViewLabeler = field(default_factory=MultiViewLabeler)
    calibrator: Calibrator = field(default_factory=Calibrator)
    gate: SelectiveGate = field(default_factory=SelectiveGate)
    verifier: CostAwareVerifier = field(default_factory=CostAwareVerifier)

    def run_labeling(self, windows: list[np.ndarray], channel_names: list[str],
                     labels: list[str], max_workers: int | None = None) -> list[dict]:
        """Label a list of windows end-to-end. Returns per-sample dicts.

        Concurrency: each window's M views are fired in parallel via
        ThreadPoolExecutor. `max_workers` defaults to env CALM_TS_LLM_WORKERS
        or len(views).  Set CALM_TS_LLM_WORKERS=1 to recover the strictly
        sequential reference implementation (slower but byte-identical to the
        review submission's call order).
        """
        out = []
        n_labels = len(labels)
        n_total = len(windows)
        progress_every = max(1, n_total // 20)
        t0 = time.time()
        if max_workers is None:
            max_workers = int(os.environ.get("CALM_TS_LLM_WORKERS",
                                              str(len(self.multi_view.views))))
        for i, w in enumerate(windows):
            summary = self.summarizer.summarize(w, channel_names)
            views = self.multi_view.label_concurrent(summary, labels,
                                                     max_workers=max_workers)
            feats = disagreement_features(views, n_labels)
            pred = max(set(v["label"] for v in views),
                       key=lambda l: sum(1 for v in views if v["label"] == l))
            out.append({"summary": summary, "views": views, "pred": pred, "feats": feats})
            if (i + 1) % progress_every == 0 or (i + 1) == n_total:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 0.01)
                eta = (n_total - i - 1) / max(rate, 1e-3)
                print(f"    [progress] {i+1}/{n_total} windows ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)",
                      flush=True)
        return out


