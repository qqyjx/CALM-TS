"""LLM-call response cache for sensitivity sweeps.

Sweep experiments (gating threshold τ, verification budget B, gold-set size)
do not change the underlying LLM input — only the post-hoc gate / verifier
configuration. Re-running the full CALM-TS pipeline for each sweep cell would
re-issue the same gpt-4o-mini calls thousands of times, which is both slow and
wasteful (and was the reason early sweeps used identical 5-seed averages
rounded to 3 decimal places — see integrity_audit_2026-05-18.html).

This cache lets a sweep:
  1. Pre-populate from results/runs/seed*/raw_trace.jsonl.gz so the original
     camera-ready calls are zero-cost.
  2. Append any new (model, temperature, prompt) tuple it sees, so the next
     sweep cell hits the cache instead of the API.

Cache key: SHA-256 of "model|temperature|prompt" (truncated to 16 hex chars).
Storage:   one shared .jsonl.gz under results/runs/llm_cache.jsonl.gz; each
           line is {"k": <hex>, "r": <response_text>}.

Activate by setting the env var CALM_TS_LLM_CACHE=1 (default off so legacy
unit tests still hit the live mock).
"""
from __future__ import annotations
import gzip
import hashlib
import json
import os
import threading
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "results" / "runs" / "llm_cache.jsonl.gz"


def cache_key(model: str, temperature: float, prompt: str) -> str:
    s = f"{model}|{temperature:.4f}|{prompt}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


class _LLMCache:
    """Process-wide singleton cache.

    Thread-safe for read (dict lookup); write is serialized through a lock so
    parallel view workers do not corrupt the on-disk JSONL.
    """

    def __init__(self, path: Path = _DEFAULT_PATH):
        self.path = path
        self._mem: dict[str, str] = {}
        self._lock = threading.Lock()
        self._loaded = False
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def load(self):
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if self.path.exists():
                with gzip.open(self.path, "rt", encoding="utf-8") as f:
                    for line in f:
                        try:
                            d = json.loads(line)
                            self._mem[d["k"]] = d["r"]
                        except (json.JSONDecodeError, KeyError):
                            continue
            self._loaded = True

    def prefill_from_raw_traces(self, seed_dirs: list[Path], model: str = "gpt-4o-mini"):
        """Load every (prompt, response) pair from existing raw_trace.jsonl.gz
        files into the cache. Idempotent — re-running does not duplicate keys.

        Returns (added, present, skipped) counts. New entries are appended to
        the on-disk JSONL atomically per-batch (after the seed_dirs walk).
        """
        self.load()
        added = present = skipped = 0
        new_entries: list[tuple[str, str]] = []
        for sd in seed_dirs:
            trace_path = Path(sd) / "raw_trace.jsonl.gz"
            if not trace_path.exists():
                continue
            with gzip.open(trace_path, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    if d.get("response") is None or d.get("prompt") is None:
                        skipped += 1
                        continue
                    k = cache_key(d.get("model", model),
                                  float(d.get("temperature", 0.0)),
                                  d["prompt"])
                    if k in self._mem:
                        present += 1
                    else:
                        self._mem[k] = d["response"]
                        new_entries.append((k, d["response"]))
                        added += 1
        if new_entries:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(self.path, "at", encoding="utf-8") as f:
                    for k, r in new_entries:
                        f.write(json.dumps({"k": k, "r": r}, ensure_ascii=False) + "\n")
        return added, present, skipped

    def get(self, model: str, temperature: float, prompt: str):
        if not self._loaded:
            self.load()
        k = cache_key(model, temperature, prompt)
        r = self._mem.get(k)
        if r is None:
            self.misses += 1
        else:
            self.hits += 1
        return r

    def put(self, model: str, temperature: float, prompt: str, response: str):
        if response is None:
            return
        k = cache_key(model, temperature, prompt)
        with self._lock:
            if k in self._mem:
                return
            self._mem[k] = response
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(self.path, "at", encoding="utf-8") as f:
                f.write(json.dumps({"k": k, "r": response}, ensure_ascii=False) + "\n")
            self.writes += 1

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes,
                "n_keys": len(self._mem), "path": str(self.path)}


_singleton: _LLMCache | None = None
_singleton_lock = threading.Lock()


def get_cache() -> _LLMCache:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                path = Path(os.environ.get("CALM_TS_LLM_CACHE_PATH", str(_DEFAULT_PATH)))
                _singleton = _LLMCache(path)
                _singleton.load()
    return _singleton


def cache_enabled() -> bool:
    return os.environ.get("CALM_TS_LLM_CACHE", "0") not in ("", "0", "false", "False")
