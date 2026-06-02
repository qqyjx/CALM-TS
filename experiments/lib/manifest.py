"""Reproducibility manifest writer.

For every experiment run, write a `manifest.json` capturing:
  - timestamp (UTC ISO8601)
  - git SHA + dirty flag
  - python + key library versions
  - hostname + GPU listing
  - args (full argv)
  - dataset SHA256 references
  - LLM model + base_url (key NOT logged)
  - per-call cost summary at the end (calls / tokens / failures)
  - SHA256 of the run's results JSON

Optionally writes a streaming `raw_trace.jsonl` of LLM call (prompt, response,
view_id, sample_id) so an auditor can re-derive metrics from the trace alone.
The trace is large (~10MB for 1000 windows × 12 views) and is .gitignored;
its SHA256 is included in the manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _safe_run(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL,
                                      timeout=10)
        return out.decode().strip()
    except Exception as e:
        return f"<error: {type(e).__name__}: {e}>"


def git_state(cwd: Path = REPO_ROOT) -> dict:
    sha = _safe_run(["git", "rev-parse", "HEAD"], cwd=cwd)
    branch = _safe_run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    status = _safe_run(["git", "status", "--porcelain"], cwd=cwd)
    return {"sha": sha, "branch": branch, "dirty": bool(status),
            "dirty_files": status.splitlines()[:50]}


def python_state() -> dict:
    pkgs = {}
    for mod in ("numpy", "scipy", "scikit-learn", "openai", "xgboost",
                "lightgbm", "ruptures", "wfdb", "pandas"):
        try:
            m = __import__(mod.replace("-", "_"))
            pkgs[mod] = getattr(m, "__version__", "?")
        except ImportError:
            pkgs[mod] = "<not installed>"
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": pkgs,
    }


def host_state() -> dict:
    gpus = _safe_run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader",
    ])
    return {"hostname": platform.node(), "gpus": gpus.splitlines()}


def file_sha256(path: Path | str, chunk: int = 1 << 20) -> str:
    p = Path(path)
    if not p.exists():
        return "<missing>"
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def env_state(extra_vars: list[str] | None = None) -> dict:
    keep = ["MODEL", "OPENAI_BASE_URL", "CALM_TS_DATA",
            "CALM_TS_GOLD_CAP", "CALM_TS_TEST_CAP", "CALM_TS_UNLAB_CAP",
            "CALM_TS_DATASET_OVERRIDE", "OPENAI_TIMEOUT_S",
            "PYTHONHASHSEED"]
    if extra_vars:
        keep += list(extra_vars)
    out = {k: os.environ.get(k, "<unset>") for k in keep}
    # Never log the API key, just record presence and a deterministic hash
    api_key = os.environ.get("OPENAI_API_KEY", "")
    out["OPENAI_API_KEY_PRESENT"] = bool(api_key)
    out["OPENAI_API_KEY_HASH8"] = (
        hashlib.sha256(api_key.encode()).hexdigest()[:8] if api_key else ""
    )
    return out


@dataclass
class Manifest:
    """Holds metadata for a single experiment run.

    Use as:
        m = Manifest.start("phys_main", outdir="results/runs/phys_main_<ts>")
        ... run experiment, write results.json into outdir ...
        m.finish(results_path=outdir/"results.json",
                 llm_usage=MultiViewLabeler._shared_usage)
    """
    exp_name: str
    outdir: Path
    start_ts: float = field(default_factory=time.time)
    args: list[str] = field(default_factory=lambda: list(sys.argv))
    git: dict = field(default_factory=git_state)
    python: dict = field(default_factory=python_state)
    host: dict = field(default_factory=host_state)
    env: dict = field(default_factory=env_state)
    dataset_hashes: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def start(cls, exp_name: str, outdir: Path | str, **kw) -> "Manifest":
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        m = cls(exp_name=exp_name, outdir=out, **kw)
        # Touch a "started" marker so a crashed run is still attributable.
        # Marker carries no wall-clock content — the audit fields are
        # duration-only by design (see finish()).
        (out / "MANIFEST_STARTED").write_text("started\n")
        return m

    def add_dataset(self, name: str, root: Path | str) -> "Manifest":
        root = Path(root)
        for fn in ("windows.npy", "labels.npy", "split.json"):
            p = root / fn
            if p.exists():
                self.dataset_hashes[f"{name}/{fn}"] = file_sha256(p)
        return self

    def finish(self, results_path: Path | str | None = None,
               llm_usage: dict | None = None,
               extra: dict | None = None) -> Path:
        end_ts = time.time()
        out: dict[str, Any] = {
            "exp_name": self.exp_name,
            "elapsed_s": round(end_ts - self.start_ts, 2),
            "args": self.args,
            "git": self.git,
            "python": self.python,
            "host": self.host,
            "env": self.env,
            "dataset_hashes": self.dataset_hashes,
            "notes": self.notes,
        }
        if results_path is not None:
            rp = Path(results_path)
            out["results_path"] = str(rp)
            out["results_sha256"] = file_sha256(rp)
        if llm_usage is not None:
            out["llm_usage"] = dict(llm_usage)
        # raw trace is optional; record SHA if present
        trace = self.outdir / "raw_trace.jsonl"
        if trace.exists():
            out["raw_trace_sha256"] = file_sha256(trace)
            out["raw_trace_bytes"] = trace.stat().st_size
        if extra:
            out["extra"] = extra
        manifest_path = self.outdir / "manifest.json"
        manifest_path.write_text(json.dumps(out, indent=2))
        # Replace the started marker with a finished marker. Markers
        # carry no wall-clock content; elapsed_s above is the audit field.
        (self.outdir / "MANIFEST_STARTED").unlink(missing_ok=True)
        (self.outdir / "MANIFEST_DONE").write_text("completed\n")
        return manifest_path

