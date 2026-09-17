"""Immutable, complete ten-seed releases and a fixed quarterly retraining plan."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from hyfe.paper_rules import RULE_VERSION

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "work/paper_trade"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with tmp.open("w") as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_manifest(manifest, check_weights=False):
    import numpy as np
    if manifest.get("rule_version") != RULE_VERSION or len(manifest.get("models", [])) != 10:
        raise ValueError("a complete ten-seed release is required")
    seen = set()
    feature_schema = None
    for item in manifest["models"]:
        stem = Path(item["stem"])
        meta = json.loads(stem.with_suffix(".json").read_text())
        a = meta["args"]
        expected = {"model": "i1", "render": "heatf", "res": "4h", "W": 60, "H": 84, "label": "relbin"}
        if any(a.get(k) != v for k, v in expected.items()):
            raise ValueError("incompatible CNN release")
        if a.get("img_h", 96) != 96 or a.get("px", 3) != 3 or a.get("channels", "gray") != "gray" or a.get("teacher", ""):
            raise ValueError("incompatible image geometry/teacher")
        if feature_schema is None:
            feature_schema = meta["feat_cols"]
        if meta["feat_cols"] != feature_schema:
            raise ValueError("mixed feature schemas")
        seed = a["seed"]
        if seed in seen or seed != item["seed"]:
            raise ValueError("duplicate/mismatched seed")
        seen.add(seed)
        if len(meta["feat_cols"]) != 37 or not np.isfinite(meta["feat_mu"]).all() or not np.isfinite(meta["feat_sd"]).all() or min(meta["feat_sd"]) <= 0:
            raise ValueError("invalid feature normalization")
        for suffix in (".pt", ".json"):
            if sha(stem.with_suffix(suffix)) != item["sha256"][suffix]:
                raise ValueError("model release was modified")
        if check_weights:
            from hyfe.live_cnn import load_model
            import torch
            model, _ = load_model(str(stem))
            if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
                raise ValueError("nonfinite model weights")
    if seen != set(range(10)):
        raise ValueError("seeds must be exactly 0..9")
    event = manifest.get("event_model")
    if not event or sha(event["path"]) != event["sha256"]:
        raise ValueError("missing/modified event model")
    if check_weights:
        import lightgbm as lgb
        if lgb.Booster(model_file=event["path"]).num_feature() != 27:
            raise ValueError("event model must have the 27 paper features")
    return manifest


def release_manifest(directory, version, cutoff, **extra):
    entries = []
    for seed in range(10):
        stem = Path(directory) / f"final_i1_heatf_4h_W60_H84_sd{seed}"
        entries.append({"seed": seed, "stem": str(stem.resolve()),
                        "sha256": {ext: sha(stem.with_suffix(ext)) for ext in (".pt", ".json")}})
    event = Path(directory) / "event_model.txt"
    return {"version": version, "rule_version": RULE_VERSION, "data_cutoff": cutoff,
            "created_ms": int(time.time() * 1000), "models": entries,
            "event_model": {"path": str(event.resolve()), "sha256": sha(event)}, **extra}


def bootstrap(runtime=RUNTIME):
    runtime = Path(runtime)
    pointer = runtime / "active_models.json"
    if pointer.exists():
        return validate_manifest(json.loads(pointer.read_text()))
    version = "baseline-20260908-v1"
    dest = runtime / "models" / version
    dest.mkdir(parents=True, exist_ok=True)
    for seed in range(10):
        for ext in (".pt", ".json"):
            name = f"final_i1_heatf_4h_W60_H84_sd{seed}{ext}"
            if not (dest / name).exists():
                shutil.copy2(ROOT / "work/models" / name, dest / name)
    if not (dest / "event_model.txt").exists():
        shutil.copy2(ROOT / "work/results/hold_gbm_event_15m_W120_H10_s4_model.txt", dest / "event_model.txt")
    manifest = release_manifest(dest, version, "2026-09-01", training_validation_start="2026-06-01",
                                note="Existing frozen deployment; validation June-July, test August.")
    validate_manifest(manifest, check_weights=True)
    atomic_json(dest / "manifest.json", manifest)
    atomic_json(pointer, manifest)
    return manifest


def training_plan(now=None):
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    cutoff = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    val = cutoff - pd.DateOffset(months=2)
    next_q = pd.Timestamp(year=now.year, month=((now.month - 1) // 3) * 3 + 1, day=1, tz="UTC") + pd.DateOffset(months=3)
    return {"schema": 1, "rule_version": RULE_VERSION, "data_cutoff": cutoff.isoformat(),
            "validation_start": val.isoformat(), "next_quarter": next_q.isoformat(),
            "training_window_end_exclusive": (val - pd.Timedelta(hours=4 * (60 + 84))).isoformat(),
            "label_horizon_hours": 336, "seeds": list(range(10)), "epochs": 8, "batch_size": 256,
            "selection": "validation AP early stopping; no live-profit or best-seed selection",
            "promotion": "all ten validated models, only new cohorts use the new release"}


def resource_status():
    memory = dict((a.rstrip(":"), int(b)) for a, b, *_ in
                  (line.split() for line in Path("/proc/meminfo").read_text().splitlines()))
    free_gb = memory["MemAvailable"] / 1024**2
    try:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=5)
        utilization = max(float(v) for v in raw.splitlines())
    except (OSError, ValueError, subprocess.SubprocessError):
        utilization = 100.0
    return {"available_gb": round(free_gb, 1), "gpu_percent": utilization,
            "ready": free_gb >= 40 and utilization < 15}


def prepare_snapshot(dest, bases, buffer, cutoff_ms):
    """Resume exactly the original snapshot, even if history is later corrected."""
    from hyfe.paper_data import refresh_bars
    dest = Path(dest)
    manifest = dest / "data_manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        if data["cutoff_ms"] != cutoff_ms or any(sha(dest / p) != h for p, h in data["files"].items()):
            raise ValueError("training snapshot changed during resume")
        return data["bases"]
    for res in ("4h", "15m"):
        (dest / "bars" / res).mkdir(parents=True, exist_ok=True)
    latest = []
    for base in bases:
        for res, step in (("4h", 4*3_600_000), ("15m", 15*60_000)):
            d = refresh_bars(base, res, buffer, cutoff_ms)
            if d.empty:
                raise ValueError("empty training bars")
            if res == "4h":
                latest.append(int(d.ts.max()) + step)
            d.to_parquet(dest / "bars" / res / f"{base}.parquet", index=False)
    if max(latest) < cutoff_ms:
        raise ValueError("training cache is stale")
    atomic_json(manifest, {"cutoff_ms": cutoff_ms, "bases": bases,
                "files": {str(p.relative_to(dest)): sha(p) for p in (dest / "bars").rglob("*.parquet")}})
    return bases


def retrain(runtime=RUNTIME, run=False):
    """Low-cost scheduler check; busy hardware leaves the existing model intact."""
    runtime = Path(runtime)
    plan = training_plan()
    active = bootstrap(runtime)
    release = "quarterly-" + plan["data_cutoff"][:10]
    statefile = runtime / "training_status.json"
    if active["version"] == release:
        return {"status": "current", "version": release}
    # An initial refresh is due once, followed by calendar quarters.
    if active["version"].startswith("quarterly-") and pd.Timestamp.now(tz="UTC") < pd.Timestamp(active["next_quarter"]):
        return {"status": "current", "next_quarter": active["next_quarter"]}
    resources = resource_status()
    status = {"status": "planned" if not run else "waiting_resources", "plan": plan, "resources": resources}
    import fcntl
    with (runtime / "training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "already_running"}
        atomic_json(runtime / "training_plan.json", plan)
        atomic_json(statefile, status)
        if not run or not resources["ready"]:
            return status
        try:
            from hyfe.live import load_buffer
            from hyfe.paper_data import monthly_universe
            buffer = load_buffer()
            cutoff_ms = int(pd.Timestamp(plan["data_cutoff"]).timestamp() * 1000)
            bases = monthly_universe(runtime, buffer, cutoff_ms)
            if len(bases) < 100:
                raise ValueError("insufficient training universe")
            dest = runtime / "models" / release
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / "manifest.json").exists():
                ready = validate_manifest(json.loads((dest / "manifest.json").read_text()), True)
                atomic_json(runtime / "active_models.json", ready)
                return {"status": "promoted", "version": release}
            bases = prepare_snapshot(dest, bases, buffer, cutoff_ms)
            (dest / "bases.txt").write_text("\n".join(bases) + "\n")
            env = dict(os.environ, HYFE_BARS=str(dest / "bars"), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
            with (dest / "train.log").open("a") as log:
                for seed in range(10):
                    stem = dest / f"final_i1_heatf_4h_W60_H84_sd{seed}"
                    if stem.with_suffix(".json").exists() and stem.with_suffix(".pt").exists():
                        continue
                    atomic_json(statefile, {"status": "training", "seed": seed, "plan": plan})
                    cmd = [sys.executable, "-m", "hyfe.train_cnn", "--res", "4h", "--W", "60", "--H", "84",
                           "--model", "i1", "--render", "heatf", "--label", "relbin", "--stride", "1", "--bases", "@" + str(dest / "bases.txt"),
                           "--start", "2023-01", "--val", plan["validation_start"][:10], "--test", plan["data_cutoff"][:10],
                           "--test_end", plan["data_cutoff"][:10], "--seed", str(seed), "--cap", "5000000", "--epochs", "8", "--bs", "256",
                           "--out", str(stem.with_suffix(".json")), "--save_pred", "--final"]
                    subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                event_cmd = [sys.executable, "-m", "hyfe.train_final", "--cutoff", plan["data_cutoff"][:10],
                             "--os_months", "2", "--configs", "15m:120:10", "--bases", "@" + str(dest / "bases.txt"),
                             "--threads", "4", "--out", str(dest), "--tag", "event"]
                subprocess.run(event_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                shutil.copy2(dest / "event_15m_W120_H10_model.txt", dest / "event_model.txt")
            manifest = release_manifest(dest, release, plan["data_cutoff"], next_quarter=plan["next_quarter"], plan=plan)
            validate_manifest(manifest, check_weights=True)
            atomic_json(dest / "manifest.json", manifest)
            atomic_json(runtime / "active_models.json", manifest)
            status = {"status": "promoted", "version": release, "plan": plan}
        except Exception as exc:
            status = {"status": "failed", "error": str(exc)[:300], "plan": plan}
        atomic_json(statefile, status)
        return status


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    print(json.dumps(retrain(run=args.run), ensure_ascii=False))
