"""Forward-only heatf cohorts, independent of the legacy GBM/slot engine.

Run through live_tick.sh's flock. Runtime and full experiment records live in
work/paper_trade (a vault symlink). The web reads an allowlisted view of SQLite.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hyfe import bars as B, live_cnn as LC
from hyfe.live import load_buffer, recent_1m, liquid_bases
from hyfe.pilot_gbm import add_xs
from hyfe.paper_rules import STEP_MS, ensemble
from hyfe.paper_models import RUNTIME, ROOT, bootstrap, atomic_json, validate_manifest
from hyfe.paper_store import connect, load_state, apply_decision, health, internal_report, resolve_executions
from hyfe.paper_funding import collect
from hyfe.paper_data import monthly_universe


def mark_prices(bars, close):
    out = {}
    for base, frame in bars.items():
        rows = frame[frame.ts == close - STEP_MS]
        if not rows.empty:
            out[base] = float(rows.iloc[-1].c)
    return out


def infer(decision, bases, minute_data, manifest):
    """Same closed data boundary for every seed, feature and event prediction."""
    four, quarter = {}, {}
    for base in bases:
        raw = minute_data.get(base)
        if raw is None or raw.empty or int(raw.ts.iloc[-1]) < decision - 60_000:
            continue
        four[base] = B.to_res(raw, 240)
        quarter[base] = B.to_res(raw, 15)
    return infer_bars(decision, bases, four, quarter, manifest)


def closed_features(bars, res, W, H, decision):
    """The live feature window, with synthetic label padding only after close."""
    from hyfe import features as F
    step = B.RES_MIN[res] * 60_000
    if bars is None:
        return None
    bars = bars[bars.ts + step <= decision]
    if len(bars) < W + 2:
        return None
    b30 = max(W, 30 * 1440 // B.RES_MIN[res])
    bars = bars.tail(b30 + W + 5).reset_index(drop=True)
    pad = pd.concat([bars.iloc[[-1]]] * H, ignore_index=True)
    pad["ts"] = bars.ts.iloc[-1] + step * np.arange(1, H + 1)
    f = F.make(pd.concat([bars, pad], ignore_index=True), W, H, stride=1, bars_per_30d=b30)
    f = f[f.ts == bars.ts.iloc[-1]]
    return f.iloc[-1] if len(f) else None


def infer_bars(decision, bases, four, quarter, manifest, loaded_models=None, event_model=None):
    """Shared live/replay inference; preaggregated inputs are clipped at close."""
    import lightgbm as lgb
    from hyfe import features as F
    rows, bb = [], {}
    event_rows = []
    for base in bases:
        bars = four.get(base)
        if bars is None:
            continue
        bars = bars[bars.ts + STEP_MS <= decision]
        if len(bars) < 242 or int(bars.ts.iloc[-1]) + STEP_MS != decision:
            continue
        feat = closed_features(bars, "4h", 60, 84, decision)
        event = closed_features(quarter.get(base), "15m", 120, 10, decision)
        if feat is None or event is None or int(feat.ts) + STEP_MS != decision or int(event.ts) + 900_000 != decision:
            continue
        rows.append({**feat.to_dict(), "base": base, "ts": decision, "entry": float(bars.c.iloc[-1])})
        event_rows.append(event[F.FEATURES].to_numpy(float))
        bb[base] = bars.tail(400).copy()
    if len(rows) < 40:
        raise ValueError(f"only {len(rows)} fresh complete symbols")
    frame = add_xs(pd.DataFrame(rows))
    ctx = LC.bar_context(bb)
    contexts = dict(tuple(ctx.groupby("base", sort=False)))
    columns = []
    for i, item in enumerate(manifest["models"]):
        model, meta = LC.load_model(item["stem"]) if loaded_models is None else loaded_models[i]
        images = [LC.render(contexts[row.base], row.base, decision, row, meta) for _, row in frame.iterrows()]
        if any(im is None or not torch.isfinite(im).all() for im in images):
            raise ValueError("invalid heatf image")
        values = np.concatenate([LC.score(model, torch.cat(images[i:i + 16])) for i in range(0, len(images), 16)])
        col = f"seed{item['seed']}"
        frame[col] = values
        columns.append(col)
    frame["score"] = ensemble(frame, columns)
    event_path = manifest["event_model"]["path"]
    booster = lgb.Booster(model_file=event_path) if event_model is None else event_model
    probabilities = booster.predict(np.stack(event_rows), num_threads=2)
    frame["event"] = probabilities[:, 1] + probabilities[:, 2]
    frame["event_decision_ms"] = decision
    keep = ["base", "ts", "entry", "score", "event", "event_decision_ms"] + columns + manifest_features(manifest)
    # Column names must remain unique: raw feature 'hour' is not a future label.
    return frame[list(dict.fromkeys(keep))]


def manifest_features(manifest):
    return json.loads(Path(manifest["models"][0]["stem"]).with_suffix(".json").read_text())["feat_cols"]


def tick(runtime=RUNTIME, dry=False, now_ms=None):
    runtime = Path(runtime)
    if not (runtime / "enabled.json").exists() and not dry:
        return {"status": "disabled"}
    torch.set_num_threads(2)
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    decision = now_ms // STEP_MS * STEP_MS
    manifest = (validate_manifest(json.loads((runtime / "active_models.json").read_text()))
                if dry else bootstrap(runtime))
    # Dry runs do not create or mutate the live database, logs or internal report.
    con = connect(runtime / "paper.sqlite3", readonly=dry) if (runtime / "paper.sqlite3").exists() else (None if dry else connect(runtime / "paper.sqlite3"))
    state = load_state(con) if con is not None else None
    buf = load_buffer()
    if not dry:
        resolve_executions(con, buf, now_ms)
    if state and decision <= state["last_ts"]:
        if con is not None:
            con.close()
        return {"status": "already_processed", "decision": decision}
    try:
        forward = 0 <= now_ms - decision <= 45 * 60_000
        if state is None and not forward:
            return {"status": "awaiting_next_close", "decision": decision}
        bases = (liquid_bases() if dry else monthly_universe(runtime, buf, decision)) if forward else []
        held = {p["base"] for c in (state or {}).get("cohorts", []) for p in c["positions"]}
        all_bases = sorted(set(bases) | held)
        minute = {base: recent_1m(base, buf) for base in all_bases}
        bars = {base: B.to_res(raw, 240) for base, raw in minute.items() if raw is not None and len(raw) >= 240}
        bars = {b: d for b, d in bars.items() if d is not None}
        frame = infer(decision, bases, minute, manifest) if forward else None
        boundaries = list(range(state["last_ts"] + STEP_MS, decision + 1, STEP_MS)) if state else []
        prices = {ts: mark_prices(bars, ts) for ts in boundaries}
        if dry:
            for ts in boundaries:
                missing = held - prices[ts].keys()
                if missing:
                    raise ValueError(f"missing holding prices for {len(missing)} symbols")
            return {"status": "dry", "decision": decision, "n_symbols": len(frame) if frame is not None else 0,
                    "models": len(manifest["models"]), "mark_boundaries": len(boundaries)}
        funding = collect(runtime, boundaries, held) if boundaries and held else {ts: {} for ts in boundaries}
        generated = max(now_ms, int(time.time() * 1000))
        changed = apply_decision(con, decision, generated, manifest["version"], frame, prices, funding)
        atomic_json(runtime / "internal_report.json", internal_report(con))
        health(con, "ok", generated, decision=decision, changed=changed, model_version=manifest["version"])
        return {"status": "ok", "decision": decision, "n_symbols": len(frame) if frame is not None else 0,
                "new_cohort": bool(frame is not None and changed), "models": len(manifest["models"])}
    except Exception as exc:
        if not dry:
            health(con, "error", int(time.time() * 1000), decision=decision, error=str(exc)[:300])
        raise
    finally:
        if con is not None:
            con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    try:
        print(json.dumps(tick(dry=args.dry), ensure_ascii=False), flush=True)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)[:300]}, ensure_ascii=False), flush=True)
        sys.exit(1)
