"""As-of monthly liquidity and private training snapshots, never collector writes."""
import json
from pathlib import Path

import pandas as pd

from hyfe import bars as B
from hyfe.paper_models import ROOT, atomic_json


def refresh_bars(base, resolution, buffer, cutoff_ms):
    from hyfe.live import recent_1m
    path = ROOT / "work/bars_hold" / resolution / (base + ".parquet")
    old = pd.read_parquet(path) if path.exists() else None
    minutes = B.RES_MIN[resolution]
    # If the immutable historical cache already covers the cutoff, no 1m read.
    if old is not None and not old.empty and int(old.ts.max()) + minutes*60_000 >= cutoff_ms:
        return old[old.ts + minutes*60_000 <= cutoff_ms].copy()
    raw = recent_1m(base, buffer)
    recent = B.to_res(raw, minutes) if raw is not None and len(raw) >= minutes else None
    pieces = [d for d in [old, recent] if d is not None and not d.empty]
    if not pieces:
        return pd.DataFrame(columns=["ts", "o", "h", "l", "c", "v", "qv"])
    d = pd.concat(pieces).drop_duplicates("ts", keep="last").sort_values("ts")
    return d[d.ts + minutes*60_000 <= cutoff_ms].reset_index(drop=True)


def monthly_universe(runtime, buffer, decision_ms, n=200):
    close = pd.Timestamp(decision_ms, unit="ms", tz="UTC").replace(day=1, hour=0, minute=0, second=0)
    cutoff = int(close.timestamp()*1000)
    path = Path(runtime) / f"universe-{close:%Y-%m}.json"
    if path.exists():
        return json.loads(path.read_text())["bases"]
    u = pd.read_csv(ROOT / "work/universe.csv")
    candidates = u[u.exclude.fillna("") == ""].base.tolist()
    values = {}
    for base in candidates:
        d = refresh_bars(base, "1h", buffer, cutoff)
        d = d[(d.ts >= cutoff-12*30*86400000) & (d.ts < cutoff)]
        if len(d) >= 24*60:
            values[base] = float(d.qv.sum())
    bases = sorted(sorted(values, key=lambda b: (-values[b], b))[:n])
    if len(bases) < 40:
        raise ValueError("insufficient as-of liquidity universe")
    atomic_json(path, {"cutoff_ms": cutoff, "lookback_days": 360, "bases": bases,
                       "quote_volume": {b: values[b] for b in bases}})
    return bases
