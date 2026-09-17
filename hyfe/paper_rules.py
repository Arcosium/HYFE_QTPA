"""Causal signal and cohort rules shared by research and forward paper trading.

All times are UTC milliseconds. A decision timestamp is the CLOSE of the input
bar, whereas historical prediction files store the OPEN of that last bar.
"""
import numpy as np
import pandas as pd

STEP_MS = 4 * 3_600_000
HOLD = 84
COST = 0.001
RULE_VERSION = "heatf-xs-cohort-v1"


def ensemble(frame, columns, weights=None):
    """Population z scores within each decision, then a fixed seed average.

The complete seed intersection is mandatory. Missing/NaN predictions must not
silently become zero or reduce an ensemble to whichever models happened to load.
"""
    if not columns or frame.empty or frame.duplicated(["ts", "base"]).any():
        raise ValueError("empty ensemble or duplicate decision/base")
    x = frame[columns].astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("nonfinite seed prediction")
    w = np.ones(len(columns)) if weights is None else np.asarray(weights, float)
    if w.shape != (len(columns),) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("invalid ensemble weights")
    g = x.groupby(frame.ts)
    mu = g.transform("mean")
    sd = g.transform("std", ddof=0)
    z = (x - mu) / sd.where(sd > 1e-12, 1.0)
    return pd.Series(z.to_numpy() @ (w / w.sum()), index=frame.index)


def select_legs(scores, q=0.1):
    """Match the paper's percentile-rank boundaries, including ties."""
    s = pd.Series(scores, dtype=float)
    if not 0 < q < 0.5 or not np.isfinite(s).all():
        raise ValueError("invalid scores/quantile")
    rank = s.rank(pct=True, method="average")
    return rank >= 1 - q, rank <= q


def leg_weights(values, n):
    """Unit leg weights; the caller applies the 0.5 capital allocation."""
    if n < 1:
        raise ValueError("empty leg")
    a = np.ones(n) if values is None else np.asarray(values, dtype=float).copy()
    if a.shape != (n,) or np.isinf(a).any() or (a[np.isfinite(a)] < 0).any():
        raise ValueError("invalid leg weights")
    valid = np.isfinite(a)
    a[~valid] = a[valid].mean() if valid.any() else 1.0
    if a.sum() <= 0:
        raise ValueError("zero leg weight")
    return a / a.sum()


def leg_mean(values, weights=None):
    """Historical missing-price convention, also used by cohort.py."""
    a = np.asarray(values, float)
    w = leg_weights(weights, a.shape[-1])
    den = (np.isfinite(a) * w).sum(axis=-1)
    return np.divide(np.nansum(a * w, axis=-1), den,
                     out=np.full(den.shape, np.nan), where=den > 0)


def cohort_path(long_values, short_values, long_weights=None, short_weights=None,
                cost=COST, hold=HOLD):
    """Entry-relative simple P&L with exactly 2*cost/hold accrued each bar."""
    l = leg_mean(long_values, long_weights)
    s = leg_mean(short_values, short_weights)
    return 0.5 * (l - s) - 2 * cost * np.arange(len(l)) / hold


def signal_positions(frame, q=0.1):
    """Two preregistered variants share selections: equal and event rank."""
    if frame.duplicated("base").any():
        raise ValueError("duplicate signal base")
    if not np.isfinite(frame[["score", "entry", "event"]].to_numpy(float)).all():
        raise ValueError("nonfinite signal")
    if (frame.entry <= 0).any() or ((frame.event < 0) | (frame.event > 1 + 1e-6)).any():
        raise ValueError("invalid entry/event probability")
    d = frame.copy()
    d["event_rank"] = d.event.rank(pct=True, method="average")
    long, short = select_legs(d.score, q)
    if not long.any() or not short.any():
        raise ValueError("scores do not select both legs")
    positions = []
    for side, mask in (("long", long), ("short", short)):
        g = d[mask].sort_values(["score", "base"], ascending=[side == "short", True])
        ew = leg_weights(g.event_rank.to_numpy(), len(g))
        for (_, row), w in zip(g.iterrows(), ew):
            positions.append({"base": str(row.base), "side": side,
                              "entry": float(row.entry), "score": float(row.score),
                              "equal": 0.5 / len(g), "event": 0.5 * float(w)})
    return positions


def advance_cohort(cohort, prices, funding=None, cost=COST, hold=HOLD):
    """One completed holding bar. Mutates only a caller-owned working copy.

Missing marks fail closed. Uncovered funding is estimated from the same leg,
as in the paper, and its coverage is returned explicitly (never called actual).
"""
    result = {}
    funding = funding or {}
    pp = cohort["positions"]
    for p in pp:
        px = prices.get(p["base"])
        if px is None or not np.isfinite(px) or px <= 0:
            raise ValueError(f"missing/invalid holding mark: {p['base']}")
    for variant in ("equal", "event"):
        gross = charge = covered = 0.0
        for side, sign in (("long", 1), ("short", -1)):
            leg = [p for p in pp if p["side"] == side]
            observed = [p for p in leg if p["base"] in funding]
            den = sum(p[variant] for p in observed)
            proxy = sum(p[variant] * funding[p["base"]] for p in observed) / den if den else 0.0
            for p in leg:
                w = p[variant]
                gross += sign * w * (prices[p["base"]] - p.get("mark", p["entry"])) / p["entry"]
                charge += sign * w * funding.get(p["base"], proxy)
                covered += w * (p["base"] in funding)
                p.setdefault("funding", {}).setdefault(variant, 0.0)
                p["funding"][variant] += funding.get(p["base"], proxy)
        result[variant] = {"gross": gross, "net": gross - 2 * cost / hold,
                           "funded": gross - 2 * cost / hold - charge,
                           "funding": charge, "funding_coverage": covered}
    for p in pp:
        p["mark"] = float(prices[p["base"]])
    cohort["age"] = cohort.get("age", 0) + 1
    return result
