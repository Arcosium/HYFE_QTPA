"""Transactional, append-only forward experiment. No historical signals invented.

Only this database holds the new books; the original work/live ledger remains
intact. A complete decision, both variants and all marks commit together.
"""
import copy
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from hyfe.paper_rules import STEP_MS, HOLD, COST, RULE_VERSION, advance_cohort, signal_positions


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def connect(path, readonly=False):
    path = Path(path)
    if readonly:
        return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True, timeout=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS decisions (ts INTEGER PRIMARY KEY, generated_ms INTEGER NOT NULL,
            model_version TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS marks (ts INTEGER PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS closed (id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS executions (cohort INTEGER, base TEXT, side TEXT, kind TEXT,
            target_ms INTEGER NOT NULL, price REAL, observed_ms INTEGER,
            PRIMARY KEY(cohort,base,side,kind));
        CREATE TABLE IF NOT EXISTS health (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL);
    """)
    return con


def load_state(con):
    r = con.execute("SELECT payload FROM state WHERE id=1").fetchone()
    return json.loads(r[0]) if r else None


def new_state(ts):
    return {"schema": 1, "rule_version": RULE_VERSION, "step_ms": STEP_MS,
            "hold": HOLD, "cost": COST, "started": ts, "last_ts": ts,
            "cohorts": [], "n_closed": 0, "n_closed_positions": 0, "model_version": None,
            "books": {v: {"net": 1.0, "funded": 1.0, "funding_coverage": 1.0}
                      for v in ("equal", "event")}}


def apply_decision(con, decision, generated_ms, model_version, frame, prices, funding):
    """prices/funding map close timestamps to base->value dictionaries.

Missed boundaries settle existing holdings, but never open retroactive cohorts.
frame=None performs mark-only catch-up. A signal is forward only if generated
within 45 minutes after its decision, with future execution references recorded.
"""
    decision = int(decision)
    if decision % STEP_MS:
        raise ValueError("decision is not a 4h close")
    if frame is not None and not decision <= generated_ms <= decision + 45 * 60_000:
        raise ValueError("signal outside its forward decision window")
    pp = signal_positions(frame) if frame is not None else None
    con.execute("BEGIN IMMEDIATE")
    try:
        state = load_state(con)
        if con.execute("SELECT 1 FROM decisions WHERE ts=?", (decision,)).fetchone():
            con.rollback()
            return False
        if state is None:
            if pp is None:
                con.rollback()
                return False
            state = new_state(decision)
        if state["rule_version"] != RULE_VERSION:
            raise ValueError("rule version changed; a new experiment is required")
        if decision < state["last_ts"]:
            raise ValueError("out-of-order decision")
        if decision == state["last_ts"] and state["cohorts"]:
            # Do not add a late cohort to an already-accounted mark-only boundary.
            con.rollback()
            return False
        for ts in range(state["last_ts"] + STEP_MS, decision + 1, STEP_MS):
            active = state["cohorts"]
            results = [advance_cohort(c, prices[ts], funding[ts], state["cost"], state["hold"])
                       for c in active]
            for variant, book in state["books"].items():
                if results:
                    net = float(np.mean([r[variant]["net"] for r in results]))
                    funded = float(np.mean([r[variant]["funded"] for r in results]))
                    if min(net, funded) <= -1:
                        raise ValueError("portfolio insolvent")
                    book["net"] *= 1 + net
                    book["funded"] *= 1 + funded
                    book["funding_coverage"] = float(np.mean([r[variant]["funding_coverage"] for r in results]))
            expired = [c for c in active if c["age"] >= state["hold"]]
            for c in expired:
                con.execute("INSERT INTO closed VALUES (?,?)", (c["id"], dumps(c)))
            state["n_closed"] += len(expired)
            state["n_closed_positions"] += sum(len(c["positions"]) for c in expired)
            state["cohorts"] = [c for c in active if c["age"] < state["hold"]]
            con.execute("INSERT INTO marks VALUES (?,?)", (ts, dumps({
                "books": copy.deepcopy(state["books"]), "n_active": len(active),
                "closed": len(expired), "signal_missing": ts != decision or pp is None,
                "prices": prices[ts], "funding": funding[ts]})))
        if pp is not None:
            cohort = {"id": decision, "age": 0, "generated_ms": int(generated_ms),
                      "exit_ts": decision + HOLD * STEP_MS, "model_version": model_version,
                      "positions": pp}
            state["cohorts"].append(cohort)
            records = frame.to_dict(orient="records")
            con.execute("INSERT INTO decisions VALUES (?,?,?,?)", (
                decision, generated_ms, model_version,
                dumps({"rule_version": RULE_VERSION, "universe": records, "cohort": cohort})))
            target = (int(generated_ms) // 60_000 + 1) * 60_000
            for p in pp:
                for kind, when in (("entry", target), ("exit", target + HOLD * STEP_MS)):
                    con.execute("INSERT INTO executions(cohort,base,side,kind,target_ms) VALUES (?,?,?,?,?)",
                                (decision, p["base"], p["side"], kind, when))
            state["model_version"] = model_version
            state["last_signal"] = {"ts": decision, "n": len(frame), "n_positions": len(pp),
                                    "top": frame.nlargest(5, "score")[["base", "score"]].values.tolist(),
                                    "bottom": frame.nsmallest(5, "score")[["base", "score"]].values.tolist()}
        state["last_ts"] = decision
        state["updated_ms"] = int(generated_ms)
        con.execute("INSERT OR REPLACE INTO state VALUES (1,?)", (dumps(state),))
        con.commit()
        return True
    except BaseException:
        con.rollback()
        raise


def health(con, status, at_ms, **details):
    with con:
        con.execute("INSERT OR REPLACE INTO health VALUES (1,?)",
                    (dumps({"status": status, "at_ms": int(at_ms), **details}),))


def resolve_executions(con, minute_bars, observed_ms):
    """Observe the open of the first scheduled minute strictly after inference.

No forward-fill: if that exact minute was not actually collected, leave pending.
These observations are internal, and do not rewrite the paper-price book.
"""
    pending = con.execute("SELECT cohort,base,side,kind,target_ms FROM executions WHERE price IS NULL AND target_ms<?",
                          (int(observed_ms) - 60_000,)).fetchall()
    count = 0
    with con:
        for cid, base, side, kind, target in pending:
            bars = minute_bars.get(base)
            if bars is None:
                continue
            rows = bars[bars.ts == target]
            if rows.empty:
                continue
            px = float(rows.iloc[0].open)
            if not np.isfinite(px) or px <= 0:
                continue
            con.execute("UPDATE executions SET price=?,observed_ms=? WHERE cohort=? AND base=? AND side=? AND kind=? AND price IS NULL",
                        (px, int(observed_ms), cid, base, side, kind))
            count += 1
    return count


def internal_report(con):
    """Never exposed by the web API."""
    state = load_state(con)
    if state is None:
        return {"status": "not_started"}
    points = [(state["started"], {v: {"net": 1., "funded": 1.} for v in state["books"]})]
    points += [(ts, json.loads(p)["books"]) for ts, p in con.execute("SELECT ts,payload FROM marks ORDER BY ts")]
    report = {"rule_version": RULE_VERSION, "started_ms": state["started"], "last_ms": state["last_ts"],
              "decisions": con.execute("SELECT count(*) FROM decisions").fetchone()[0],
              "closed_cohorts": state["n_closed"], "variants": {}}
    for variant in state["books"]:
        series = pd.Series([b[variant]["funded"] for _, b in points],
                           index=pd.to_datetime([t for t, _ in points], unit="ms", utc=True))
        daily = series.groupby(series.index.floor("D")).last()
        r = daily.pct_change().dropna()
        report["variants"][variant] = {"total_return": float(series.iloc[-1] - 1),
            "mdd": float((series / series.cummax() - 1).min()), "days": len(r),
            "sharpe": float(r.mean() / r.std() * np.sqrt(365)) if len(r) >= 30 and r.std() > 0 else None,
            "funding_coverage": state["books"][variant]["funding_coverage"],
            "daily": {str(k.date()): float(v) for k, v in daily.items()}}
    return report
