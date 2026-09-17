"""Re-evaluate frozen predictions with causal rules, writing private new outputs.

Never rewrites the submitted paper or the original experiment artifacts.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hyfe import bars as B
from hyfe.cohort import run
from hyfe.paper_rules import STEP_MS, HOLD, ensemble, RULE_VERSION
from hyfe.paper_models import ROOT, RUNTIME, atomic_json
from hyfe.perf import stats


def audit(out, folds=(0, 1, 2, 3, 4)):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / "work/results"
    rows, pooled = [], {"equal": [], "event": []}
    for fold in folds:
        kind = "hold" if fold == 4 else "full"
        middle = "" if fold == 4 else "_liq"
        tables = []
        for seed in range(10):
            seed_tag = "" if seed == 0 else f"_sd{seed}"
            p = source / f"{kind}_i1_heatf{seed_tag}{middle}_4h_W60_H84_s{fold}_pred.npz"
            z = np.load(p, allow_pickle=True)
            d = pd.DataFrame({"ts": z["ts"], "base": z["base"].astype(str),
                              f"seed{seed}": z["p"][:, 1] - z["p"][:, 2]})
            tables.append(d)
        frame = tables[0]
        for d in tables[1:]:
            frame = frame.merge(d, on=["ts", "base"], validate="one_to_one")
        frame["score"] = ensemble(frame, [f"seed{i}" for i in range(10)])
        pred = out / f"causal_s{fold}_pred.npz"
        s = frame.score.to_numpy()
        np.savez_compressed(pred, ts=frame.ts, base=frame.base.to_numpy(str), p=np.c_[np.zeros(len(s)), s.clip(0), (-s).clip(0)])
        event_file = source / f"{kind}_gbm_event_15m_W120_H10_s{fold}_pred.npz"
        variants = {"equal": ""}
        coverage = None
        if event_file.exists():
            ev = np.load(event_file, allow_pickle=True)
            event = pd.DataFrame({"decision": ev["ts"] + 15*60_000, "base": ev["base"].astype(str),
                                  "event": ev["p"][:, 1] + ev["p"][:, 2]})
            aligned = pd.merge_asof(frame.assign(decision=frame.ts+STEP_MS).sort_values("decision"),
                                    event.sort_values("decision"), on="decision", by="base",
                                    direction="backward", tolerance=75*60_000)
            coverage = float(aligned.event.notna().mean())
            # Same-time median only. Whole-test medians would leak future data.
            aligned["event"] = aligned.event.fillna(aligned.groupby("ts").event.transform("median"))
            if aligned.event.isna().any():
                raise ValueError(f"fold {fold}: a complete event cross section is unavailable")
            aligned["w"] = aligned.groupby("ts").event.rank(pct=True)
            wp = out / f"causal_s{fold}_event_weights.npz"
            np.savez_compressed(wp, ts=aligned.ts, base=aligned.base.to_numpy(str), w=aligned.w)
            variants["event"] = str(wp)
        B.OUT = str(ROOT / ("work/bars_hold" if fold == 4 else "work/bars_full"))
        for variant, weight in variants.items():
            returns, daily = run(str(pred), "4h", HOLD, weight=weight)
            result = {"rule_version": RULE_VERSION, "fold": fold, "variant": variant,
                      "event_coverage": coverage, "summary": stats(returns, daily, 14),
                      "daily": {str(k.date()): float(v) for k, v in daily.items()}}
            atomic_json(out / f"{variant}_s{fold}.json", result)
            rows.append({k: v for k, v in result.items() if k != "daily"})
            if fold != 4:
                pooled[variant].append(returns)
        print(json.dumps({"fold": fold, "status": "saved", "variants": list(variants)}), flush=True)
    for variant, values in pooled.items():
        if len(values) == 4:
            r = pd.concat(values).sort_index()
            rows.append({"fold": "pooled", "variant": variant, "summary": stats(r, (1+r).cumprod(), 14)})
    atomic_json(out / "summary.json", {"rule_version": RULE_VERSION,
                "notes": ["Per-decision population standardization; no future period statistics.",
                          "4h and 15m OPEN timestamps converted to CLOSE before event alignment.",
                          "Same-time event median for missing observations; no future imputation.",
                          "Fees included; this re-evaluation is before funding. Original artifacts unchanged."],
                "results": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    args = parser.parse_args()
    audit(args.out, tuple(int(i) for i in args.folds.split(",")))
