"""MLP 대조군(m1=피처 37, m2=heatf 수치 행렬 48×60) 판정: 시드 0·1·2 × 4폴드 코호트 롱숏 → 시드별·3시드 앙상블·합산. 출력 vault/paper/mlp_table.csv"""
import os, sys, json, glob, subprocess, re
import pandas as pd, numpy as np
os.chdir("/home/arcosium/projects/HYFE_QTPA"); sys.path.insert(0, "."); os.environ["HYFE_BARS"] = "work/bars_full"
from hyfe.perf import daily_returns, stats
R = "work/results"; OUT = "/home/arcosium/vault/HYFE_QTPA/paper"
def coh(pred):
    out = pred.replace("_pred.npz", "_cohort_H84.json")
    if not os.path.exists(out): subprocess.run(f"python3 -m hyfe.cohort --pred {pred} --res 4h --H 84 --cost 0.001", shell=True, capture_output=True)
    return json.load(open(out))["summary"] if os.path.exists(out) else None
rows = []
for m in ("m1", "m2"):
    per = {}
    for s in (0, 1, 2):
        sd = "" if s == 0 else f"_sd{s}"
        for f in "0123":
            p = f"{R}/full_{m}_heatf{sd}_liq_4h_W60_H84_s{f}_pred.npz"
            if os.path.exists(p): per[(s, f)] = coh(p)
    for s in (0, 1, 2):
        r = {"model": f"{m} 시드 {s}"}
        for f, lab in zip("3210", "ABCD"): r[lab] = per.get((s, f), {}).get("sharpe", np.nan)
        fs = [f"{R}/full_{m}_heatf{'' if s == 0 else f'_sd{s}'}_liq_4h_W60_H84_s{f}_cohort_H84.json" for f in "0123"]
        if all(os.path.exists(x) for x in fs):
            rr = pd.concat([daily_returns(x)[0] for x in fs]); st = stats(rr, (1 + rr).cumprod(), 14); r.update(pooled=st["sharpe"], p=st["p"], mdd=st["mdd"])
        rows.append(r)
    r = {"model": f"{m} 시드 3개 앙상블"}; fs = []
    for f in "0123":
        preds = [f"{R}/full_{m}_heatf{'' if s == 0 else f'_sd{s}'}_liq_4h_W60_H84_s{f}_pred.npz" for s in (0, 1, 2)]; preds = [x for x in preds if os.path.exists(x)]
        if len(preds) < 2: continue
        ens = f"{R}/ens_{m}{len(preds)}_direct_4h_s{f}_pred.npz"
        if not os.path.exists(ens): subprocess.run(f"python3 -m hyfe.zavg {' '.join(preds)} --out {ens}", shell=True, capture_output=True)
        sm = coh(ens); r["ABCD"["3210".index(f)]] = sm["sharpe"] if sm else np.nan; fs.append(ens.replace("_pred.npz", "_cohort_H84.json"))
    if len(fs) == 4:
        rr = pd.concat([daily_returns(x)[0] for x in fs]); st = stats(rr, (1 + rr).cumprod(), 14); r.update(pooled=st["sharpe"], p=st["p"], mdd=st["mdd"])
    rows.append(r)
# 참고: heatf CNN 시드 3개 앙상블·시드 10개 앙상블·LightGBM
fs = [f"{R}/ens_heatf3_direct_4h_s{f}_cohort_H84.json" for f in "0123"]; rr = pd.concat([daily_returns(x)[0] for x in fs]); st = stats(rr, (1 + rr).cumprod(), 14)
rows.append({"model": "heatf 소형 CNN 시드 3개 앙상블", **{lab: json.load(open(f"{R}/ens_heatf3_direct_4h_s{f}_cohort_H84.json"))["summary"]["sharpe"] for f, lab in zip("3210", "ABCD")}, "pooled": st["sharpe"], "p": st["p"], "mdd": st["mdd"]})
rows.append({"model": "heatf 소형 CNN 시드 10개 앙상블", "A": 5.59, "B": 4.95, "C": 2.92, "D": 2.28, "pooled": 3.53, "p": 0.0, "mdd": -4.2})
rows.append({"model": "LightGBM v2 (같은 피처 37개)", "A": 4.48, "B": 2.68, "C": 1.54, "D": 1.22, "pooled": 2.18, "p": 0.0005, "mdd": -5.7})
df = pd.DataFrame(rows); pd.set_option("display.width", 200); print(df.round(3).to_string(index=False)); df.to_csv(f"{OUT}/mlp_table.csv", index=False)
