"""확장 ⑤ (④ 가 유의할 때만 논문에 넣는다) — 운용 규모별 충격비용과 국면별 성과. 학습 없음, 시드 10 앙상블 4폴드 예측.
충격: 제곱근 법칙 cost_side = Y·σ_day·√(Q/ADV), Y=0.7(Tóth 2011 범위 0.5~1), Q = 종목 하루 거래 명목 = F/(2·H·n_leg)×6 판단×2(진입+청산)/2 … 편도 기준 F·6/(2·84·n),
   ADV = 판단 직전 30일 일평균 달러 거래대금(4h 봉 qv), σ_day = sigH/√84·√6. 다리 평균 충격을 코호트 편도 비용에 더해 재판정(0.001 + 충격).
국면: 합산 일수익을 BTC 직전 30일 수익(−10% 미만/±10%/+10% 초과)과 30일 실현변동성(중앙값 기준 저/고)으로 갈라 Sharpe.
usage: HYFE_BARS=work/bars_full PYTHONPATH=. python3 work/ext/capacity.py → results/ext_capacity.csv, results/ext_regime.csv"""
import json, os, subprocess
import numpy as np
import pandas as pd
from hyfe import bars as B
from hyfe.perf import daily_returns, stats

R = "work/results"; Y = 0.7; H = 84; NLEG = 20; FUNDS = [1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9]


def adv_table(bases):
    out = {}
    for b in bases:
        p = B.path("4h", b)
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p, columns=["ts", "qv"]); d["day"] = pd.to_datetime(d.ts, unit="ms").dt.floor("D")
        dv = d.groupby("day").qv.sum(); out[b] = dv.rolling(30, min_periods=10).mean().shift(1)   # 판단일 이전 30일
    return pd.DataFrame(out)


rows, impacts = [], {}
for s in range(4):
    z = np.load(f"{R}/ens_heatf10_direct_4h_s{s}_pred.npz", allow_pickle=True)
    d = pd.DataFrame({"ts": z["ts"], "base": z["base"].astype(str), "s": z["p"][:, 1] - z["p"][:, 2], "sigH": z["sigH"]})
    d["pr"] = d.groupby("ts").s.rank(pct=True); legs = d[(d.pr >= 0.9) | (d.pr <= 0.1)].copy(); legs["day"] = pd.to_datetime(legs.ts, unit="ms").dt.floor("D")
    A = adv_table(sorted(legs.base.unique())); legs["adv"] = [A.at[dy, b] if (b in A.columns and dy in A.index) else np.nan for dy, b in zip(legs.day, legs.base)]
    legs = legs.dropna(subset=["adv"]); legs["sig_day"] = legs.sigH / np.sqrt(H) * np.sqrt(6)
    for F in FUNDS:
        Q = F * 6 / (2 * H * NLEG)                                      # 종목당 하루 편도 거래 명목
        imp = (Y * legs.sig_day * np.sqrt(Q / legs.adv)).mean()         # 다리 평균 편도 충격(수익률 단위)
        impacts.setdefault(F, []).append(imp)
        link = f"{R}/ens_heatf10_direct_4h_s{s}_F{int(F/1e6)}m_pred.npz"
        if not os.path.exists(link):
            os.symlink(f"ens_heatf10_direct_4h_s{s}_pred.npz", link)
        out = link.replace("_pred.npz", f"_cohort_H{H}.json")
        if not os.path.exists(out):
            subprocess.run(f"python3 -m hyfe.cohort --pred {link} --res 4h --H {H} --cost {0.001 + imp:.6f}", shell=True, capture_output=True)
    print(f"fold {s}: 다리 종목 ADV 중앙값 ${legs.adv.median()/1e6:.1f}M, 하위 10% ${legs.adv.quantile(0.1)/1e6:.1f}M", flush=True)
for F in FUNDS:
    rs = pd.concat([daily_returns(f"{R}/ens_heatf10_direct_4h_s{s}_F{int(F/1e6)}m_cohort_H{H}.json")[0] for s in range(4)])
    rows.append({"fund_usd_m": F / 1e6, "impact_bp_side": round(np.mean(impacts[F]) * 1e4, 1), **stats(rs, (1 + rs).cumprod(), 14)})
cap = pd.DataFrame(rows); cap.to_csv("results/ext_capacity.csv", index=False); print(cap.to_string(index=False))

r = pd.concat([daily_returns(f"results/cohort/ens_heatf10_direct_4h_s{s}_cohort_H84.json")[0] for s in range(4)]).sort_index()
btc = pd.read_parquet(B.path("4h", "BTC"), columns=["ts", "c"]); btc.index = pd.to_datetime(btc.ts, unit="ms"); bd = btc.c.resample("D").last()
tr30 = (bd / bd.shift(30) - 1).shift(1); vol30 = np.log(bd).diff().rolling(30).std().shift(1) * np.sqrt(365)
g = pd.DataFrame({"r": r, "tr": tr30.reindex(r.index), "vol": vol30.reindex(r.index)}).dropna()
g["trend"] = pd.cut(g.tr, [-1, -0.10, 0.10, 10], labels=["BTC 30일 −10% 미만", "±10%", "+10% 초과"]); g["volr"] = np.where(g.vol > g.vol.median(), "고변동", "저변동")
reg = []
for k, sub in list(g.groupby("trend", observed=True)) + list(g.groupby("volr")):
    reg.append({"regime": k, **stats(sub.r, (1 + sub.r).cumprod(), 14)})
reg = pd.DataFrame(reg); reg.to_csv("results/ext_regime.csv", index=False); print(reg.to_string(index=False))
