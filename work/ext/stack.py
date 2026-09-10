"""확장 ③ CNN+GBM 결합 · 급등급락 확률로 크기 조절 — 학습 없이 기존 예측 파일만으로. 4폴드(홀드아웃은 GBM v2 예측이 없어 제외).
  cnn  = work/results/ens_heatf10_direct_4h_s{i}_pred.npz   gbm = ens_v2_4h_s{i}_pred.npz   event = full_gbm_event_15m_W120_H10_s{i}_pred.npz(15m, 4h 정렬 시각만)
  변형: mix(z-합 50/50) · wev(CNN, 다리 안 비중 ∝ 급등급락 확률 순위) · gate(CNN, 확률 상위 절반 안에서만 십분위) · wiv(CNN, 비중 ∝ 1/σ_H)
usage: HYFE_BARS=work/bars_full PYTHONPATH=. python3 work/ext/stack.py → results/stack_table.csv"""
import json, os, subprocess
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from hyfe.perf import daily_returns, stats

R = "work/results"; os.makedirs("results", exist_ok=True)


def load(f):
    z = np.load(f, allow_pickle=True); return pd.DataFrame({"ts": z["ts"], "base": z["base"].astype(str), "s": z["p"][:, 1] - z["p"][:, 2], "y": z["y"], "fwd": z["fwd"], "sigH": z["sigH"]})


def save(df, s, out):
    p = np.c_[np.zeros(len(df)), np.clip(s, 0, None), np.clip(-s, 0, None)]
    np.savez_compressed(out, ts=df.ts.to_numpy(), base=df.base.to_numpy().astype(str), y=df.y.to_numpy(), p=p, fwd=df.fwd.to_numpy(), sigH=df.sigH.to_numpy())


def coh(pred, weight="", tag="", q=0.1):
    out = pred.replace("_pred.npz", "_cohort_H84" + (f"_w{tag}" if weight else "") + ".json")
    if not os.path.exists(out):
        subprocess.run(f"python3 -m hyfe.cohort --pred {pred} --res 4h --H 84 --cost 0.001 --q {q}" + (f" --weight {weight} --tag {tag}" if weight else ""), shell=True, check=True, capture_output=True)
    return out


rows, pooled, corr = [], {}, []
for s in range(4):
    c = load(f"{R}/ens_heatf10_direct_4h_s{s}_pred.npz"); g = load(f"{R}/ens_v2_4h_s{s}_pred.npz")
    m = c.merge(g[["ts", "base", "s"]].rename(columns={"s": "sg"}), on=["ts", "base"])
    rho = m.groupby("ts").apply(lambda d: spearmanr(d.s, d.sg)[0] if len(d) > 10 else np.nan).mean(); corr.append(rho)
    z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
    ev = load(f"{R}/full_gbm_event_15m_W120_H10_s{s}_pred.npz")
    zz = np.load(f"{R}/full_gbm_event_15m_W120_H10_s{s}_pred.npz", allow_pickle=True); ev["pev"] = zz["p"][:, 1] + zz["p"][:, 2]
    ev = ev[["ts", "base", "pev"]].sort_values("ts")   # 발생 모델은 15m 격자에서 5봉(75분)마다 점수 → 판단 시각 직전 75분 안의 최신 점수를 붙인다
    ce = pd.merge_asof(c.sort_values("ts"), ev, on="ts", by="base", direction="backward", tolerance=75 * 60_000); cov = ce.pev.notna().mean()
    ce["pev"] = ce.pev.fillna(ce.groupby("ts").pev.transform("median")).fillna(ce.pev.median())
    ce["rk"] = ce.groupby("ts").pev.rank(pct=True)
    variants = {"cnn": (f"{R}/ens_heatf10_direct_4h_s{s}_pred.npz", "", ""), "gbm": (f"{R}/ens_v2_4h_s{s}_pred.npz", "", "")}
    save(m, 0.5 * z(m.s) + 0.5 * z(m.sg), f"{R}/ens_cnngbm_4h_s{s}_pred.npz"); variants["mix"] = (f"{R}/ens_cnngbm_4h_s{s}_pred.npz", "", "")
    np.savez_compressed(f"{R}/w_ev_4h_s{s}.npz", ts=ce.ts.to_numpy(), base=ce.base.to_numpy().astype(str), w=ce.rk.to_numpy()); variants["wev"] = (variants["cnn"][0], f"{R}/w_ev_4h_s{s}.npz", "ev")
    np.savez_compressed(f"{R}/w_iv_4h_s{s}.npz", ts=ce.ts.to_numpy(), base=ce.base.to_numpy().astype(str), w=1.0 / np.maximum(ce.sigH.to_numpy(), 1e-6)); variants["wiv"] = (variants["cnn"][0], f"{R}/w_iv_4h_s{s}.npz", "iv")
    np.savez_compressed(f"{R}/w_evi_4h_s{s}.npz", ts=ce.ts.to_numpy(), base=ce.base.to_numpy().astype(str), w=ce.rk.to_numpy() / np.maximum(ce.sigH.to_numpy(), 1e-6)); variants["wevi"] = (variants["cnn"][0], f"{R}/w_evi_4h_s{s}.npz", "evi")
    gt = ce[ce.rk >= 0.5]; save(gt, gt.s.to_numpy(), f"{R}/ens_heatf10gate_4h_s{s}_pred.npz"); variants["gate"] = (f"{R}/ens_heatf10gate_4h_s{s}_pred.npz", "", "")
    for k, (pred, w, tag) in variants.items():
        j = coh(pred, w, tag, q=0.2 if k == "gate" else 0.1)   # gate 는 절반 안에서 상·하위 20% = 전체 대비 같은 종목 수
        st = json.load(open(j))["summary"]; rows.append({"variant": k, "fold": s, **st, "ev_cov": round(cov, 3) if k in ("wev", "gate") else None})
        pooled.setdefault(k, []).append(daily_returns(j)[0])
    print(f"fold {s}: CNN·GBM 시각별 Spearman {rho:.3f}, 발생확률 덮음 {cov:.2%}", flush=True)
def dsharpe_ci(a, b, block=14, n=2000, seed=0):
    """합산 일수익 두 열의 Sharpe 차(a − b) 블록 부트스트랩 95% CI 와 P(Δ ≤ 0). 같은 날을 함께 뽑아 짝지은 비교."""
    a, b = a.align(b, join="inner"); rng = np.random.default_rng(seed); n_ = len(a); nb = int(np.ceil(n_ / block)); d = []
    sh = lambda x: x.mean() / (x.std() + 1e-12) * np.sqrt(365)
    for _ in range(n):
        ii = np.concatenate([np.arange(s, min(s + block, n_)) for s in rng.integers(0, n_ - block + 1, nb)])[:n_]
        d.append(sh(a.iloc[ii]) - sh(b.iloc[ii]))
    d = np.array(d); return round(float(sh(a) - sh(b)), 2), round(float(np.quantile(d, 0.025)), 2), round(float(np.quantile(d, 0.975)), 2), round(float((d <= 0).mean()), 3)


base_r = pd.concat(pooled["cnn"])
for k, rs in pooled.items():
    rr = pd.concat(rs); row = {"variant": k, "fold": "pooled", **stats(rr, (1 + rr).cumprod(), 14)}
    if k != "cnn":
        row["dSharpe_vs_cnn"], row["ci_lo"], row["ci_hi"], row["p_le0"] = dsharpe_ci(rr, base_r)
    rows.append(row)
df = pd.DataFrame(rows); df.to_csv("results/stack_table.csv", index=False)
print(df[df.fold == "pooled"][["variant", "sharpe", "p", "mdd", "dSharpe_vs_cnn", "ci_lo", "ci_hi", "p_le0"]].to_string(index=False))
print(df.pivot(index="variant", columns="fold", values="sharpe").to_string()); print("pooled p:", df[df.fold == "pooled"].set_index("variant").p.to_dict(), "| corr", np.round(corr, 3))
