"""경제 지표 백테스트(P6 예비) — 예측 npz(ts, base, p, fwd, sigH) 로 점수(급등−급락) 상·하위 q 분위 매매.
체결 규약: 신호 봉 e 의 종가 확정 후 다음 봉 시가 진입, H 봉 뒤 종가 청산, 편도 비용 cost(수수료 0.05%+슬리피지 0.05% 기본).
usage: HYFE_BARS=work/bars_full python3 -m hyfe.backtest --pred work/results/gbm_map_top200_15m_W240_H60_s0_pred.npz --res 15m --H 60
"""
import argparse, json, os
import numpy as np
import pandas as pd
from hyfe import bars as B, metrics as M


def trades(pred, res, H):
    """예측 행마다 실제 진입가(다음 봉 시가)·청산가(H 봉 뒤 종가)를 붙인다."""
    z = np.load(pred, allow_pickle=True)
    d = pd.DataFrame({"ts": z["ts"], "base": z["base"].astype(str), "y": z["y"], "score": M.score(z["p"]), "p_ev": z["p"][:, 1] + z["p"][:, 2], "fwd": z["fwd"], "sigH": z["sigH"]})
    out = []
    for b, g in d.groupby("base"):
        p = B.path(res, b)
        if not os.path.exists(p):
            continue
        bars = pd.read_parquet(p); ts = bars.ts.to_numpy()
        i = np.searchsorted(ts, g.ts.to_numpy())
        ok = (i + H < len(ts)) & (ts[np.minimum(i, len(ts) - 1)] == g.ts.to_numpy())
        g = g[ok]; i = i[ok]
        entry = bars.o.to_numpy()[i + 1]; exit_ = bars.c.to_numpy()[i + H]
        g = g.assign(entry=entry, exit=exit_, ret=exit_ / entry - 1, ts_exit=ts[i + H])
        out.append(g)
    return pd.concat(out, ignore_index=True)


def equity(x):
    """고정 비중 포트폴리오 근사: 동시 보유 중앙값 K 로 자본을 나눠 거래당 1/K 배분(무복리). 일별 수익률·MDD."""
    ev = np.concatenate([np.c_[x.ts.to_numpy(), np.ones(len(x))], np.c_[x.ts_exit.to_numpy(), -np.ones(len(x))]])
    ev = ev[np.argsort(ev[:, 0], kind="stable")]; open_ = np.cumsum(ev[:, 1])
    K = max(float(np.median(open_[open_ > 0])), 1.0)
    day = pd.to_datetime(x.ts_exit, unit="ms").dt.floor("D")
    daily = x.groupby(day).net.sum() / K
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"), fill_value=0.0)
    eq = 1 + daily.cumsum(); mdd = float((eq / eq.cummax() - 1).min())
    return K, daily, mdd


def summarize(t, cost, q, bars_per_day):
    lo, hi = t.score.quantile([q, 1 - q])
    L = t[t.score >= hi].copy(); S = t[t.score <= lo].copy()
    L["net"] = L.ret - 2 * cost; S["net"] = -S.ret - 2 * cost
    both = pd.concat([L.assign(side="L"), S.assign(side="S")])
    drift = float(t.ret.mean())                               # 같은 기간 무작위 분위(전체 후보)의 평균 수익 — 시장 베타 대조
    r = {"drift_bp": drift * 1e4}
    for name, x in (("long", L), ("short", S), ("LS", both)):
        if len(x) < 50:
            continue
        K, daily, mdd = equity(x)
        blocks = (pd.to_datetime(x.ts, unit="ms").dt.strftime("%Y-%m") + x.base).to_numpy()
        rng = np.random.default_rng(0); _, gi = np.unique(blocks, return_inverse=True)
        members = [np.where(gi == k)[0] for k in range(gi.max() + 1)]; net = x.net.to_numpy()
        bs = [net[np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])].mean() for _ in range(300)]
        base = {"long": drift, "short": -drift, "LS": 0.0}[name] - 2 * cost   # 무작위 분위로 같은 매매를 했을 때의 순수익
        r[name] = {"n": len(x), "net_bp": float(x.net.mean() * 1e4), "net_lo_bp": float(np.quantile(bs, 0.025) * 1e4), "net_hi_bp": float(np.quantile(bs, 0.975) * 1e4),
                   "excess_bp": float((x.net.mean() - base) * 1e4), "hit": float((x.net > 0).mean()), "K": K, "days": int(len(daily)),
                   "ann_ret_pct": float(daily.mean() * 365 * 100), "sharpe": float(daily.mean() / (daily.std() + 1e-12) * np.sqrt(365)), "mdd_pct": mdd * 100}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True); ap.add_argument("--res", required=True); ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--cost", type=float, default=0.001, help="편도 비용(0.001 = 0.1%: 수수료 0.05%+슬리피지 0.05%)")
    ap.add_argument("--q", type=float, default=0.1)
    a = ap.parse_args()
    t = trades(a.pred, a.res, a.H)
    u = pd.read_csv("work/universe.csv"); hold = set(u[u.holdout].base)
    bpd = 1440 // B.RES_MIN[a.res]
    res = {"all": summarize(t, a.cost, a.q, bpd), "unseen": summarize(t[t.base.isin(hold)], a.cost, a.q, bpd), "n_rows": len(t),
           "period": [str(pd.to_datetime(t.ts.min(), unit="ms").date()), str(pd.to_datetime(t.ts.max(), unit="ms").date())]}
    for part in ("all", "unseen"):
        print(f"{part}: drift {res[part]['drift_bp']:+.1f}bp/trade (무작위 분위 기준)")
        for k, v in res[part].items():
            if k == "drift_bp":
                continue
            print(f"{part:6s} {k:5s} n={v['n']:6d} net={v['net_bp']:+6.1f}bp [{v['net_lo_bp']:+.1f},{v['net_hi_bp']:+.1f}] excess={v['excess_bp']:+6.1f}bp hit={v['hit']:.3f} K={v['K']:.0f} ann={v['ann_ret_pct']:+.0f}% sharpe={v['sharpe']:+.2f} mdd={v['mdd_pct']:+.1f}%")
    lo, hi = t.score.quantile([a.q, 1 - a.q])            # 월별 분해: 수익이 며칠에 몰렸는지
    S = t[t.score <= lo].assign(net=lambda x: -x.ret - 2 * a.cost); L = t[t.score >= hi].assign(net=lambda x: x.ret - 2 * a.cost)
    m = pd.DataFrame({"short_bp": S.groupby(pd.to_datetime(S.ts, unit="ms").dt.strftime("%Y-%m")).net.mean() * 1e4,
                      "long_bp": L.groupby(pd.to_datetime(L.ts, unit="ms").dt.strftime("%Y-%m")).net.mean() * 1e4,
                      "n_short": S.groupby(pd.to_datetime(S.ts, unit="ms").dt.strftime("%Y-%m")).size()})
    print(m.round(1).to_string()); res["monthly"] = m.round(1).to_dict()
    d5 = S.assign(day=pd.to_datetime(S.ts_exit, unit="ms").dt.date).groupby("day").net.sum().sort_values(ascending=False)
    print(f"short 총이익 중 상위 5일 비중 {d5.head(5).sum() / max(d5[d5 > 0].sum(), 1e-9):.2f}")
    out = a.pred.replace("_pred.npz", f"_bt_q{a.q}_c{a.cost}.json"); json.dump(res, open(out, "w"), indent=1, default=str); print("wrote", out)


if __name__ == "__main__":
    main()
