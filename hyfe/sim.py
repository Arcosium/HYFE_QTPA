"""자본 제약 포트폴리오 시뮬레이터(P6) — 신호 npz(ts, base, p) 를 시간순으로 돌며 공매도(점수 하위) 포지션을 K 개 상한으로 연다.
체결: 신호 봉 종가 확정 → 다음 봉 시가 진입 → H 봉 뒤 종가 청산. 편도 비용 cost. 펀딩비(work/funding.parquet)가 있으면 보유 중 8h 마다 반영(공매도는 +rate 수취).
자본은 현재 순자산을 K 로 나눠 배분(복리). 문턱은 --thr 지정 또는 신호 분포의 q 분위(ROS 자체 분위 — 문턱만 사후 정보라 주의).
usage: HYFE_BARS=work/bars_full python3 -m hyfe.sim --pred work/results/gbm_roll4_top200_15m_W240_H60_s1_pred.npz --res 15m --H 60 --K 20
"""
import argparse, json, os
import numpy as np
import pandas as pd
from hyfe import bars as B, metrics as M


def load_prices(bases, res):
    px = {}
    for b in bases:
        p = B.path(res, b)
        if os.path.exists(p):
            d = pd.read_parquet(p); px[b] = (d.ts.to_numpy(), d.o.to_numpy(), d.c.to_numpy(), d.v.to_numpy())
    return px


def run(pred, res, H, K, cost, q, thr, side="short", fund=None, delay=1, exclude=(), shuffle=0, no_compound=False, hedge=False, regime=0):
    z = np.load(pred, allow_pickle=True)
    sig = pd.DataFrame({"ts": z["ts"], "base": z["base"].astype(str), "score": M.score(z["p"])}).sort_values("ts")
    sig = sig[~sig.base.isin(set(exclude))]
    if shuffle:   # 대조: 점수를 무작위로 섞어 같은 종목군·같은 시각에 아무거나 공매도했을 때(시장 베타·구조 효과만 남는다)
        sig["score"] = np.random.default_rng(shuffle).permutation(sig.score.to_numpy())
    if thr is None:
        thr = float(sig.score.quantile(q if side == "short" else 1 - q))
    universe = sig.base.unique()
    if side in ("follow", "fade"):
        ev = pd.Series(z["p"][:, 1] + z["p"][:, 2], index=np.arange(len(z["ts"])))
        sig["ev"] = ev.reindex(sig.index).to_numpy(); sig = sig[~sig.base.isin(set(exclude))]
        if shuffle:   # 대조: 경보 자체를 무작위로
            sig["ev"] = np.random.default_rng(shuffle).permutation(sig.ev.to_numpy())
        thr = float(sig.ev.quantile(1 - q)); sig = sig[sig.ev >= thr].copy(); sig["sd"] = "?"
    elif side == "ls":
        thr_hi = float(sig.score.quantile(1 - q)); sig = sig[(sig.score <= thr) | (sig.score >= thr_hi)].copy()
        sig["sd"] = np.where(sig.score <= thr, "short", "long")
    else:
        sig = sig[sig.score <= thr] if side == "short" else sig[sig.score >= thr]; sig = sig.assign(sd=side)
    px = load_prices(sig.base.unique(), res)
    gate = None
    if regime:   # 체제 필터: BTC 직전 regime 일 수익률 < 0 일 때만 신규 공매도(롱 전략이면 > 0)
        tb, ob, cb, vb = load_prices(["BTC"], res)["BTC"]; bpd = 1440 // B.RES_MIN[res]
        trend = pd.Series(cb, index=tb); trend = trend / trend.shift(regime * bpd) - 1
        gate = trend
    idx = None
    if hedge:   # 유니버스 동일비중 지수(봉 수익률 평균의 누적) — 공매도와 같은 크기의 롱 헤지 → 시장 중립
        upx = load_prices(universe, res); frames = []
        for b, (t_, o_, c_, v_) in upx.items():
            r = pd.Series(c_, index=t_).pct_change(); frames.append(r.rename(b))
        R = pd.concat(frames, axis=1).sort_index(); R = R.loc[R.index >= sig.ts.min() - 200 * B.RES_MIN[res] * 60_000]
        idx = (1 + R.mean(axis=1).fillna(0.0)).cumprod()
    bar_ms = B.RES_MIN[res] * 60_000
    equity, cash_pnl = 1.0, 0.0
    open_pos = {}     # base -> (exit_ts, size, entry, signal_ts)
    curve, trades, cand, taken = [], [], 0, 0
    hedge_pos, hedge_prev_i, hedge_pnl, hedge_cost = 0.0, None, 0.0, 0.0     # 포트폴리오 단일 헤지(순노출만큼 지수 반대 포지션)
    pos_side = {}; cap = {"short": K // 2, "long": K - K // 2} if side == "ls" else ({"short": K, "long": K} if side in ("follow", "fade") else {side: K})
    for ts, g in sig.groupby("ts", sort=True):
        if idx is not None:
            i = idx.index.searchsorted(ts)
            if hedge_prev_i is not None and i < len(idx) and i > hedge_prev_i:
                d = hedge_pos * float(idx.iloc[i] / idx.iloc[hedge_prev_i] - 1); equity += d; hedge_pnl += d
            hedge_prev_i = min(i, len(idx) - 1)
        # 만기 도래 포지션 청산
        for b in [b for b, v in open_pos.items() if v[0] <= ts]:
            exit_ts, size, entry, sts = open_pos.pop(b)
            t_, o_, c_, v_ = px[b]; i = np.searchsorted(t_, exit_ts)
            if i >= len(t_) or t_[i] != exit_ts:
                continue
            sd = pos_side.pop(b, side)
            ret = (1 - c_[i] / entry) if sd == "short" else (c_[i] / entry - 1)   # 공매도 손익 = (진입−청산)/진입. entry/exit−1 은 큰 하락을 부풀린다(버그였음)
            fr = 0.0
            if fund is not None and b in fund:
                ft, fv = fund[b]; m = (ft > sts) & (ft <= exit_ts); fr = float(fv[m].sum()) * (1 if sd == "short" else -1)
            pnl = size * (ret - 2 * cost + fr); equity += pnl
            trades.append({"base": b, "ts": sts, "exit": exit_ts, "ret": ret, "fund": fr, "pnl": pnl})
        curve.append((ts, equity, len(open_pos)))
        if side in ("follow", "fade"):   # 경보 직후 봉의 방향으로(또는 반대로) 진입, 진입 봉은 e+2
            sds = []
            for _, r in g.iterrows():
                t_, o_, c_, v_ = px.get(r.base, (None,) * 4)
                if t_ is None:
                    sds.append(None); continue
                i = np.searchsorted(t_, r.ts)
                if i + 2 >= len(t_) or t_[i] != r.ts:
                    sds.append(None); continue
                d1 = c_[i + 1] - o_[i + 1]; sd = ("long" if d1 > 0 else "short") if side == "follow" else ("short" if d1 > 0 else "long")
                sds.append(sd if d1 != 0 else None)
            g = g.assign(sd=sds).dropna(subset=["sd"]).sort_values("ev", ascending=False)
        g = pd.concat([g[g.sd == "short"].sort_values("score"), g[g.sd == "long"].sort_values("score", ascending=False)]) if side not in ("follow", "fade") else g
        if gate is not None:
            j = gate.index.searchsorted(ts) - 1; tr_now = float(gate.iloc[j]) if j >= 0 else 0.0
            g = g[(g.sd == "short") & (tr_now < 0) | (g.sd == "long") & (tr_now > 0)]
        cand += len(g)
        for _, r in g.iterrows():
            n_sd = sum(1 for v in pos_side.values() if v == r.sd)
            if n_sd >= cap[r.sd]:
                continue
            if r.base in open_pos or r.base not in px:
                continue
            t_, o_, c_, v_ = px[r.base]; i = np.searchsorted(t_, r.ts)
            dl = delay + (1 if side in ("follow", "fade") else 0)     # 추종·역추세는 확인 봉 하나 뒤에 진입
            if i + dl >= len(t_) or t_[i] != r.ts or v_[i + dl] <= 0:   # 진입 봉이 무거래(보간) 봉이면 체결 불가로 본다
                continue
            open_pos[r.base] = (r.ts + (H + dl - 1) * bar_ms, (1.0 if no_compound else equity) / K, o_[i + dl], r.ts); pos_side[r.base] = r.sd; taken += 1
        if idx is not None:   # 순공매도 노출만큼 지수 롱(롱 전략이면 숏) 유지, 변화분에만 편도 비용
            target = sum(v[1] * (1 if pos_side.get(b, side) == "short" else -1) for b, v in open_pos.items())
            c = abs(target - hedge_pos) * cost; equity -= c; hedge_cost += c; hedge_pos = target
    cv = pd.DataFrame(curve, columns=["ts", "equity", "n_open"])
    daily = cv.groupby(pd.to_datetime(cv.ts, unit="ms").dt.floor("D")).equity.last()
    dr = daily.pct_change().dropna()
    mdd = float((daily / daily.cummax() - 1).min())
    tr = pd.DataFrame(trades)
    days = max((daily.index[-1] - daily.index[0]).days, 1)
    return {"final_equity": float(equity), "ann_ret_pct": float((equity ** (365 / days) - 1) * 100), "sharpe": float(dr.mean() / (dr.std() + 1e-12) * np.sqrt(365)),
            "mdd_pct": mdd * 100, "trades": len(tr), "candidates": cand, "fill": taken / max(cand, 1), "avg_open": float(cv.n_open.mean()),
            "hit": float((tr.pnl > 0).mean()) if len(tr) else None, "avg_trade_bp": float(tr.pnl.mean() / (1 / K) * 1e4) if len(tr) else None,
            "fund_bp_per_trade": float(tr.fund.mean() * 1e4) if len(tr) else None, "hedge_pnl": float(hedge_pnl), "hedge_cost": float(hedge_cost),
            "days": days, "thr": thr}, daily


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True); ap.add_argument("--res", required=True); ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--K", type=int, default=20); ap.add_argument("--cost", type=float, default=0.001); ap.add_argument("--q", type=float, default=0.1)
    ap.add_argument("--thr", type=float); ap.add_argument("--side", default="short", choices=["short", "long", "ls", "follow", "fade"]); ap.add_argument("--no_fund", action="store_true"); ap.add_argument("--delay", type=int, default=1, help="진입 봉 지연(1=다음 봉 시가, 2=한 봉 더 늦게)"); ap.add_argument("--exclude_source", default="", help="예: bybit_trades — 체결 집계 상폐분 제외 민감도"); ap.add_argument("--shuffle", type=int, default=0, help="0 이 아니면 그 시드로 점수를 섞은 무작위 대조"); ap.add_argument("--no_compound", action="store_true"); ap.add_argument("--only", default="", help="매매 후보를 이 파일의 종목(한 줄 하나)으로 제한"); ap.add_argument("--hedge", action="store_true", help="같은 크기의 유니버스 지수 반대 포지션(시장 중립)"); ap.add_argument("--regime", type=int, default=0, help="N>0: BTC 직전 N일 추세가 하락일 때만 공매도(롱은 상승일 때만)")
    a = ap.parse_args()
    fund = None
    if not a.no_fund and os.path.exists("work/funding.parquet"):
        f = pd.read_parquet("work/funding.parquet"); fund = {b: (g.ts.to_numpy(), g.rate.to_numpy()) for b, g in f.sort_values("ts").groupby("base")}
    u = pd.read_csv("work/universe.csv"); excl = u[u.source == a.exclude_source].base.tolist() if a.exclude_source else []
    if a.only:
        keep = set(l.strip() for l in open(a.only) if l.strip()); excl = sorted(set(u.base) - keep)
    r, daily = run(a.pred, a.res, a.H, a.K, a.cost, a.q, a.thr, a.side, fund, a.delay, excl, a.shuffle, a.no_compound, a.hedge, a.regime)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, ensure_ascii=False))
    out = a.pred.replace("_pred.npz", f"_sim_{a.side}_K{a.K}_c{a.cost}" + ("_hedge" if a.hedge else "") + (f"_reg{a.regime}" if a.regime else "") + (f"_shuf{a.shuffle}" if a.shuffle else "") + ".json")
    json.dump({"summary": r, "daily": {str(k.date()): float(v) for k, v in daily.items()}}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
