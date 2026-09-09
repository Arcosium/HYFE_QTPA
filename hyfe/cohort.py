"""겹침 코호트 롱숏 포트폴리오(Jegadeesh–Titman 식) — 슬롯 표본 잡음이 없는 판정용.
매 판단 시각 t 에 횡단면 상·하위 q 를 롱·숏 코호트로 만들고(각 다리 동일비중, 달러 중립), 다음 봉부터 H 봉 보유.
포트폴리오 봉수익 = 활성 코호트들의 평균 수익. 비용은 코호트 진입·청산 왕복 2×cost 를 보유 봉에 나눠 차감. 펀딩 미반영.
usage: python3 -m hyfe.cohort --pred work/results/x_pred.npz --res 4h --H 84 [--q 0.1] [--shuffle 1] → 일별 순자산 json(<stem>_cohort_H<H>[_shufN].json) + Sharpe·NW p·MDD
"""
import argparse, json
import numpy as np
import pandas as pd
from hyfe import bars as B, metrics as M
from hyfe.perf import stats


def run(pred, res, H, q=0.1, cost=0.001, shuffle=0, long_only=False, funding=""):
    z = np.load(pred, allow_pickle=True)
    sig = pd.DataFrame({"ts": z["ts"], "base": z["base"].astype(str), "s": M.score(z["p"])})
    if shuffle:
        sig["s"] = np.random.default_rng(shuffle).permutation(sig.s.to_numpy())
    bases = sorted(sig.base.unique()); bar_ms = B.RES_MIN[res] * 60_000
    t0, t1 = sig.ts.min(), sig.ts.max() + (H + 2) * bar_ms
    px = {}
    for b in bases:
        d = pd.read_parquet(B.path(res, b), columns=["ts", "c"]); d = d[(d.ts >= t0 - bar_ms) & (d.ts <= t1)]
        px[b] = d.set_index("ts").c
    P = pd.DataFrame(px).sort_index(); grid = P.index.to_numpy(); pos = {t: i for i, t in enumerate(grid)}; A = P.to_numpy(); col = {b: j for j, b in enumerate(bases)}
    Fm = None
    if funding:   # 펀딩비(8h 정산) → 격자 봉별 합. 롱은 rate 를 내고 숏은 받는다. 자료 없는 종목(bybit 전용 등)은 같은 다리의 평균 펀딩을 적용(nanmean)
        fd = pd.read_parquet(funding); fd = fd[fd.base.isin(bases) & (fd.ts >= grid[0]) & (fd.ts <= grid[-1])]
        Fm = np.full((len(grid), len(bases)), np.nan)
        for b, gb in fd.groupby("base"):
            colv = np.zeros(len(grid)); np.add.at(colv, np.searchsorted(grid, gb.ts.to_numpy(), side="left").clip(0, len(grid) - 1), gb.rate.to_numpy()); Fm[:, col[b]] = colv
        run.funding_cov = int(fd.base.nunique()); run.funding_paid = []
    pnl = np.zeros(len(grid)); nact = np.zeros(len(grid))                       # 코호트 손익 증분 합, 활성 코호트 수
    pr = sig.groupby("ts").s.rank(pct=True)
    for t, g in sig.assign(pr=pr).groupby("ts"):
        if t not in pos:
            continue
        L = [col[b] for b in g[g.pr >= 1 - q].base]; S = [col[b] for b in g[g.pr <= q].base]
        i0 = pos[t]; i1 = min(i0 + H, len(grid) - 1)
        if not L or not S or i1 <= i0:
            continue
        base_px = A[i0]; V = A[i0:i1 + 1] / base_px                                     # 진입(판단봉 종가) 대비 가치 경로, 보유 중 매수 후 보유
        vl = np.nanmean(V[:, L], axis=1); vs = np.nanmean(V[:, S], axis=1)
        if long_only:   # 롱 다리 초과수익: 상위 q 동일비중 − 유니버스 동일비중(같은 시각 전 종목), 비용은 롱 다리 왕복만
            vu = np.nanmean(V, axis=1); path = (vl - 1) - (vu - 1) - 2 * cost * np.linspace(0, 1, len(vl))
        else:
            path = 0.5 * (vl - 1) - 0.5 * (vs - 1) - 2 * cost * np.linspace(0, 1, len(vl))   # 달러 중립 코호트 손익(왕복 비용은 보유 중 선형 차감)
        if Fm is not None:   # 보유 중 정산된 펀딩: 롱 −, 숏 + (봉 단위 누적, 다리 안 동일비중 평균)
            with np.errstate(all="ignore"):
                fl = np.nan_to_num(np.nanmean(Fm[i0 + 1:i1 + 1][:, L], axis=1)); fs = np.nan_to_num(np.nanmean(Fm[i0 + 1:i1 + 1][:, S], axis=1))
            fl, fs = np.r_[0.0, np.cumsum(fl)], np.r_[0.0, np.cumsum(fs)]
            path = path - fl if long_only else path - 0.5 * fl + 0.5 * fs
            run.funding_paid.append((fl[-1], fs[-1]))
        pnl[i0 + 1:i1 + 1] += np.diff(path); nact[i0 + 1:i1 + 1] += 1
    act = nact > 0; r = np.where(act, pnl / np.maximum(nact, 1), 0.0)             # 활성 코호트 동일 자본 배분 → 포트폴리오 봉수익
    eq = pd.Series(np.cumprod(1 + r), index=pd.to_datetime(grid, unit="ms"))
    daily = eq.groupby(eq.index.floor("D")).last(); dr = daily.pct_change().dropna()
    return dr, daily


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pred", required=True); ap.add_argument("--res", default="4h"); ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--q", type=float, default=0.1); ap.add_argument("--cost", type=float, default=0.001); ap.add_argument("--shuffle", type=int, default=0); ap.add_argument("--lag", type=int, default=0); ap.add_argument("--long_only", action="store_true", help="롱 다리만: 상위 q 매수 − 유니버스 동일비중(초과수익). 롱온리 계좌(타임폴리오) 판정용")
    ap.add_argument("--funding", default="", help="펀딩비 parquet(symbol,ts,rate,base) — 보유 중 정산 펀딩을 롱 −/숏 + 로 반영. 결과 파일 접미 _fund")
    a = ap.parse_args(); dr, daily = run(a.pred, a.res, a.H, a.q, a.cost, a.shuffle, a.long_only, a.funding)
    lag = a.lag or max(1, a.H * B.RES_MIN[a.res] // 1440)
    st = stats(dr, daily, lag); out = a.pred.replace("_pred.npz", f"_cohort_H{a.H}" + ("_long" if a.long_only else "") + (f"_shuf{a.shuffle}" if a.shuffle else "") + ("_fund" if a.funding else "") + ".json")
    if a.funding:
        fp = np.array(run.funding_paid); st["funding"] = {"covered_bases": run.funding_cov, "long_paid_bp_per_hold": round(float(fp[:, 0].mean() * 1e4), 2), "short_received_bp_per_hold": round(float(fp[:, 1].mean() * 1e4), 2)}
    json.dump({"summary": st, "daily": {str(k.date()): float(v) for k, v in daily.items()}}, open(out, "w"), indent=1)
    print(json.dumps(st), out)


if __name__ == "__main__":
    main()
