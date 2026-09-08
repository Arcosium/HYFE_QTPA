"""성과 판정표 — Sharpe(연율, 일별 수익)·유의확률(Newey-West HAC t, H0: 평균 일수익 ≤ 0)·MDD(보조). sim 결과 json 의 daily 순자산을 쓴다.
usage: python3 -m hyfe.perf --stem work/results/ens_v2_4h --side ls --K 100 --splits 0,1,2,3 --lag 14 [--shuf]
  파일 = <stem>_s<i>_sim_<side>_K<K>_c0.001.json (셔플 대조는 _shuf1/_shuf2). --lag 는 보유일(자기상관 창).
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy.stats import norm


def daily_returns(path):
    d = json.load(open(path))["daily"]; eq = pd.Series(d, dtype=float); eq.index = pd.to_datetime(eq.index)
    return eq.pct_change().dropna(), eq


def nw_pvalue(r, lag):
    """평균 > 0 검정. Newey-West(Bartlett) 분산으로 자기상관 보정, 단측 p."""
    r = np.asarray(r, float); n = len(r); m = r.mean(); e = r - m
    s2 = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        s2 += 2 * (1 - k / (lag + 1)) * (e[:-k] @ e[k:]) / n
    se = np.sqrt(max(s2, 1e-18) / n); t = m / se
    return float(t), float(1 - norm.cdf(t))


def stats(r, eq, lag):
    sh = float(r.mean() / (r.std() + 1e-12) * np.sqrt(365)); t, p = nw_pvalue(r, lag)
    mdd = float((eq / eq.cummax() - 1).min() * 100)
    return dict(sharpe=round(sh, 2), t=round(t, 2), p=round(p, 4), mdd=round(mdd, 1), final=round(float(eq.iloc[-1]), 4), days=len(r))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stem", required=True); ap.add_argument("--side", default="ls"); ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--cost", default="0.001"); ap.add_argument("--splits", default="0,1,2,3"); ap.add_argument("--lag", type=int, default=7); ap.add_argument("--shuf", action="store_true"); ap.add_argument("--label", default=""); ap.add_argument("--suffix", default="", help="예: _xs (횡단면 경계 시뮬)")
    a = ap.parse_args(); rows = []; pooled = []
    for s in a.splits.split(","):
        base = f"{a.stem}_s{s}_sim_{a.side}_K{a.K}_c{a.cost}{a.suffix}"
        variants = [("실제", base + ".json")] + ([(f"셔플{k}", f"{base}_shuf{k}.json") for k in (1, 2)] if a.shuf else [])
        for name, path in variants:
            if not os.path.exists(path):
                continue
            r, eq = daily_returns(path); st = stats(r, eq, a.lag); rows.append(dict(fold=f"s{s}", run=name, **st))
            if name == "실제":
                pooled.append(r)
    if len(pooled) > 1:
        r = pd.concat(pooled); eq = (1 + r).cumprod(); rows.append(dict(fold="합산", run="실제", **stats(r, eq, a.lag)))
    df = pd.DataFrame(rows); print(a.label or a.stem); print(df.to_string(index=False))


if __name__ == "__main__":
    main()
