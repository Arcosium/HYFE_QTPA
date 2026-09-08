"""두 전략 장부의 자본 혼합(일별 수익 가중 평균) 성과 — 상관·Sharpe·NW p·MDD. 폴드별 + 합산.
usage: python3 -m hyfe.mix --a work/results/ens_v1os_4h --b work/results/ens_v2os_4h --suffix _xs --splits 0,1,2,3 --w 0.5 --lag 14
"""
import argparse, os
import numpy as np
import pandas as pd
from hyfe.perf import daily_returns, stats


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--suffix", default="")
    ap.add_argument("--splits", default="0,1,2,3"); ap.add_argument("--w", type=float, default=0.5, help="a 의 비중"); ap.add_argument("--lag", type=int, default=14); ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--files", nargs="*", default=[], help="직접 지정: a.json b.json [a2.json b2.json ...] (코호트 json 등, 두 개씩 짝)")
    a = ap.parse_args(); rows = []; pooled = {"a": [], "b": [], "mix": []}
    pairs = [(a.files[i], a.files[i + 1], f"p{i // 2}") for i in range(0, len(a.files) - 1, 2)] if a.files else \
        [(f"{a.a}_s{s}_sim_ls_K{a.K}_c0.001{a.suffix}.json", f"{a.b}_s{s}_sim_ls_K{a.K}_c0.001{a.suffix}.json", f"s{s}") for s in a.splits.split(",")]
    for pa, pb, s in pairs:
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        ra, _ = daily_returns(pa); rb, _ = daily_returns(pb)
        idx = ra.index.union(rb.index); ra = ra.reindex(idx).fillna(0.0); rb = rb.reindex(idx).fillna(0.0)
        rm = a.w * ra + (1 - a.w) * rb; corr = float(np.corrcoef(ra, rb)[0, 1]) if len(ra) > 2 else float("nan")
        for k, r in (("a", ra), ("b", rb), ("mix", rm)):
            st = stats(r, (1 + r).cumprod(), a.lag); rows.append(dict(fold=s, run=k, corr=round(corr, 2) if k == "mix" else "", **st)); pooled[k].append(r)
    for k in ("a", "b", "mix"):
        if len(pooled[k]) > 1:
            r = pd.concat(pooled[k]); rows.append(dict(fold="합산", run=k, corr="", **stats(r, (1 + r).cumprod(), a.lag)))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
