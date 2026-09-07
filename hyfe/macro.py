"""시장 매크로 표 — 해상도별 ts 마다 BTC 24h·7d 수익률과 7d 변동성, 시장 폭(상위 200 중 24h 양수 비율), 유니버스 24h 중앙 수익, 펀딩(BTC 8h).
GBM 피처(--macro)와 CNN 채널(macro 평면)이 같은 표를 쓴다. 전부 판단 시각 이전 값만.
usage: HYFE_BARS=work/bars_full python3 -m hyfe.macro --res 15m,1h,4h,1d --top 200 → work/macro_<res>.parquet
"""
import argparse, os
import numpy as np
import pandas as pd
from hyfe import bars as B

MACRO = ["btc_r24", "btc_r7d", "btc_vol7d", "breadth24", "univ_med24", "fund_btc"]


def build(res, bases):
    bpd = 1440 // B.RES_MIN[res]
    btc = pd.read_parquet(B.path(res, "BTC")).set_index("ts").c
    lb = np.log(btc)
    m = pd.DataFrame({"btc_r24": lb - lb.shift(bpd), "btc_r7d": lb - lb.shift(7 * bpd), "btc_vol7d": lb.diff().rolling(7 * bpd).std() * np.sqrt(bpd)})
    r24 = {}
    for b in bases:
        p = B.path(res, b)
        if os.path.exists(p):
            c = pd.read_parquet(p, columns=["ts", "c"]).set_index("ts").c; l = np.log(c.clip(lower=1e-12)); r24[b] = l - l.shift(bpd)
    R = pd.DataFrame(r24).reindex(m.index)
    m["breadth24"] = (R > 0).sum(axis=1) / R.notna().sum(axis=1).clip(lower=1); m["univ_med24"] = R.median(axis=1)
    fp = "work/funding.parquet"
    if os.path.exists(fp):
        f = pd.read_parquet(fp); f = f[f.base == "BTC"].sort_values("ts")
        m["fund_btc"] = pd.Series(f.rate.to_numpy(), index=f.ts.to_numpy()).reindex(m.index, method="ffill").fillna(0.0)
    else:
        m["fund_btc"] = 0.0
    return m.reset_index()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="15m,1h,4h,1d"); ap.add_argument("--top", type=int, default=200)
    a = ap.parse_args()
    u = pd.read_csv("work/universe.csv"); u = u[(u.exclude.fillna("") == "") & (u.liq_rank <= a.top)]
    for res in a.res.split(","):
        m = build(res, u.base.tolist()); m.to_parquet(f"work/macro_{res}.parquet", index=False)
        print(res, len(m), m[MACRO].describe().loc[["mean", "std"]].round(4).to_dict())


if __name__ == "__main__":
    main()
