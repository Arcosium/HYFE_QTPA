"""1분봉 → 균일 격자 → 해상도별 봉 캐시 (work/bars/<res>/<base>.parquet)
결측 분은 종가 forward-fill, o/h/l=종가, 거래량 0. 격자는 첫 실봉~마지막 실봉 사이만.
usage: python3 -m hyfe.bars --bases BTC,ETH --start 2025-01 --end 2026-02
"""
import argparse, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HIST = "/home/arcosium/projects/CryptoBars/data/history"
OUT = os.environ.get("HYFE_BARS", "work/bars")
RES_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
MS = 60_000


def load_1m(base, start, end):
    months = [m.strftime("%Y-%m") for m in pd.period_range(start, end, freq="M")]
    parts = [f"{HIST}/base={base}/part-{m}.parquet" for m in months]
    parts = [p for p in parts if os.path.exists(p)]
    if not parts:
        return None
    df = pd.concat(pq.read_table(p, columns=["ts", "open", "high", "low", "close", "volume", "quote_volume"]).to_pandas() for p in parts)
    df = df.drop_duplicates("ts").sort_values("ts")
    df["quote_volume"] = df.quote_volume.fillna(df.close * df.volume)
    grid = np.arange(df.ts.iloc[0], df.ts.iloc[-1] + MS, MS)
    df = df.set_index("ts").reindex(grid)
    df["close"] = df.close.ffill()
    for c in ("open", "high", "low"):
        df[c] = df[c].fillna(df.close)
    df[["volume", "quote_volume"]] = df[["volume", "quote_volume"]].fillna(0.0)
    df.index.name = "ts"
    return df.reset_index()


def to_res(g, m):
    """균일 1분 격자 g 를 m분 봉으로. 경계(ts % m == 0)에 맞춰 앞을 자르고 reshape."""
    if m == 1:
        return g.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v", "quote_volume": "qv"})
    step = m * MS
    off = int(((-g.ts.iloc[0]) % step) // MS)
    n = (len(g) - off) // m
    if n <= 0:
        return None
    a = g.iloc[off:off + n * m]
    r = lambda col: a[col].to_numpy().reshape(n, m)
    return pd.DataFrame({
        "ts": a.ts.to_numpy()[::m], "o": r("open")[:, 0], "h": r("high").max(1), "l": r("low").min(1),
        "c": r("close")[:, -1], "v": r("volume").sum(1), "qv": r("quote_volume").sum(1)})


def path(res, base):
    return f"{OUT}/{res}/{base}.parquet"


def build(bases, start, end, res_list=tuple(RES_MIN)):
    for base in bases:
        if all(os.path.exists(path(r, base)) for r in res_list):
            continue
        g = load_1m(base, start, end)
        if g is None:
            print("no data", base); continue
        for res in res_list:
            b = to_res(g, RES_MIN[res])
            if b is None:
                continue
            os.makedirs(f"{OUT}/{res}", exist_ok=True)
            b.to_parquet(path(res, base), index=False)
        print(base, len(g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", required=True, help="쉼표 구분 또는 @파일(한 줄에 하나)"); ap.add_argument("--start", default="2023-01"); ap.add_argument("--end", default="2026-02")
    ap.add_argument("--res", default=",".join(RES_MIN))
    a = ap.parse_args()
    bases = [l.strip() for l in open(a.bases[1:]) if l.strip()] if a.bases.startswith("@") else a.bases.split(",")
    build(bases, a.start, a.end, res_list=a.res.split(","))


if __name__ == "__main__":
    main()
