"""종목별 평균 일 거래대금(ADV, USD) — 유동성 순위용. 지정 기간의 월 파일만 열어 quote_volume(없으면 close*volume) 합산.
usage: python3 -m hyfe.adv --start 2025-07 --end 2026-06 --out work/adv.csv
"""
import argparse, os, glob
from multiprocessing import Pool
import pandas as pd
import pyarrow.parquet as pq

HIST = "/home/arcosium/projects/CryptoBars/data/history"


def one(args):
    base, months = args
    usd = 0.0; days = set()
    for m in months:
        f = f"{HIST}/base={base}/part-{m}.parquet"
        if not os.path.exists(f):
            continue
        t = pq.read_table(f, columns=["ts", "close", "volume", "quote_volume"]).to_pandas()
        qv = t.quote_volume.fillna(t.close * t.volume)
        usd += float(qv.sum())
        days |= set((t.ts // 86_400_000).unique())
    return base, usd / max(len(days), 1), len(days)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-07"); ap.add_argument("--end", default="2026-06")
    ap.add_argument("--out", default="work/adv.csv"); ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    months = [m.strftime("%Y-%m") for m in pd.period_range(a.start, a.end, freq="M")]
    bases = sorted(d.split("base=")[1] for d in glob.glob(f"{HIST}/base=*"))
    with Pool(a.procs) as p:
        rows = p.map(one, [(b, months) for b in bases], chunksize=8)
    pd.DataFrame(rows, columns=["base", "adv_usd", "active_days"]).sort_values("adv_usd", ascending=False).to_csv(a.out, index=False)
    print("wrote", a.out, len(rows))


if __name__ == "__main__":
    main()
