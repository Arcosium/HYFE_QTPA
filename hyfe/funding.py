"""Binance USDⓈ-M 펀딩비 이력 수집 → work/funding.parquet (symbol, ts, rate). 공매도 백테스트의 펀딩 손익 반영용.
usage: python3 -m hyfe.funding --top 200 --start 2024-05-01
"""
import argparse, json, os, time, urllib.request
import pandas as pd

API = "https://fapi.binance.com/fapi/v1/fundingRate?symbol={s}&startTime={t}&limit=1000"


def fetch(symbol, start_ms):
    out, t = [], start_ms
    while True:
        try:
            rows = json.loads(urllib.request.urlopen(API.format(s=symbol, t=t), timeout=30).read())
        except Exception as e:
            print("err", symbol, str(e)[:80]); break
        if not rows:
            break
        out += rows
        if len(rows) < 1000:
            break
        t = rows[-1]["fundingTime"] + 1
        time.sleep(0.1)
    return [(symbol, r["fundingTime"], float(r["fundingRate"])) for r in out]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--top", type=int, default=200); ap.add_argument("--start", default="2024-05-01")
    a = ap.parse_args()
    u = pd.read_csv("work/universe.csv"); u = u[(u.exclude.fillna("") == "") & (u.liq_rank <= a.top)]
    cov = pd.read_csv("data/coverage.csv")[["base", "symbol"]]
    u = u.merge(cov, on="base", how="left")
    start = int(pd.Timestamp(a.start).value // 1_000_000)
    rows = []
    for _, r in u.iterrows():
        sym = r.symbol if isinstance(r.symbol, str) else r.base + "USDT"
        if r.source == "bybit_trades" or r.source == "bybit":
            continue   # 바이낸스 외 거래소는 생략(펀딩 체계 다름)
        rows += fetch(sym, start)
    df = pd.DataFrame(rows, columns=["symbol", "ts", "rate"]); df["base"] = df.symbol.str.replace(r"(USDT|USDC|BUSD|USD)$", "", regex=True)
    df.to_parquet("work/funding.parquet", index=False)
    print(len(df), "rows,", df.base.nunique(), "bases; mean 8h rate", round(df.rate.mean() * 1e4, 2), "bp")


if __name__ == "__main__":
    main()
