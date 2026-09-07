"""정본 종목표 → work/universe.csv
coverage(출처·상폐) + binance exchangeInfo(토큰화 주식·원자재 분류) + ADV(유동성) 를 합쳐
제외 사유(exclude), 유동성 순위(liq_rank), 미학습 홀드아웃(holdout, 심볼 해시 20%) 을 붙인다.
usage: python3 -m hyfe.universe
"""
import hashlib, json
import pandas as pd

GOLD = {"PAXG", "XAUT"}  # 금 연동 토큰: COIN 으로 분류되지만 원자재 취급


def main():
    cov = pd.read_csv("data/coverage.csv")
    adv = pd.read_csv("work/adv.csv")
    info = json.load(open("work/binance_exchangeinfo.json"))
    kind = {s["baseAsset"]: s["underlyingType"] for s in info}
    u = adv.merge(cov[["base", "source", "first_utc", "last_utc", "delisted"]], on="base", how="left")
    u["source"] = u.source.fillna("binance")
    u["delisted"] = u.delisted.fillna(0).astype(int)
    u["kind"] = u.base.map(kind).fillna("COIN")
    u["exclude"] = ""
    u.loc[u.kind != "COIN", "exclude"] = "tradfi:" + u.kind
    u.loc[u.base.isin(GOLD), "exclude"] = "gold-token"
    u.loc[(u.exclude == "") & (u.active_days < 90), "exclude"] = "short<90d"
    u["holdout"] = u.base.map(lambda b: int(hashlib.sha1(b.encode()).hexdigest(), 16) % 5 == 0)
    ok = u.exclude == ""
    u["liq_rank"] = 0
    u.loc[ok, "liq_rank"] = u[ok].adv_usd.rank(ascending=False, method="first").astype(int)
    u = u.sort_values(["exclude", "liq_rank"])
    u.to_csv("work/universe.csv", index=False)
    print(len(u), "bases;", ok.sum(), "usable;", u[ok].holdout.sum(), "holdout")
    print(u.exclude.value_counts().to_string())
    print(u[ok].head(15)[["base", "adv_usd", "active_days", "source", "holdout", "liq_rank"]].to_string())


if __name__ == "__main__":
    main()
