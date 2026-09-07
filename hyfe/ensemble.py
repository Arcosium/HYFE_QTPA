"""같은 지평(15h)의 예측 npz 여러 개를 **판단 시각(봉 마감 = ts + 봉 길이)** 기준으로 (base, 판단시각) inner join 해 확률 평균 → 앙상블 npz.
ts 로 맞추면 1h 봉은 15m 봉보다 45분 늦게 닫혀 미래 정보가 섞이므로 반드시 판단 시각으로 맞춘다. 출력 ts 는 첫 파일(15m) 의 ts.
파일명에서 해상도(_15m_/_1h_/_5m_)를 읽는다. usage: python3 -m hyfe.ensemble --out ens_s0_pred.npz a_15m_pred.npz b_15m_pred.npz c_1h_pred.npz
"""
import argparse, re
import numpy as np
import pandas as pd
from hyfe import bars as B


def load(p, n):
    z = np.load(p, allow_pickle=True)
    res = re.search(r"_(1m|5m|15m|1h|4h|1d)_", p).group(1); bar = B.RES_MIN[res] * 60_000
    d = pd.DataFrame({"ts": z["ts"], "dt": z["ts"] + bar, "base": z["base"].astype(str), "y": z["y"], "fwd": z["fwd"], "sigH": z["sigH"]})
    for k in range(3):
        d[f"p{k}_{n}"] = z["p"][:, k]
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("preds", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = load(a.preds[0], 0); n = 1
    for p in a.preds[1:]:
        d = load(p, n).drop(columns=["ts", "y", "fwd", "sigH"]); base = base.merge(d, on=["dt", "base"], how="inner"); n += 1
    prob = np.mean([base[[f"p{k}_{i}" for k in range(3)]].to_numpy() for i in range(n)], axis=0)
    np.savez_compressed(a.out, ts=base.ts.to_numpy(), base=base.base.to_numpy().astype(str), y=base.y.to_numpy(), p=prob, fwd=base.fwd.to_numpy(), sigH=base.sigH.to_numpy())
    print("rows", len(base), "models", n, "->", a.out)


if __name__ == "__main__":
    main()
