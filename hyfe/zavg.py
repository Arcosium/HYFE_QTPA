"""여러 예측(_pred.npz, 같은 폴드)의 방향 점수를 z 평균해 하나의 예측 파일로 — 앙상블 점수를 sim/backtest/blend 에 그대로 넣기 위해.
usage: python3 -m hyfe.zavg --out work/results/ens_g60g120_4h_s0_pred.npz a_s0_pred.npz b_s0_pred.npz [--w 0.5,0.5]
점수는 p[:,1]−p[:,2] 로 두고 fwd·sigH·y 는 첫 파일 것을 쓴다(교집합 행만)."""
import argparse
import numpy as np
import pandas as pd
from hyfe import metrics as M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("preds", nargs="+"); ap.add_argument("--out", required=True); ap.add_argument("--w", default="")
    a = ap.parse_args(); w = [float(x) for x in a.w.split(",")] if a.w else [1.0] * len(a.preds)
    d = None
    for i, p in enumerate(a.preds):
        z = np.load(p, allow_pickle=True); s = z["p"][:, 1] - z["p"][:, 2]
        t = pd.DataFrame({"ts": z["ts"], "base": z["base"].astype(str), f"s{i}": (s - s.mean()) / (s.std() + 1e-12)})
        if i == 0:
            t["y"] = z["y"]; t["fwd"] = z["fwd"]; t["sigH"] = z["sigH"]
        d = t if d is None else d.merge(t, on=["ts", "base"])
    s = sum(w[i] * d[f"s{i}"] for i in range(len(a.preds))) / sum(w)
    p = np.c_[np.zeros(len(d)), s.clip(lower=0), (-s).clip(lower=0)]
    np.savez_compressed(a.out, ts=d.ts.to_numpy(), base=d.base.to_numpy().astype(str), y=d.y.to_numpy(), p=p, fwd=d.fwd.to_numpy(), sigH=d.sigH.to_numpy())
    r = M.evaluate(d.y.to_numpy(), p, d.fwd.to_numpy(), d.sigH.to_numpy())
    print(f"{a.out} rows {len(d)} spread_z {r['spread_z']:.3f} hit {r['hit_top']:.3f}/{r['hit_bot']:.3f}")


if __name__ == "__main__":
    main()
