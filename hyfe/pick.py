"""S1 결과에서 승자 설정을 고른다(검증 PR-AUC 평균, i1·비셔플). 고정% 라벨 문턱도 여기서 계산.
usage: python3 -m hyfe.pick            → "15m:120:10"
       python3 -m hyfe.pick fixed 15m 120 10 → 3×중앙값|fwd| (상위 20종목)
"""
import glob, json, sys
import numpy as np
import pandas as pd


def winner(d="work/results"):
    acc = {}
    for f in glob.glob(f"{d}/i1_*_s[01].json"):
        r = json.load(open(f)); a = r["args"]
        if a.get("shuffle"):
            continue
        acc.setdefault(f"{a['res']}:{a['W']}:{a['H']}", []).append(r["val"]["ap"])
    best = max(acc, key=lambda k: np.mean(acc[k]))
    return best


def fixed_thr(res, W, H, top=20):
    from hyfe import bars as B, features as F
    u = pd.read_csv("work/universe.csv"); u = u[(u.exclude.fillna("") == "") & (u.liq_rank <= top)]
    med = [np.median(np.abs(F.label_windows(pd.read_parquet(B.path(res, b)), W, H).fwd)) for b in u.base if __import__("os").path.exists(B.path(res, b))]
    return 3 * float(np.median(med))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fixed":
        print(f"{fixed_thr(sys.argv[2], int(sys.argv[3]), int(sys.argv[4])):.5f}")
    else:
        print(winner())
