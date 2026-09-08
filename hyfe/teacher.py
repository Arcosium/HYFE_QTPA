"""교사 점수 파일 — walk-forward 앙상블의 표본 밖 점수(폴드별 _pred.npz)를 이어 붙여 (ts, base, score) npz 로.
usage: python3 -m hyfe.teacher --stem work/results/ens_wfv2_4h --splits 0-7 --out work/results/teacher_v2.npz"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stem", required=True); ap.add_argument("--splits", default="0-7"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); lo, hi = map(int, a.splits.split("-")); ts, base, sc = [], [], []
    for s in range(lo, hi + 1):
        z = np.load(f"{a.stem}_s{s}_pred.npz", allow_pickle=True); ts.append(z["ts"]); base.append(z["base"].astype(str)); sc.append(z["p"][:, 1] - z["p"][:, 2])
    ts, base, sc = np.concatenate(ts), np.concatenate(base), np.concatenate(sc).astype(np.float32)
    np.savez_compressed(a.out, ts=ts, base=base, score=sc); print(a.out, len(ts), "rows", "score std %.3f" % sc.std())


if __name__ == "__main__":
    main()
