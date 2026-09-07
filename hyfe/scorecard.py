"""급등·급락 예측 성적표 — 예측 npz(ts, base, y, p) 마다 클래스별 유병률·PR-AUC·lift 와 운용 문턱(상위 1·5·10%)의 정밀도·재현율.
usage: python3 -m hyfe.scorecard work/results/gbm_roll4_liq12full_15m_W120_H10_s*_pred.npz [--csv out.csv]
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def one(path):
    z = np.load(path, allow_pickle=True); y = z["y"]; p = z["p"]
    name = os.path.basename(path).replace("_pred.npz", "")
    rows = []
    for cls, nm in ((1, "급등"), (2, "급락")):
        yy = (y == cls).astype(int); s = p[:, cls]; prev = yy.mean(); ap = average_precision_score(yy, s)
        r = {"run": name, "cls": nm, "n": len(y), "prev%": prev * 100, "PR_AUC": ap, "lift": ap / max(prev, 1e-9)}
        order = np.argsort(-s)
        for q in (0.01, 0.05, 0.10):
            k = max(int(len(y) * q), 1); top = yy[order[:k]]
            r[f"prec@{int(q*100)}%"] = top.mean() * 100; r[f"recall@{int(q*100)}%"] = top.sum() / max(yy.sum(), 1) * 100
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("preds", nargs="+"); ap.add_argument("--csv")
    a = ap.parse_args()
    files = sorted(f for pat in a.preds for f in glob.glob(pat))
    df = pd.DataFrame([r for f in files for r in one(f)])
    pd.set_option("display.width", 250)
    print(df.round(2).to_string(index=False))
    if a.csv:
        df.to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()
