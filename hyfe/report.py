"""결과 모아 리더보드 + CNN vs GBM 짝지은 블록 부트스트랩(같은 시험 집합, 종목×월 블록).
usage: python3 -m hyfe.report [work/results]
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
from hyfe import metrics as M


def leaderboard(d):
    rows = []
    for f in sorted(glob.glob(f"{d}/*.json")):
        r = json.load(open(f)); a = r["args"]
        tag = f"{a['res']}_W{a['W']}_H{a['H']}"
        row = {"run": os.path.basename(f)[:-5], "cfg": tag, "model": a["model"], "label": a.get("label", "ksigma"), "vol": not a.get("no_volume"),
               "render": a.get("render", "bar"), "shuffle": a.get("shuffle", False), "val": a["val"], "n_train": r["n_train"], "epochs": len(r["hist"]), "sec": r["sec"]}
        for part, nm in (("val", "OS"), ("test", "ROS"), ("unseen", "ROS_unseen")):   # IS=학습, OS=검증(선택), ROS=시험(판정)
            e = r.get(part) or {}
            row[f"{nm}_ap"] = e.get("ap"); row[f"{nm}_lift"] = e.get("lift"); row[f"{nm}_spread_z"] = e.get("spread_z")
        rows.append(row)
    for f in sorted(glob.glob(f"{d}/gbm_*.csv")):
        g = pd.read_csv(f)
        for _, x in g.iterrows():
            stem = os.path.basename(f)[:-4]          # 같은 설정의 GBM 이 여러 csv(지도·창안·라벨 변형)에 있으므로 파일명으로 구분
            kk = "" if float(x.get("k", 2)) == 2 else f"_k{x.k}"
            rows.append({"run": f"{stem}{kk}_{x.res}_W{x.W}_H{x.H}_s{x.split}", "cfg": f"{x.res}_W{x.W}_H{x.H}", "model": "gbm", "label": x.get("label", "ksigma"),
                         "vol": True, "render": "-", "shuffle": False, "val": ["2025-09", "2025-06"][int(x.split)], "n_train": x.n_train, "epochs": x.best_iter, "sec": x.sec,
                         "OS_ap": x.val_ap, "OS_lift": x.val_lift, "OS_spread_z": x.get("val_spread_z"), "ROS_ap": x.test_ap, "ROS_lift": x.test_lift,
                         "ROS_spread_z": x.get("test_spread_z"), "ROS_unseen_ap": x.get("unseen_ap"), "ROS_unseen_lift": x.get("unseen_lift"), "ROS_unseen_spread_z": x.get("unseen_spread_z")})
    return pd.DataFrame(rows)


def paired(d, cnn_pred, gbm_pred):
    a, b = np.load(cnn_pred, allow_pickle=True), np.load(gbm_pred, allow_pickle=True)
    ka = pd.DataFrame({"ts": a["ts"], "base": a["base"]}).reset_index(); kb = pd.DataFrame({"ts": b["ts"], "base": b["base"]}).reset_index()
    j = ka.merge(kb, on=["ts", "base"], suffixes=("_a", "_b"))
    if len(j) < 1000:
        return None
    ia, ib = j.index_a.to_numpy(), j.index_b.to_numpy()
    y = a["y"][ia]; assert (y == b["y"][ib]).all()
    month = (pd.to_datetime(j.ts, unit="ms").dt.strftime("%Y-%m") + "_" + j.base.astype(str)).to_numpy()
    return {"n": len(j), **M.paired_bootstrap(y, a["p"][ia], b["p"][ib], month, n=500)}


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "work/results"
    lb = leaderboard(d)
    cols = ["run", "model", "label", "vol", "shuffle", "n_train", "OS_lift", "ROS_lift", "ROS_unseen_lift", "ROS_spread_z", "ROS_unseen_spread_z"]
    print(lb.sort_values(["cfg", "val"]).to_string(index=False, columns=[c for c in cols if c in lb.columns], float_format=lambda v: f"{v:.3f}"))
    lb.to_csv(f"{d}/leaderboard.csv", index=False)
    # GBM 예측을 (res,W,H,k,label,split) 로 색인 — 같은 라벨의 GBM 만 짝지음. 후보가 여럿이면 S1 기준선 > 지도 > 창안 순(파일명 정렬)
    gpred = {}
    for f in sorted(glob.glob(f"{d}/gbm_*.csv")):
        g = pd.read_csv(f); stem = os.path.basename(f)[:-4]
        if "window" in stem or "feats" in g.columns and (g.feats == "window").any():
            continue
        for _, x in g.iterrows():
            key = (x.res, int(x.W), int(x.H), float(x.get("k", 2)), x.get("label", "ksigma"), int(x.split))
            p = f"{d}/{stem}_{x.res}_W{x.W}_H{x.H}_s{x.split}_pred.npz"
            if os.path.exists(p):
                gpred.setdefault(key, p)
    for f in sorted(glob.glob(f"{d}/*_s[01].json")):
        r = json.load(open(f)); a = r["args"]
        if a.get("shuffle"):
            continue
        s = int(f[-6]); key = (a["res"], a["W"], a["H"], float(a.get("k", 2)), a.get("label", "ksigma"), s)
        cp = f[:-5] + "_pred.npz"
        if key in gpred and os.path.exists(cp):
            pr = paired(d, cp, gpred[key])
            if pr:
                print(f"{os.path.basename(f)[:-5]} vs {os.path.basename(gpred[key])[:-9]}: ΔAP={pr['diff']:+.4f} 95%CI[{pr['lo']:+.4f},{pr['hi']:+.4f}] P(Δ≤0)={pr['p_le0']:.3f} n={pr['n']}")


if __name__ == "__main__":
    main()
