"""점수 혼합 — GBM 점수 z 와 CNN 점수 z 를 OS 에서 고른 가중치로 섞어 ROS 상대수익 spread_z 가 GBM 단독보다 오르는지 짝지어 판정.
CNN 예측은 OS·ROS 를 모두 덮는 stackdir 실행(--val 2025-03 --test 2025-06 --test_end 2026-03)의 _pred.npz.
usage: python3 -m hyfe.blend --gbm work/results/dir_screen_relbin_1d_W20_H20 --cnn work/results/stackdir_i1_1d_W20_H20_pred.npz --splits 0,1
"""
import argparse, json
import numpy as np
import pandas as pd


def load(p, tag):
    z = np.load(p, allow_pickle=True)
    d = pd.DataFrame({"ts": z["ts"], "base": z["base"].astype(str), f"s_{tag}": z["p"][:, 1] - z["p"][:, 2]})
    if tag == "g":
        d["fwd"] = z["fwd"]; d["sigH"] = z["sigH"]
    return d


def zs(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def spread_z(d, s):
    z = d.fwd / np.maximum(d.sigH, 1e-9); q = s.quantile([0.1, 0.9])
    return float(z[s >= q[0.9]].mean() - z[s <= q[0.1]].mean()), float((d.fwd[s >= q[0.9]] > 0).mean()), float((d.fwd[s <= q[0.1]] < 0).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gbm", required=True); ap.add_argument("--cnn", required=True); ap.add_argument("--splits", default="0,1"); ap.add_argument("--n", type=int, default=500); ap.add_argument("--w", type=float, help="고정 가중치(OS 선택 생략, ROS 만 있는 예측에 사용)")
    a = ap.parse_args(); cnn = load(a.cnn, "c"); rng = np.random.default_rng(0)
    for s in map(int, a.splits.split(",")):
        cnn = load(a.cnn.replace("_s0_", f"_s{s}_"), "c") if "_s0_" in a.cnn else cnn
        ros = load(f"{a.gbm}_s{s}_pred.npz", "g").merge(cnn, on=["ts", "base"])
        if a.w is None:
            os_ = load(f"{a.gbm}_s{s}_ospred.npz", "g").merge(cnn, on=["ts", "base"])
            if len(os_) < 500 or len(ros) < 500:
                print(f"s{s} too few rows OS {len(os_)} ROS {len(ros)}"); continue
            ws = np.round(np.arange(0, 1.01, 0.1), 1)
            os_sz = {w: spread_z(os_, (1 - w) * zs(os_.s_g) + w * zs(os_.s_c))[0] for w in ws}
            w = max(os_sz, key=os_sz.get)
        else:
            w = a.w; os_ = ros; os_sz = {0.0: float("nan"), w: float("nan")}
        base = spread_z(ros, zs(ros.s_g)); bl = spread_z(ros, (1 - w) * zs(ros.s_g) + w * zs(ros.s_c)); cn = spread_z(ros, zs(ros.s_c))
        grp = (pd.to_datetime(ros.ts, unit="ms").dt.strftime("%Y-%m") + "_" + ros.base).to_numpy(); _, gi = np.unique(grp, return_inverse=True)
        members = [np.where(gi == i)[0] for i in range(gi.max() + 1)]; diffs = []
        for _ in range(a.n):
            ii = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))]); r = ros.iloc[ii]
            diffs.append(spread_z(r, (1 - w) * zs(r.s_g) + w * zs(r.s_c))[0] - spread_z(r, zs(r.s_g))[0])
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        print(f"s{s} rows OS {len(os_)} ROS {len(ros)} | OS best w={w} (OS spread_z GBM {os_sz[0.0]:.3f} → {os_sz[w]:.3f}) | ROS spread_z GBM {base[0]:.3f} hit {base[1]:.3f}/{base[2]:.3f}  blend {bl[0]:.3f} hit {bl[1]:.3f}/{bl[2]:.3f}  CNN alone {cn[0]:.3f} | Δ {bl[0]-base[0]:+.3f} [{lo:+.3f},{hi:+.3f}]")


if __name__ == "__main__":
    main()
