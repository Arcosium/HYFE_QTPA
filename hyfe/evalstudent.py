"""학생(증류 CNN) vs 교사 — 같은 ROS 구간·같은 종목·같은 시뮬 규칙으로 spread_z 와 Sharpe·NW p·MDD 를 나란히.
usage: python3 -m hyfe.evalstudent --student work/results/student_i1_heat_gray_v2_4h_W60_H84_sA_pred.npz --teacher_stem work/results/ens_wfv2_4h --tsplits 6,7 --H 84 --only work/liqwf_all.txt
"""
import argparse, json, os, subprocess
import numpy as np
import pandas as pd
from hyfe import metrics as M
from hyfe.perf import daily_returns, stats


def load(p):
    z = np.load(p, allow_pickle=True)
    return pd.DataFrame({"ts": z["ts"], "base": z["base"].astype(str), "s": z["p"][:, 1] - z["p"][:, 2], "fwd": z["fwd"], "sigH": z["sigH"], "y": z["y"]})


def sz(d, s):
    z = d.fwd / np.maximum(d.sigH, 1e-9); q = s.quantile([0.1, 0.9])
    return float(z[s >= q[0.9]].mean() - z[s <= q[0.1]].mean()), float((d.fwd[s >= q[0.9]] > 0).mean()), float((d.fwd[s <= q[0.1]] < 0).mean())


def sim(pred, H, only, extra=""):
    cmd = f"HYFE_BARS=work/bars_full python3 -m hyfe.sim --pred {pred} --res 4h --H {H} --K 100 --side ls --q 0.10 --xs_thr --only {only} {extra}"
    subprocess.run(cmd, shell=True, capture_output=True); return pred.replace("_pred.npz", f"_sim_ls_K100_c0.001_xs{'_shuf' + extra.split()[-1] if extra else ''}.json")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--student", required=True); ap.add_argument("--teacher_stem", required=True); ap.add_argument("--tsplits", default="6,7")
    ap.add_argument("--H", type=int, default=84); ap.add_argument("--only", default="work/liqwf_all.txt"); ap.add_argument("--lag", type=int, default=14)
    a = ap.parse_args()
    st = load(a.student); te = pd.concat([load(f"{a.teacher_stem}_s{s}_pred.npz") for s in a.tsplits.split(",")])
    both = st.merge(te[["ts", "base", "s"]].rename(columns={"s": "t"}), on=["ts", "base"])
    print(f"student rows {len(st)}  teacher rows {len(te)}  공통 {len(both)}  점수 상관 {np.corrcoef(both.s, both.t)[0,1]:.3f}")
    for name, s in (("학생", both.s), ("교사", both.t), ("학생+교사 z평균", (both.s - both.s.mean()) / both.s.std() + (both.t - both.t.mean()) / both.t.std())):
        v = sz(both, s); print(f"  {name:12s} spread_z {v[0]:+.3f} 상위/하위 적중 {v[1]:.3f}/{v[2]:.3f}")
    # 교사 예측을 같은 구간으로 이어 붙인 파일 → 같은 시뮬
    tp = a.student.replace("_pred.npz", "_T_pred.npz"); z = [np.load(f"{a.teacher_stem}_s{s}_pred.npz", allow_pickle=True) for s in a.tsplits.split(",")]
    np.savez_compressed(tp, ts=np.concatenate([x["ts"] for x in z]), base=np.concatenate([x["base"].astype(str) for x in z]), y=np.concatenate([x["y"] for x in z]), p=np.concatenate([x["p"] for x in z]), fwd=np.concatenate([x["fwd"] for x in z]), sigH=np.concatenate([x["sigH"] for x in z]))
    zs = np.load(a.student, allow_pickle=True); sdf = pd.DataFrame({"ts": zs["ts"], "base": zs["base"].astype(str)}); sdf["i"] = np.arange(len(sdf))
    idx = sdf.merge(te[["ts", "base"]], on=["ts", "base"]).i.to_numpy(); sc_ = a.student.replace("_pred.npz", "_C_pred.npz")
    np.savez_compressed(sc_, ts=zs["ts"][idx], base=zs["base"][idx].astype(str), y=zs["y"][idx], p=zs["p"][idx], fwd=zs["fwd"][idx], sigH=zs["sigH"][idx])
    rows = []
    for name, pred in (("학생(전체 유니버스)", a.student), ("학생(교사 행만)", sc_), ("교사", tp)):
        for k, extra in (("실제", ""), ("셔플1", "--shuffle 1"), ("셔플2", "--shuffle 2")):
            out = sim(pred, a.H, a.only, extra)
            if os.path.exists(out):
                r, eq = daily_returns(out); rows.append(dict(who=name, run=k, **stats(r, eq, a.lag)))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
