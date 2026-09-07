"""스태킹 — 이미지의 순기여 판정. CNN 의 표본 밖 점수(p_up, p_dn, 차)를 GBM 피처에 더해, 같은 IS·OS·ROS 의 GBM 과 짝지어 비교한다.
CNN 은 --val 2025-03 --test 2025-06 --test_end 2026-03 으로 학습해 2025-06 이후 점수가 전부 표본 밖.
usage: HYFE_BARS=work/bars_full python3 -m hyfe.stack --res 15m --W 120 --H 10 --pred work/results/stack_i1_15m_W120_H10_pred.npz
IS 2025-06~08(엠바고) / OS 2025-09~11 / ROS 2025-12~2026-02, 홀드아웃 종목은 unseen.
"""
import argparse, json, os
import numpy as np
import pandas as pd
import lightgbm as lgb
from hyfe import bars as B, features as F, metrics as M
from hyfe.pilot_gbm import PARAMS, ms, dataset


def fit_eval(tr, va, te, cols, un, seed=0):
    m = lgb.train({**PARAMS, "seed": seed}, lgb.Dataset(tr[cols], tr.label), 600, valid_sets=[lgb.Dataset(va[cols], va.label)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    p = m.predict(te[cols])
    ev = lambda d, pp: M.evaluate(d.label.to_numpy(), pp, d.fwd.to_numpy(), d.sigH.to_numpy())
    imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
    return {"OS": ev(va, m.predict(va[cols])), "ROS": ev(te[~un], p[~un]), "ROS_unseen": ev(te[un], p[un]), "top": imp.index[:6].tolist()}, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True); ap.add_argument("--W", type=int, required=True); ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--k", type=float, default=2.0); ap.add_argument("--pred", required=True); ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--is_start", default="2025-06"); ap.add_argument("--val", default="2025-09"); ap.add_argument("--test", default="2025-12"); ap.add_argument("--test_end", default="2026-03")
    ap.add_argument("--out"); ap.add_argument("--label", default="ksigma"); ap.add_argument("--stride", type=int, default=0)
    a = ap.parse_args()
    u = pd.read_csv("work/universe.csv"); u = u[(u.exclude.fillna("") == "") & (u.liq_rank <= a.top)]
    holdout = set(u[u.holdout].base)
    rel = a.label.startswith("rel")
    df = dataset(u.base.tolist(), a.res, a.W, a.H, a.k, "ksigma" if rel else a.label, stride=a.stride or None)
    if rel:   # pilot_gbm 과 같은 횡단면 상대 라벨
        cnt = df.groupby("ts").fwd.transform("size"); df = df[cnt >= 20].copy()
        df["fwd"] = df.fwd - df.groupby("ts").fwd.transform("mean")
        df["label"] = F.label_of(df.fwd.to_numpy(), df.sigH.to_numpy(), a.k, "binary" if a.label == "relbin" else "ksigma", 0.02)
    df = df[df.ts >= ms(a.is_start)]
    z = np.load(a.pred, allow_pickle=True)
    cnn = pd.DataFrame({"ts": z["ts"], "base": z["base"].astype(str), "cnn_up": z["p"][:, 1], "cnn_dn": z["p"][:, 2]})
    cnn["cnn_dir"] = cnn.cnn_up - cnn.cnn_dn
    df = df.merge(cnn, on=["base", "ts"], how="inner")
    emb = (a.W + a.H) * B.RES_MIN[a.res] * 60_000
    seen = ~df.base.isin(holdout)
    tr = df[seen & (df.ts < ms(a.val) - emb)]
    va = df[seen & (df.ts >= ms(a.val)) & (df.ts < ms(a.test) - emb)]
    te = df[(df.ts >= ms(a.test)) & (df.ts < ms(a.test_end))]
    un = te.base.isin(holdout).to_numpy()
    print(f"rows {len(df)} IS {len(tr)} OS {len(va)} ROS {len(te)} (unseen {un.sum()})")
    base_cols = list(F.FEATURES); stack_cols = base_cols + ["cnn_up", "cnn_dn", "cnn_dir"]
    rA, pA = fit_eval(tr, va, te, base_cols, un)
    rB, pB = fit_eval(tr, va, te, stack_cols, un)
    pc = np.c_[1 - va.cnn_up - va.cnn_dn, va.cnn_up, va.cnn_dn]      # CNN 단독(같은 OS 에서) 참고값
    rC = {"OS": M.evaluate(va.label.to_numpy(), pc, va.fwd.to_numpy(), va.sigH.to_numpy())}
    month = (pd.to_datetime(te.ts, unit="ms").dt.strftime("%Y-%m") + "_" + te.base.astype(str)).to_numpy()
    boot = M.paired_bootstrap(te.label.to_numpy(), pB, pA, month, n=500)
    boot_un = M.paired_bootstrap(te.label.to_numpy()[un], pB[un], pA[un], month[un], n=500) if un.sum() > 1000 else None
    res = {"args": vars(a), "gbm": rA, "gbm+cnn": rB, "cnn_alone_OS": rC["OS"], "paired_ROS": boot, "paired_ROS_unseen": boot_un}
    for k in ("gbm", "gbm+cnn"):
        r = res[k]; print(f"{k:8s} OS {r['OS']['lift']:.3f}  ROS {r['ROS']['lift']:.3f}  unseen {r['ROS_unseen']['lift']:.3f}  top {r['top'][:4]}")
    print(f"ΔAP(gbm+cnn − gbm) ROS {boot['diff']:+.4f} [{boot['lo']:+.4f},{boot['hi']:+.4f}] P(Δ≤0)={boot['p_le0']:.3f}")
    if boot_un:
        print(f"ΔAP unseen {boot_un['diff']:+.4f} [{boot_un['lo']:+.4f},{boot_un['hi']:+.4f}] P(Δ≤0)={boot_un['p_le0']:.3f}")
    out = a.out or a.pred.replace("_pred.npz", "_stack.json")
    json.dump(res, open(out, "w"), indent=1, default=float); print("wrote", out)


if __name__ == "__main__":
    main()
