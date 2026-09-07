"""실전용 최종 모델 — 설정(res:W:H)마다 cutoff 이전 전부를 IS 로(마지막 os_months 개월은 조기종료용 OS), 부스터·OS 문턱을 work/models/ 에 저장.
usage: HYFE_BARS=work/bars_hold python3 -m hyfe.train_final --cutoff 2026-09 --configs 15m:120:10,15m:240:60,15m:120:60,1h:60:15
"""
import argparse, json, os
import numpy as np
import pandas as pd
import lightgbm as lgb
from hyfe import bars as B, features as F, metrics as M
from hyfe.pilot_gbm import PARAMS, PRESETS, XS_FEATS, ms, dataset, relativize, add_xs, liq_top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True, help="이 월(포함 안 함) 이전 데이터로 학습"); ap.add_argument("--os_months", type=int, default=3)
    ap.add_argument("--configs", required=True); ap.add_argument("--cap", type=int, default=3_000_000); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="work/models")
    ap.add_argument("--label", default="ksigma"); ap.add_argument("--stride", type=int, default=0); ap.add_argument("--xs", action="store_true")
    ap.add_argument("--preset", default="base", choices=list(PRESETS)); ap.add_argument("--liq_months", type=int, default=0); ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--tag", default="", help="모델 파일 접두(예: v2)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); PARAMS["num_threads"] = a.threads; PARAMS.update(PRESETS[a.preset])
    u = pd.read_csv("work/universe.csv"); u = u[u.exclude.fillna("") == ""]; bases = u.base.tolist()
    if a.liq_months:
        bases = liq_top(bases, a.cutoff, a.liq_months, a.top)
    feats = list(F.FEATURES) + (XS_FEATS if a.xs else [])
    os_start = (pd.Timestamp(a.cutoff) - pd.DateOffset(months=a.os_months)).strftime("%Y-%m")
    for cfg in a.configs.split(","):
        res, W, H = cfg.split(":"); W, H = int(W), int(H)
        stride = a.stride or ({"15m": 4, "1h": 1}.get(res) if H >= 15 else None)      # 방향 모델은 시간 격자 정렬, 발생 모델은 기본 간격
        B.build(bases, "2023-01", a.cutoff, res_list=[res])
        df = dataset(bases, res, W, H, 2.0, "ksigma" if a.label.startswith("rel") else a.label, stride=stride)
        if a.label.startswith("rel"):
            df = relativize(df, a.label)
        if a.xs:
            df = add_xs(df)
        emb = (W + H) * B.RES_MIN[res] * 60_000
        tr = df[df.ts < ms(os_start) - emb]; va = df[(df.ts >= ms(os_start)) & (df.ts < ms(a.cutoff))]
        if len(tr) > a.cap:
            tr = tr.sample(a.cap, random_state=0)
        m = lgb.train({**PARAMS, "seed": 0}, lgb.Dataset(tr[feats], tr.label), 600, valid_sets=[lgb.Dataset(va[feats], va.label)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        pv = m.predict(va[feats]); ev = M.evaluate(va.label.to_numpy(), pv, va.fwd.to_numpy(), va.sigH.to_numpy())
        tag = f"{a.tag + '_' if a.tag else ''}{res}_W{W}_H{H}"; m.save_model(f"{a.out}/{tag}_model.txt")
        sc = M.score(pv)
        meta = {"cfg": cfg, "cutoff": a.cutoff, "os": [os_start, a.cutoff], "n_train": len(tr), "n_os": len(va), "best_iter": m.best_iteration,
                "os_lift": ev["lift"], "os_spread_z": ev.get("spread_z"), "os_hit_top": ev.get("hit_top"), "os_hit_bot": ev.get("hit_bot"),
                "thr_q10": float(np.quantile(sc, 0.10)), "thr_q05": float(np.quantile(sc, 0.05)), "score_mean": float(sc.mean()), "score_std": float(sc.std()),
                "features": feats, "label": a.label, "xs": a.xs, "preset": a.preset, "stride": stride, "bases": sorted(bases)}
        json.dump(meta, open(f"{a.out}/{tag}_meta.json", "w"), indent=1)
        print(f"{tag}: train {len(tr)} OS {len(va)} iter {m.best_iteration} OS lift {ev['lift']:.2f} spread_z {ev.get('spread_z', float('nan')):.3f} thr10 {meta['thr_q10']:+.4f}")


if __name__ == "__main__":
    main()
