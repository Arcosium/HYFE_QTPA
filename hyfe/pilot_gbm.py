"""S0 GBM 지도: (해상도 R × 창 W × 지평 H) 마다 LightGBM 을 학습해 PR-AUC 를 잰다.
분할은 시간(학습 < 검증 < 시험, 엠바고 W+H 봉) + 종목(holdout 20% 는 학습에서 빼고 unseen 으로 따로 평가).
usage: python3 -m hyfe.pilot_gbm --top 60 --out work/pilot_gbm.csv [--configs 1m:240:60,...]
"""
import argparse, itertools, os, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from hyfe import bars as B, features as F, metrics as M

RES = ["1m", "5m", "15m", "1h", "4h", "1d"]
WS = [60, 120, 240]
HDIV = [12, 4, 2]
# 해상도별 학습 시작(길수록 표본 적은 저해상도에 유리). 검증·시험 창은 공통.
START = {"1m": "2025-01", "5m": "2025-01", "15m": "2024-01", "1h": "2024-01", "4h": "2023-01", "1d": "2023-01"}
SPLITS = [  # (OS 시작, ROS 시작, ROS 끝) 월 단위 — 롤링 폴드. IS 는 OS 시작 전(엠바고 W+H 봉), --train_months 로 고정 길이
    ("2025-09", "2025-12", "2026-03"),
    ("2025-06", "2025-09", "2025-12"),
    ("2024-09", "2024-12", "2025-03"),
    ("2024-03", "2024-06", "2024-09"),
    ("2025-12", "2026-03", "2026-09"),   # 4 = 최종 홀드아웃(HYFE_BARS=work/bars_hold 로 한 번만). IS ~2025-11, OS 2025-12~26-02, ROS 2026-03~08
]
PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8)
PRESETS = {"base": {}, "small": dict(num_leaves=15, learning_rate=0.03, min_data_in_leaf=500, feature_fraction=0.6),
           "big": dict(num_leaves=127, learning_rate=0.05, min_data_in_leaf=100), "reg": dict(num_leaves=31, learning_rate=0.02, feature_fraction=0.5, lambda_l2=10.0)}
XS_COLS = ["ret_W", "ret_q", "vol_W", "vol_ratio", "rng_mean", "qv_rel30", "amihud", "v_last_rel", "rsi", "z_W"]   # 횡단면 순위(같은 판단 시각 유니버스 안 백분위)
XS_FEATS = [f"xs_{c}" for c in XS_COLS]


def relativize(df, label, k=2.0, fixed=0.02):
    """횡단면 상대 수익 라벨: 같은 판단 시각 유니버스 평균을 뺀다(시장 베타 제거). 20종목 미만 시각은 버림."""
    cnt = df.groupby("ts").fwd.transform("size"); df = df[cnt >= 20].copy()
    df["fwd_abs"] = df.fwd; df["fwd"] = df.fwd - df.groupby("ts").fwd.transform("mean")
    df["label"] = F.label_of(df.fwd.to_numpy(), df.sigH.to_numpy(), k, "binary" if label == "relbin" else "ksigma", fixed)
    return df


def add_xs(df):
    """상대 라벨엔 상대 피처: 판단 시각별 백분위 순위(실전에서도 같은 시각 유니버스 전체로 계산)."""
    g = df.groupby("ts")
    for c in XS_COLS:
        df[f"xs_{c}"] = g[c].rank(pct=True)
    return df


def ms(month):
    return int(pd.Timestamp(month).value // 1_000_000)


MAJORS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "TON"]


def universe_bases(u, g, delisted="in"):
    """S4 종목군. 상폐 제외는 bybit_trades(체결 집계 상폐분)와 2026-07 이전 종료 종목을 뺀다."""
    if delisted == "out":
        u = u[(u.source != "bybit_trades") & (u.last_utc.fillna("2026-08") >= "2026-07")]
    if g == "G1":
        return [b for b in MAJORS if b in set(u.base)]
    if g in ("G2", "G3"):
        return u[u.liq_rank <= (50 if g == "G2" else 200)].base.tolist()
    if g == "G4":
        return u.base.tolist()
    if g == "G5":  # 1h 수익률 표준편차로 3분위, 분위마다 유동성 상위 50
        vp = "work/vol.csv"
        if not os.path.exists(vp):
            rows = []
            for b in u.base:
                p = B.path("1h", b)
                if os.path.exists(p):
                    c = pd.read_parquet(p).c.to_numpy(); rows.append((b, float(np.std(np.diff(np.log(np.maximum(c, 1e-12)))))))
            pd.DataFrame(rows, columns=["base", "vol_1h"]).to_csv(vp, index=False)
        v = pd.read_csv(vp).merge(u[["base", "liq_rank"]], on="base")
        v["tier"] = pd.qcut(v.vol_1h, 3, labels=False)
        return v.sort_values("liq_rank").groupby("tier").head(50).base.tolist()
    raise ValueError(g)


def dataset(bases, res, W, H, k, label="ksigma", fixed=0.02, per_base=0, seed=0, stride=None):
    """per_base>0 이면 종목당 창을 그 수로 무작위 축소(1m 처럼 창이 수십만인 해상도의 메모리 상한)."""
    out = []
    for b in bases:
        p = B.path(res, b)
        if not os.path.exists(p):
            continue
        f = F.make(pd.read_parquet(p), W, H, k, stride=stride, bars_per_30d=max(W, 30 * 1440 // B.RES_MIN[res]), mode=label, fixed=fixed, align=bool(stride))
        if f is not None and len(f):
            if per_base and len(f) > per_base:
                f = f.sample(per_base, random_state=seed).sort_values("ts")
            f["base"] = b; out.append(f)
    return pd.concat(out, ignore_index=True) if out else None


def liq_top(bases, val_start, months=12, n=200):
    """OS 시작 전 months 개월의 달러 거래대금(1h 캐시 qv)으로 상위 n — 종목 선정의 미래 정보 차단."""
    t1 = ms(val_start); t0 = t1 - months * 30 * 86_400_000
    tot = {}
    for b in bases:
        p = B.path("1h", b)
        if os.path.exists(p):
            d = pd.read_parquet(p, columns=["ts", "qv"]); m = (d.ts >= t0) & (d.ts < t1)
            if m.sum() >= 24 * 60:
                tot[b] = float(d.qv[m].sum())
    return set(sorted(tot, key=tot.get, reverse=True)[:n])


def run_one(df, res, W, H, holdout, val_start, test_start, test_end, cap=300_000, seed=0, save_pred=None, train_months=0, allowed=None):
    bar_ms = B.RES_MIN[res] * 60_000
    emb = (W + H) * bar_ms
    if allowed is not None:
        df = df[df.base.isin(allowed)]
    seen = ~df.base.isin(holdout)
    is_from = ms(val_start) - train_months * 30 * 86_400_000 if train_months else 0
    tr = df[seen & (df.ts < ms(val_start) - emb) & (df.ts >= is_from)]
    va = df[seen & (df.ts >= ms(val_start)) & (df.ts < ms(test_start) - emb)]
    te = df[(df.ts >= ms(test_start)) & (df.ts < ms(test_end))]
    if len(tr) > cap:
        tr = tr.sample(cap, random_state=seed)
    if len(tr) < 5000 or len(va) < 500 or len(te) < 500:
        return None
    dtr = lgb.Dataset(tr[F.FEATURES], tr.label); dva = lgb.Dataset(va[F.FEATURES], va.label)
    m = lgb.train({**PARAMS, "seed": seed}, dtr, 600, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    r = {"n_train": len(tr), "n_val": len(va), "n_test": len(te), "best_iter": m.best_iteration}
    ev = lambda d: M.evaluate(d.label.to_numpy(), m.predict(d[F.FEATURES]), d.fwd.to_numpy(), d.sigH.to_numpy())
    r.update({f"val_{k}": v for k, v in ev(va).items()})
    un = te.base.isin(holdout)
    r.update({f"test_{k}": v for k, v in ev(te[~un]).items()})
    if un.sum() > 500:
        r.update({f"unseen_{k}": v for k, v in ev(te[un]).items()})
    imp = pd.Series(m.feature_importance("gain"), index=F.FEATURES).sort_values(ascending=False)
    r["top_feats"] = ",".join(imp.index[:5])
    if save_pred:
        np.savez_compressed(save_pred, ts=te.ts.to_numpy(), base=te.base.to_numpy().astype(str), y=te.label.to_numpy(),
                            p=m.predict(te[F.FEATURES]), fwd=te.fwd.to_numpy(), sigH=te.sigH.to_numpy())
        np.savez_compressed(save_pred.replace("_pred.npz", "_ospred.npz"), ts=va.ts.to_numpy(), base=va.base.to_numpy().astype(str), y=va.label.to_numpy(),
                            p=m.predict(va[F.FEATURES]), fwd=va.fwd.to_numpy(), sigH=va.sigH.to_numpy())   # OS 예측: 문턱 결정용(ROS 미사용)
        m.save_model(save_pred.replace("_pred.npz", "_model.txt"))   # 실전 엔진(live) 용 부스터
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=60); ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--out", default="work/pilot_gbm.csv"); ap.add_argument("--configs")
    ap.add_argument("--start", help="모든 해상도의 학습 시작월을 이 값으로(CNN 과 같은 조건으로 기준선 낼 때)")
    ap.add_argument("--cap", type=int, default=300_000); ap.add_argument("--splits", default="0,1")
    ap.add_argument("--label", default="ksigma", choices=["ksigma", "fixed", "binary", "rel", "relbin"]); ap.add_argument("--fixed", type=float, default=0.02)
    ap.add_argument("--save_pred", action="store_true", help="시험 집합 예측을 <out>_<cfg>_s<i>_pred.npz 로 저장")
    ap.add_argument("--per_base", type=int, default=0); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--universe", default="", help="S4 종목군: G1 메이저10 · G2 유동성50 · G3 200 · G4 전체 · G5 변동성층화150. 평가는 항상 G4 홀드아웃 전체")
    ap.add_argument("--delisted", default="in", choices=["in", "out"])
    ap.add_argument("--train_months", type=int, default=0, help="IS 고정 길이(개월). 0 이면 확장형")
    ap.add_argument("--feats", default="all", choices=["all", "window"], help="window: 창 밖 정보(qv_rel30·hour·dow) 제외 — CNN 과 정보량을 맞춘 기준선")
    ap.add_argument("--xs", action="store_true"); ap.add_argument("--preset", default="base", choices=list(PRESETS)); ap.add_argument("--stride", type=int, default=0, help="창 시작 간격(봉). 0 이면 H/2. 앙상블 격자 맞출 때 지정")
    ap.add_argument("--macro", action="store_true", help="work/macro_<res>.parquet 의 BTC·시장폭·펀딩 피처를 더한다")
    ap.add_argument("--liq_months", type=int, default=0, help=">0 이면 종목군을 폴드마다 OS 시작 전 N 개월 거래대금 상위 --top 으로(미래 정보 없음). 피처는 전체 종목으로 만든다")
    a = ap.parse_args(); PARAMS.update(PRESETS[a.preset])
    if a.feats == "window":
        for c in ("qv_rel30", "hour", "dow"):
            F.FEATURES.remove(c)
    if a.start:
        for r in START: START[r] = a.start
    PARAMS["num_threads"] = a.threads
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    use_splits = [int(s) for s in a.splits.split(",")]
    u = pd.read_csv("work/universe.csv"); u = u[u.exclude.fillna("") == ""]
    if a.liq_months:
        holdout = set(u[u.holdout].base); bases = u.base.tolist()      # 전체로 피처를 만들고 폴드마다 allowed 로 자른다
    elif a.universe:
        holdout = set(u[u.holdout].base)                      # 공통 미학습 집합(전체 기준) — 종목군과 무관하게 같은 잣대
        bases = sorted(set(universe_bases(u, a.universe, a.delisted)) | holdout)
    else:
        u = u[u.liq_rank <= a.top]; bases = u.base.tolist(); holdout = set(u[u.holdout].base)
    print(len(bases), "bases,", len(holdout), "holdout", a.universe, a.delisted)
    cfgs = [tuple(c.split(":")) for c in a.configs.split(",")] if a.configs else [(r, str(w), str(w // d)) for r in RES for w in WS for d in HDIV]
    done = set()
    if os.path.exists(a.out):
        d = pd.read_csv(a.out); done = set(zip(d.res, d.W, d.H, d.split))
    for res, W, H in cfgs:
        W, H = int(W), int(H)
        if all((res, W, H, i) in done for i in use_splits):
            continue
        B.build(bases, START[res], "2026-02", res_list=[res])
        t0 = time.time(); df = dataset(bases, res, W, H, a.k, "ksigma" if a.label.startswith("rel") else a.label, a.fixed, a.per_base, stride=a.stride or None)
        if df is None:
            print("skip", res, W, H); continue
        if a.macro:
            from hyfe.macro import MACRO
            mt = pd.read_parquet(f"work/macro_{res}.parquet"); df = df.merge(mt, on="ts", how="left").dropna(subset=MACRO)
            for c in MACRO:
                if c not in F.FEATURES: F.FEATURES.append(c)
        if a.label.startswith("rel"):   # 횡단면 상대 수익: 같은 판단 시각 유니버스 평균을 뺀다 → 시장 베타 제거
            df = relativize(df, a.label, a.k, a.fixed)
        if a.xs:
            df = add_xs(df)
            for c in XS_FEATS:
                if c not in F.FEATURES: F.FEATURES.append(c)
        for i, (vs, ts_, te) in enumerate(SPLITS):
            if i not in use_splits or (res, W, H, i) in done:
                continue
            pred = a.out.replace(".csv", f"_{res}_W{W}_H{H}_s{i}_pred.npz") if a.save_pred else None
            allowed = liq_top(bases, vs, a.liq_months, a.top) if a.liq_months else None
            r = run_one(df, res, W, H, holdout, vs, ts_, te, cap=a.cap, save_pred=pred, train_months=a.train_months, allowed=allowed)
            if r is None:
                print("insufficient", res, W, H, i); continue
            row = {"res": res, "W": W, "H": H, "split": i, "k": a.k, "label": a.label, "universe": a.universe or f"top{a.top}", "delisted": a.delisted, "feats": a.feats + ("+macro" if a.macro else ""),
                   "train_months": a.train_months,
                   "n_rows": len(df), **r, "sec": round(time.time() - t0)}
            pd.DataFrame([row]).to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
            print(f"{res:>3} W={W:<3} H={H:<3} s{i} val_ap={r['val_ap']:.4f} lift={r['val_lift']:.2f} test_lift={r['test_lift']:.2f} "
                  f"unseen_lift={r.get('unseen_lift', float('nan')):.2f} n={r['n_train']} {round(time.time()-t0)}s")


if __name__ == "__main__":
    main()
