"""실전용 heatf 이미지 렌더 + 소형 CNN 추론(CPU). train_cnn.Windows.heatf 와 같은 그림을 라이브 봉에서 만든다.
행 구성(48행 → 96px): [o,h,l,c 로그비(±10%→0~1)] + [v/창평균 0~3] + [30일 대비 거래량 ±3σ] + [시각] + [요일]
                    + [봉별 횡단면 순위 3: 24h 수익·24h 변동·30일 대비 거래량] + [창 끝 피처 37(표준화→시그모이드)]
"""
import json
import numpy as np
import pandas as pd
import torch
from hyfe import features as F
from hyfe.train_cnn import SmallCNN, IMG_H, PX
from hyfe.pilot_gbm import XS_FEATS

W = 60


def load_model(stem):
    """stem = work/models/final_i1_heatf_4h_W60_H84_sd0 (json + pt). json 의 feat_cols/feat_mu/feat_sd 로 표준화."""
    meta = json.load(open(stem + ".json")); m = SmallCNN(in_ch=1); m.load_state_dict(torch.load(stem + ".pt", map_location="cpu")); m.eval()
    return m, meta


def bar_context(bars_by_base, bpd=6):
    """종목별 4h 봉(ts,o,h,l,c,v) → 봉별 열: vrel30(log1p v/30일평균), hour, dow, r24, vol24 + 봉별 횡단면 순위 3열(같은 ts 유니버스 안 백분위)."""
    rows = []
    for b, d in bars_by_base.items():
        d = d.sort_values("ts").reset_index(drop=True)
        v30 = d.v.rolling(30 * bpd, min_periods=bpd).mean().bfill()
        lc = np.log(d.c.clip(lower=1e-12)); r1 = lc.diff().fillna(0.0)
        dt = pd.to_datetime(d.ts, unit="ms")
        rows.append(pd.DataFrame({"base": b, "ts": d.ts, "o": d.o, "h": d.h, "l": d.l, "c": d.c, "v": d.v,
                                  "vrel": np.log1p(d.v / np.maximum(v30, 1e-9)), "hour": dt.dt.hour / 23.0, "dow": dt.dt.dayofweek / 6.0,
                                  "r24": lc - lc.shift(bpd).fillna(lc.iloc[0]), "vol24": r1.rolling(bpd, min_periods=2).std().bfill()}))
    a = pd.concat(rows, ignore_index=True); g = a.groupby("ts")
    a["xs_r24"] = g.r24.rank(pct=True); a["xs_vol24"] = g.vol24.rank(pct=True); a["xs_vrel"] = g.vrel.rank(pct=True)
    return a


def render(ctx, base, decision_ts, feat_row, meta):
    """한 종목의 heatf 이미지 (1,1,96,180). ctx = bar_context 결과, decision_ts = 판단 시각(마지막 마감 봉의 ts + 4h)."""
    d = ctx[(ctx.base == base) & (ctx.ts < decision_ts)].tail(W)
    if len(d) < W:
        return None
    last = float(d.c.iloc[-1])
    px = np.log(d[["o", "h", "l", "c"]].to_numpy() / max(last, 1e-12)) * 100.0            # (W,4)
    px = np.clip(px / 10.0, -1, 1) * 0.5 + 0.5
    v = d.v.to_numpy(); v = np.clip(v / max(v.mean(), 1e-9) / 3.0, 0, 1)
    vrel = np.clip(d.vrel.to_numpy(), -3, 3) / 3.0 * 0.5 + 0.5
    rows = np.c_[px, v, vrel, d.hour.to_numpy() / 23.0, d.dow.to_numpy() / 6.0, d.xs_r24.to_numpy(), d.xs_vol24.to_numpy(), d.xs_vrel.to_numpy()].T   # (11,W) — 학습 경로(sequence)가 hour/23·dow/6 를 한 번 더 나누므로 그대로 맞춘다
    f = (feat_row[meta["feat_cols"]].to_numpy(dtype=float) - np.array(meta["feat_mu"])) / np.array(meta["feat_sd"])
    fr = 1.0 / (1.0 + np.exp(-f / 2.0))                                                       # (37,)
    img = np.vstack([rows, np.repeat(fr[:, None], W, 1)])                                     # (48,W)
    img = np.repeat(np.repeat(img, IMG_H // img.shape[0], 0), PX, 1)                          # (96,180)
    pad = IMG_H - img.shape[0]
    if pad > 0:
        img = np.vstack([img, np.zeros((pad, img.shape[1]))])
    return torch.tensor(img, dtype=torch.float32)[None, None]


@torch.no_grad()
def score(model, imgs):
    """(B,1,96,180) → 방향 점수 p_up − p_dn"""
    p = torch.softmax(model(imgs), 1).numpy(); return p[:, 1] - p[:, 2]


def demo():
    """자체점검: 로컬 4h 캐시 30종목으로 그림 크기·NaN·점수 범위 확인(모델 통계는 임시)."""
    import os
    from hyfe import bars as B
    os.environ.setdefault("HYFE_BARS", "work/bars_full")
    bases = [l.strip() for l in open("work/liq12_s0.txt")][:30]
    bb = {b: pd.read_parquet(B.path("4h", b)).tail(400) for b in bases}
    ctx = bar_context(bb); decision = int(ctx.ts.max()) + 4 * 3_600_000
    b = bases[0]; d = bb[b]; bar_ms = 4 * 3_600_000
    pad = pd.concat([d.iloc[[-1]]] * 84, ignore_index=True); pad["ts"] = d.ts.iloc[-1] + bar_ms * np.arange(1, 85)
    ft = F.make(pd.concat([d, pad], ignore_index=True), W, 84, stride=1, bars_per_30d=180); ft = ft[ft.ts == d.ts.iloc[-1]]
    for c in XS_FEATS: ft[c] = 0.5
    cols = list(F.FEATURES) + XS_FEATS; meta = {"feat_cols": cols, "feat_mu": [0.0] * len(cols), "feat_sd": [1.0] * len(cols)}
    img = render(ctx, b, decision, ft.iloc[-1], meta); assert img is not None and img.shape == (1, 1, 96, 180) and not torch.isnan(img).any()
    m = SmallCNN(in_ch=1); m.load_state_dict(torch.load("work/models/full_i1_heatf_liq_4h_W60_H84_s0.pt", map_location="cpu")); m.eval()
    s = score(m, img); assert np.isfinite(s).all(); print("demo ok", img.shape, float(img.min()), float(img.max()), "score", s)


if __name__ == "__main__":
    demo()
