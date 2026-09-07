"""창 표본화 + kσ 라벨 + 표/밀도 피처 (J1 GBM 입력, F1 융합 공용).
창 = 봉 [i-W, i), 라벨 = c[i-1] → c[i-1+H] 로그수익률을 창 안 σ(1봉)·√H 로 스케일.
label: 0 평상 · 1 급등 · 2 급락. 모든 피처는 창 안 과거값만 쓴다.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

EPS = 1e-12


def _rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.clip(d, 0, None)).rolling(n).mean().to_numpy()
    dn = pd.Series(np.clip(-d, 0, None)).rolling(n).mean().to_numpy()
    return 100 - 100 / (1 + up / (dn + EPS))


def label_of(fwd, sigH, k=2.0, mode="ksigma", fixed=0.02):
    """0 평상 · 1 급등 · 2 급락. ksigma: ±k·σ_H · fixed: ±fixed 로그수익 · binary: 부호(JKX식, 평상 없음)"""
    if mode == "binary":
        return np.where(fwd > 0, 1, 2)
    th = k * sigH if mode == "ksigma" else np.full_like(fwd, fixed)
    return np.where(fwd > th, 1, np.where(fwd < -th, 2, 0))


def _idx(ts, n, W, H, stride, align):
    """창 끝 인덱스 격자. align 이면 판단 시각(봉 마감 = ts[e]+bar) 이 stride×bar 의 배수인 e 부터 시작 — 해상도가 다른 모델의 앙상블 정렬용."""
    idx = np.arange(W, n - H + 1, stride)
    if align and len(ts) > 1:
        bar = int(ts[1] - ts[0]); grid = stride * bar
        for e in range(W - 1, n - H):
            if (int(ts[e]) + bar) % grid == 0:
                idx = np.arange(e + 1, n - H + 1, stride); break
    return idx


def label_windows(bars, W, H, k=2.0, stride=None, mode="ksigma", fixed=0.02, align=False):
    """이미지 학습용: 피처 없이 (창 끝 인덱스 e, label, fwd, sigH) 만. 무거래 봉 50% 초과 창은 뺀다."""
    n = len(bars)
    stride = stride or max(1, H // 2)
    if n < W + H + 2:
        return None
    c = bars.c.to_numpy(float); v = bars.v.to_numpy(float)
    logc = np.log(np.maximum(c, EPS)); r1 = np.diff(logc, prepend=logc[0])
    sig1 = pd.Series(r1).rolling(W).std().to_numpy()
    idx = _idx(bars.ts.to_numpy(), n, W, H, stride, align); e = idx - 1
    fwd = logc[e + H] - logc[e]; sigH = sig1[e] * np.sqrt(H)
    label = label_of(fwd, sigH, k, mode, fixed)
    zero = sliding_window_view(v == 0, W)[idx - W].mean(1)
    ok = (zero <= 0.5) & (sigH > 0) & np.isfinite(fwd)
    return pd.DataFrame({"e": e[ok], "ts": bars.ts.to_numpy()[e[ok]], "label": label[ok], "fwd": fwd[ok], "sigH": sigH[ok]})


def make(bars, W, H, k=2.0, stride=None, bars_per_30d=None, mode="ksigma", fixed=0.02, align=False):
    """bars: DataFrame(ts,o,h,l,c,v,qv) 균일 격자. → DataFrame(ts, label, fwd, sigH, 피처…)"""
    n = len(bars)
    stride = stride or max(1, H // 2)
    if n < W + H + 2:
        return None
    ts = bars.ts.to_numpy(); c = bars.c.to_numpy(float); h = bars.h.to_numpy(float); l = bars.l.to_numpy(float)
    v = bars.v.to_numpy(float); qv = bars.qv.to_numpy(float)
    logc = np.log(np.maximum(c, EPS)); r1 = np.diff(logc, prepend=logc[0])
    S = lambda x: pd.Series(x)
    q = max(W // 4, 2)
    sig1 = S(r1).rolling(W).std().to_numpy()
    sigq = S(r1).rolling(q).std().to_numpy()
    rng = (h - l) / np.maximum(c, EPS)
    rng_mean = S(rng).rolling(W).mean().to_numpy()
    cmin = S(c).rolling(W).min().to_numpy(); cmax = S(c).rolling(W).max().to_numpy()
    cmean = S(c).rolling(W).mean().to_numpy(); cstd = S(c).rolling(W).std().to_numpy()
    maq = S(c).rolling(q).mean().to_numpy()
    vmean = S(v).rolling(W).mean().to_numpy()
    vsum_q = S(v).rolling(q).sum().to_numpy(); vsum_W = S(v).rolling(W).sum().to_numpy()
    qvW = S(qv).rolling(W).sum().to_numpy()
    L = bars_per_30d or W * 8
    qv30 = S(qvW).rolling(L, min_periods=W).mean().to_numpy()
    amihud = S(np.abs(r1) / (qv + 1.0)).rolling(W).mean().to_numpy()
    rsi = _rsi(c, min(14, W))

    idx = _idx(ts, n, W, H, stride, align)   # 창 끝(exclusive) 인덱스
    e = idx - 1                               # 창 마지막 봉
    fwd = logc[e + H] - logc[e]
    sigH = sig1[e] * np.sqrt(H)
    label = label_of(fwd, sigH, k, mode, fixed)

    # 창 행렬 피처 (n_pos × W)
    vm = sliding_window_view(v, W)[idx - W]
    rm = sliding_window_view(r1, W)[idx - W]
    vtot = vm.sum(1) + EPS
    top5 = -np.partition(-vm, 4, axis=1)[:, :5].sum(1) / vtot if W > 5 else np.ones(len(idx))
    hhi = ((vm / vtot[:, None]) ** 2).sum(1)
    vmed = np.median(vm, 1)
    n_spike = (vm > 3 * vmed[:, None]).sum(1)
    zero_ratio = (vm == 0).mean(1)
    upv = (vm * (rm > 0)).sum(1); dnv = (vm * (rm < 0)).sum(1)
    n_big = (np.abs(rm) > 2 * sig1[e][:, None]).sum(1)

    dt = pd.to_datetime(ts[e], unit="ms")
    f = pd.DataFrame({
        "ts": ts[e], "label": label, "fwd": fwd, "sigH": sigH,
        "ret_1": r1[e], "ret_5": logc[e] - logc[np.maximum(e - 5, 0)], "ret_q": logc[e] - logc[e - q], "ret_W": logc[e] - logc[idx - W],
        "vol_W": sig1[e], "vol_q": sigq[e], "vol_ratio": sigq[e] / (sig1[e] + EPS),
        "rng_mean": rng_mean[e], "rng_last_rel": rng[e] / (rng_mean[e] + EPS),
        "pos_W": (c[e] - cmin[e]) / (cmax[e] - cmin[e] + EPS), "z_W": (c[e] - cmean[e]) / (cstd[e] + EPS),
        "c_maq": c[e] / (maq[e] + EPS) - 1, "c_maW": c[e] / (cmean[e] + EPS) - 1, "maq_maW": maq[e] / (cmean[e] + EPS) - 1,
        "rsi": rsi[e],
        "v_last_rel": v[e] / (vmean[e] + EPS), "v_recent_share": vsum_q[e] / (vsum_W[e] + EPS),
        "qv_rel30": qvW[e] / (qv30[e] + EPS), "amihud": amihud[e],
        "top5_share": top5, "hhi": hhi, "n_spike": n_spike, "zero_ratio": zero_ratio,
        "updown_vol": upv / (dnv + EPS), "n_big": n_big,
        "hour": dt.hour.to_numpy(), "dow": dt.dayofweek.to_numpy(),
    })
    f = f[(f.zero_ratio <= 0.5) & (f.sigH > 0)].replace([np.inf, -np.inf], np.nan).dropna()
    return f


FEATURES = ["ret_1", "ret_5", "ret_q", "ret_W", "vol_W", "vol_q", "vol_ratio", "rng_mean", "rng_last_rel", "pos_W", "z_W",
            "c_maq", "c_maW", "maq_maW", "rsi", "v_last_rel", "v_recent_share", "qv_rel30", "amihud", "top5_share", "hhi",
            "n_spike", "zero_ratio", "updown_vol", "n_big", "hour", "dow"]
