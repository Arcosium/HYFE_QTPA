"""논문 docx 용 그림 생성 — 하우스 규칙(ink + 액센트 1 + 회색 2단, hairline 그리드, 소형 활자). 출력 vault/HYFE_QTPA/paper/figs/*.png"""
import os, json, glob, re, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

os.chdir("/home/arcosium/projects/HYFE_QTPA"); sys.path.insert(0, ".")
OUT = "/home/arcosium/vault/HYFE_QTPA/paper/figs"; os.makedirs(OUT, exist_ok=True)
import glob as _g
for p in _g.glob(os.path.expanduser("~/.local/share/fonts/NotoSansKR-*.otf")):
    fm.fontManager.addfont(p)
plt.rcParams.update({"font.family": "Noto Sans KR", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555", "axes.linewidth": 0.6, "xtick.color": "#444", "ytick.color": "#444",
                     "grid.color": "#dddddd", "grid.linewidth": 0.5, "legend.frameon": False, "legend.fontsize": 7.5, "figure.dpi": 200, "savefig.dpi": 200, "axes.unicode_minus": False})
INK, ACC, G1, G2 = "#1c1c1c", "#b3402f", "#7a7a7a", "#c4c4c4"
R = "results/cohort"
from hyfe.perf import daily_returns, stats

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", pad_inches=0.04, facecolor="white"); plt.close(fig); print("saved", name)

# ---------- 그림 1·2: heatf 예시 / 캔들 예시 ----------
def fig_images():
    os.environ["HYFE_BARS"] = "work/bars_hold"
    from hyfe import bars as B, features as F, live_cnn as LC
    from hyfe.pilot_gbm import XS_FEATS
    import torch
    bases = [l.strip() for l in open("work/liq12_s4.txt")]
    end = int(pd.Timestamp("2026-05-31").value // 10**6)
    bb = {}
    for b in bases:
        p = B.path("4h", b)
        if not os.path.exists(p): continue
        d = pd.read_parquet(p); d = d[d.ts <= end]
        if len(d) >= 400: bb[b] = d.tail(400).reset_index(drop=True)
    ctx = LC.bar_context(bb); decision = int(ctx.ts.max()) + 4 * 3_600_000
    m, meta = LC.load_model("work/models/final_i1_heatf_4h_W60_H84_sd0")
    imgs, keys = [], []
    for b, d in bb.items():
        pad = pd.concat([d.iloc[[-1]]] * 84, ignore_index=True); pad["ts"] = d.ts.iloc[-1] + 4 * 3_600_000 * np.arange(1, 85)
        ft = F.make(pd.concat([d, pad], ignore_index=True), 60, 84, stride=1, bars_per_30d=180); ft = ft[ft.ts == d.ts.iloc[-1]]
        if len(ft) == 0: continue
        for c in XS_FEATS: ft[c] = 0.5
        im = LC.render(ctx, b, decision, ft.iloc[-1], meta)
        if im is not None: imgs.append(im); keys.append(b)
    sc = LC.score(m, torch.cat(imgs)); order = np.argsort(sc)
    hi, lo = order[-1], order[0]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6))
    groups = [("가격 O·H·L·C 4행", 0, 8), ("거래량 2행", 8, 12), ("시각·요일 2행", 12, 16), ("횡단면 순위 3행", 16, 22), ("창 끝 피처 37행", 22, 96)]
    for ax, i, t in [(axes[0], hi, f"점수 상위: {keys[hi]} ({sc[hi]:+.2f})"), (axes[1], lo, f"점수 하위: {keys[lo]} ({sc[lo]:+.2f})")]:
        ax.imshow(imgs[i][0, 0].numpy(), cmap="gray", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_title(t, loc="left"); ax.set_xticks([0, 60, 120, 179]); ax.set_xticklabels(["−60봉", "−40", "−20", "판단"]); ax.set_yticks([])
        for name, a, b_ in groups:
            ax.axhline(b_ - 0.5, color=ACC, lw=0.5, alpha=0.8)
    for name, a, b_ in groups:
        axes[0].text(-4, (a + b_) / 2, name, ha="right", va="center", fontsize=6.5, color=INK)
    fig.text(0.5, -0.02, "heatf 96×180 그레이스케일 (봉당 3px, 행당 2px). 2026-05-31 판단 시각, 최종 모델 시드 0 점수 기준", ha="center", fontsize=6.5, color=G1)
    save(fig, "fig1_heatf")
    # 캔들 예시(JKX 형): 같은 두 종목의 마지막 60봉
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.2))
    for ax, i in [(axes[0], hi), (axes[1], lo)]:
        d = bb[keys[i]]; d = d[d.ts < decision].tail(60); o, h, l, c, v = [d[k].to_numpy() for k in "ohlcv"]
        img = np.zeros((96, 180)); lo_, hi_ = l.min(), h.max()
        y = lambda x: int(71 - (x - lo_) / (hi_ - lo_ + 1e-12) * 71)
        for j in range(60):
            x = j * 3; img[y(h[j]):y(l[j]) + 1, x + 1] = 1; img[y(o[j]), x] = 1; img[y(c[j]), x + 2] = 1
            vh = int(v[j] / v.max() * 22); img[95 - vh:96, x + 1] = 1
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="auto", interpolation="nearest"); ax.set_title(f"캔들·거래량 차트: {keys[i]}", loc="left"); ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.5, -0.03, "가설 0 의 입력: 창 안 최고·최저로 정규화한 OHLC 막대(위 3/4)와 거래량(아래 1/4), 96×180 그레이스케일", ha="center", fontsize=6.5, color=G1)
    save(fig, "fig2_candle")

# ---------- 그림 3: 캔들 설정별 spread_z ----------
def fig_candle_settings():
    S = [("1d·20·20 흑백", [0.002, -0.029, -0.139, 0.026]), ("1d·20·20 3채널(+밀도·시각)", [-0.045, -0.104, -0.035, 0.066]), ("1d·20·20 5채널", [0.034, -0.044, -0.048, 0.030]),
         ("4h·120·42 흑백", [0.051, 0.033, -0.023, 0.072]), ("4h·60·42 흑백", [-0.096, -0.072]), ("4h·60·42 3채널", [-0.065]), ("4h·60·42 거래량 제외", [-0.131]),
         ("4h·20·42 흑백", [-0.046, -0.021, -0.008]), ("1h·60·168 흑백", [-0.023, -0.005, -0.025, 0.011]), ("4h·60·42 캔들(폴드별 유동성 유니버스)", [0.05, -0.03, 0.00, 0.08])]
    G = {"1d": [0.117, 0.184, 0.255, 0.143], "4h": [0.20, 0.18], "GBM v2(4h·60·84)": [0.26, 0.35, 0.36, 0.34]}
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    for i, (name, vals) in enumerate(S):
        ax.scatter(vals, [i] * len(vals), s=18, color=INK, zorder=3)
    ax.scatter(G["1d"], [len(S)] * 4, s=22, facecolors="white", edgecolors=ACC, linewidths=1.0, zorder=3)
    ax.scatter(G["GBM v2(4h·60·84)"], [len(S) + 1] * 4, s=22, facecolors="white", edgecolors=ACC, linewidths=1.0, zorder=3)
    ax.set_yticks(range(len(S) + 2)); ax.set_yticklabels([s[0] for s in S] + ["LightGBM 1d·20·20 (대조군)", "LightGBM v2 4h·60·84 (대조군)"])
    ax.axvline(0, color=INK, lw=0.6); ax.axvline(0.2, color=ACC, lw=0.6, ls="--"); ax.text(0.205, -0.9, "사전 등록 기준 0.2", color=ACC, fontsize=6.5, ha="left", va="center")
    ax.set_xlabel("시험 구간 상대방향 점수 차 spread_z (폴드별 점 하나)"); ax.grid(axis="x"); ax.set_xlim(-0.2, 0.42); ax.invert_yaxis()
    save(fig, "fig3_candle_settings")

# ---------- 그림 4: 모멘텀·왜도 진단 ----------
def fig_momentum():
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.3))
    folds = ["폴드 0", "폴드 1", "폴드 2", "폴드 3"]
    a = axes[0]; v = [0.32, -0.44, -0.44, -0.72]; a.bar(folds, v, color=[INK if x > 0 else G1 for x in v], width=0.55); a.axhline(0, color=INK, lw=0.6)
    a.set_title("CNN 점수와 직전 20일 수익률의 스피어만 상관", loc="left"); a.set_ylim(-0.9, 0.5)
    for i, x in enumerate(v): a.text(i, x + (0.03 if x > 0 else -0.08), f"{x:+.2f}", ha="center", fontsize=7)
    b = axes[1]; v2 = [-0.19, 0.09, 0.17, 0.08]; b.bar(folds, v2, color=[INK if x > 0 else G1 for x in v2], width=0.55); b.axhline(0, color=INK, lw=0.6)
    b.set_title("20일 모멘텀 자체의 상대방향 spread_z", loc="left"); b.set_ylim(-0.3, 0.3)
    for i, x in enumerate(v2): b.text(i, x + (0.015 if x > 0 else -0.04), f"{x:+.2f}", ha="center", fontsize=7)
    fig.text(0.5, -0.03, "1d·20·20 흑백 캔들 CNN, 폴드 0~3 시험 구간. 부호가 폴드마다 뒤집힌다", ha="center", fontsize=6.5, color=G1)
    save(fig, "fig4_momentum")

# ---------- 그림 5: 이미지 구성 사다리 ----------
def fig_ladder():
    rows = [("heatf 시드 10 앙상블", 3.53, ACC), ("heatf (시드 0)", 2.92, INK), ("LightGBM v2 (대조군)", 2.18, G1), ("heatx (시드 0)", 0.96, INK), ("heat (시드 0, 42봉)", None, G2), ("캔들 + 거래량 (25개 변형)", None, G2)]
    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    for i, (n, v, c) in enumerate(rows[::-1]):
        if v is None:
            ax.text(0.05, i, "코호트 판정 미도달 (spread_z ≈ 0)" if "캔들" in n else "spread_z 0.03~0.21, 폴드 간 불안정", va="center", fontsize=7, color=G1)
        else:
            ax.barh(i, v, color=c, height=0.55); ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=7.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]]); ax.set_xlim(0, 4.2); ax.set_xlabel("4폴드 합산 Sharpe (코호트 롱숏, 408일)"); ax.grid(axis="x")
    save(fig, "fig5_ladder")

# ---------- 그림 6: 시드 분포 ----------
def fig_seeds():
    df = pd.read_csv("results/seed_table.csv")
    cols = [("2024-06~08_sharpe", "A"), ("2024-12~2025-02_sharpe", "B"), ("2025-09~11_sharpe", "C"), ("2025-12~2026-02_sharpe", "D"), ("pooled_sharpe", "합산"), ("holdout_2026-03~05_sharpe", "홀드아웃")]
    ens = {"A": 5.59, "B": 4.95, "C": 2.92, "D": 2.28, "합산": 3.53, "홀드아웃": 2.49}
    fig, ax = plt.subplots(figsize=(6.3, 2.8)); rng = np.random.default_rng(0)
    for i, (c, lab) in enumerate(cols):
        v = df[c].to_numpy(); ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=INK, alpha=0.75, zorder=3)
        ax.scatter([i], [ens[lab]], s=70, marker="_", color=ACC, linewidths=2, zorder=4)
        ax.text(i + 0.22, ens[lab], f"{ens[lab]:.2f}", color=ACC, fontsize=7, va="center")
    ax.axhline(1.0, color=G1, lw=0.6, ls=":"); ax.axhline(1.5, color=G1, lw=0.6, ls="--"); ax.text(5.45, 1.55, "합산 목표 1.5", fontsize=6.5, color=G1, ha="right"); ax.text(5.45, 1.05, "구간 목표 1", fontsize=6.5, color=G1, ha="right")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{lab}" for _, lab in cols]); ax.set_ylabel("Sharpe (코호트 롱숏)"); ax.grid(axis="y"); ax.set_ylim(-0.5, 7.6)
    ax.scatter([], [], s=14, color=INK, label="시드 하나 (0~9)"); ax.scatter([], [], s=70, marker="_", color=ACC, linewidths=2, label="시드 10개 z-평균 앙상블"); ax.legend(loc="upper right", ncol=2)
    save(fig, "fig6_seeds")

# ---------- 그림 7: 순자산 곡선 ----------
def fig_equity():
    panels = [("3", "A 2024-06~08"), ("2", "B 2024-12~2025-02"), ("1", "C 2025-09~11"), ("0", "D 2025-12~2026-02")]
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.9)); axes = axes.ravel()
    def eq_of(f, cut=None):
        r, eq = daily_returns(f)
        if cut: eq = eq[eq.index <= cut]
        return eq / eq.iloc[0]
    for ax, (s, t) in zip(axes, panels):
        for k in (1, 2):
            e = eq_of(f"{R}/ens_heatf10_direct_4h_s{s}_cohort_H84_shuf{k}.json"); ax.plot(e.index, e.values, color=G2, lw=0.8)
        g = f"results/gbm/ens_v2_4h_s{s}_sim_ls_K100_c0.001.json"
        if os.path.exists(g):
            e = eq_of(g); ax.plot(e.index, e.values, color=G1, lw=0.8, ls="--")
        e = eq_of(f"{R}/ens_heatf10_direct_4h_s{s}_cohort_H84.json"); ax.plot(e.index, e.values, color=ACC, lw=1.2)
        ax.set_title(t, loc="left"); ax.grid(axis="y"); ax.tick_params(axis="x", labelrotation=0, labelsize=6); ax.set_xticks([e.index[0], e.index[-1]]); ax.set_xticklabels([e.index[0].strftime("%y-%m-%d"), e.index[-1].strftime("%y-%m-%d")])
    ax = axes[4]
    for k in (1, 2):
        e = eq_of(f"{R}/ens_heatf10_hold_4h_s4_cohort_H84_shuf{k}.json", "2026-05-31"); ax.plot(e.index, e.values, color=G2, lw=0.8)
    e = eq_of(f"{R}/ens_heatf10_hold_4h_s4_cohort_H84.json", "2026-05-31"); ax.plot(e.index, e.values, color=ACC, lw=1.2)
    ax.set_title("홀드아웃 2026-03~05", loc="left"); ax.grid(axis="y"); ax.tick_params(axis="x", labelsize=6); ax.set_xticks([e.index[0], e.index[-1]]); ax.set_xticklabels([e.index[0].strftime("%y-%m-%d"), e.index[-1].strftime("%y-%m-%d")])
    axes[5].axis("off"); axes[5].plot([], [], color=ACC, lw=1.2, label="heatf CNN 시드 10 앙상블"); axes[5].plot([], [], color=G1, lw=0.8, ls="--", label="LightGBM v2 (K=100 슬롯 시뮬)"); axes[5].plot([], [], color=G2, lw=0.8, label="점수 셔플 ×2"); axes[5].legend(loc="center left")
    fig.text(0.5, -0.01, "순자산(시작 1). 코호트 롱숏: 상·하위 10%, 14일 보유, 왕복 0.2%, 단순수익", ha="center", fontsize=6.5, color=G1); fig.tight_layout()
    save(fig, "fig7_equity")

# ---------- 그림 8: CNN vs GBM ----------
def fig_cnn_vs_gbm():
    labs = ["A", "B", "C", "D", "합산"]; cnn = [5.59, 4.95, 2.92, 2.28, 3.53]; gbm = [4.48, 2.68, 1.54, 1.22, 2.18]; s0 = [3.23, 3.57, 2.69, 2.78, 2.92]
    fig, ax = plt.subplots(figsize=(6.3, 2.5)); x = np.arange(5); w = 0.26
    ax.bar(x - w, gbm, w, color=G1, label="LightGBM v2"); ax.bar(x, s0, w, color=INK, label="heatf CNN 시드 0"); ax.bar(x + w, cnn, w, color=ACC, label="heatf CNN 시드 10 앙상블")
    for xi, v in zip(x + w, cnn): ax.text(xi, v + 0.08, f"{v:.2f}", ha="center", fontsize=6.5, color=ACC)
    for xi, v in zip(x - w, gbm): ax.text(xi, v + 0.08, f"{v:.2f}", ha="center", fontsize=6.5, color=G1)
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel("Sharpe"); ax.grid(axis="y"); ax.legend(loc="upper right", ncol=3); ax.set_ylim(0, 6.6)
    save(fig, "fig8_cnn_vs_gbm")

# ---------- 그림 9: 월별 수익 ----------
def fig_monthly():
    seq = []
    for s, t in [("3", "A"), ("2", "B"), ("1", "C"), ("0", "D")]:
        r, _ = daily_returns(f"{R}/ens_heatf10_direct_4h_s{s}_cohort_H84.json")
        for k, g in r.groupby(r.index.to_period("M")): seq.append((str(k), ((1 + g).prod() - 1) * 100, t))
    r, _ = daily_returns(f"{R}/ens_heatf10_hold_4h_s4_cohort_H84.json"); r = r[r.index <= "2026-05-31"]
    for k, g in r.groupby(r.index.to_period("M")): seq.append((str(k), ((1 + g).prod() - 1) * 100, "홀드아웃"))
    seq = [s for s in seq if True]
    fig, ax = plt.subplots(figsize=(6.3, 2.4)); x = np.arange(len(seq)); v = [s[1] for s in seq]
    ax.bar(x, v, color=[ACC if a >= 0 else G1 for a in v], width=0.6); ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([s[0][2:] for s in seq], rotation=0, fontsize=6.5); ax.set_ylabel("월 수익률 (%)"); ax.grid(axis="y")
    prev = None
    for i, s in enumerate(seq):
        if s[2] != prev: ax.text(i - 0.3, max(v) + 0.9, s[2], fontsize=7, color=INK); prev = s[2]
    ax.set_ylim(min(v) - 1, max(v) + 2.2)
    fig.text(0.5, -0.03, "heatf CNN 시드 10 앙상블의 월별 코호트 롱숏 수익률. 폴드 A~D 와 홀드아웃(각 3개월, 마지막 달은 14일 보유 잔여 포함)", ha="center", fontsize=6.5, color=G1)
    save(fig, "fig9_monthly")

# ---------- 그림 10: 발생 예측 lift ----------
def fig_event():
    labs = ["폴드 0", "폴드 1", "폴드 2", "폴드 3"]; gray = [1.94, 1.90, 1.67, 1.62]; c5 = [1.87, 2.15, 1.74, 1.73]; f1 = [2.22, 2.18, 2.04, 1.83]; gbm = [2.30, 2.46, 2.27, 1.94]
    fig, ax = plt.subplots(figsize=(6.3, 2.4)); x = np.arange(4); w = 0.2
    ax.bar(x - 1.5 * w, gray, w, color=G2, label="흑백 캔들 CNN"); ax.bar(x - 0.5 * w, c5, w, color=INK, label="5채널 캔들 CNN"); ax.bar(x + 0.5 * w, f1, w, color=ACC, label="융합(이미지+피처 27)"); ax.bar(x + 1.5 * w, gbm, w, color=G1, label="LightGBM(피처 27)")
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel("시험 구간 lift (상위 5% 창의 사건율 / 기저)"); ax.set_ylim(1.0, 2.8); ax.grid(axis="y"); ax.legend(loc="upper right", ncol=2)
    fig.text(0.5, -0.03, "급등·급락 발생 예측(15m·120봉·10봉 지평, 740종목 전수, 학습 500만 창). 미학습 종목 값은 표 참조", ha="center", fontsize=6.5, color=G1)
    save(fig, "fig10_event")

# ---------- 그림 11: GBM walk-forward 8폴드 ----------
def fig_wf():
    labs = ["24-03", "24-06", "24-09", "24-12", "25-03", "25-06", "25-09", "25-12"]
    v1 = [-0.98, 3.91, 2.69, 0.86, 2.85, 3.37, -0.43, 2.90]; v2 = [-1.48, 2.98, 4.45, 5.05, 0.15, 5.65, 2.77, -1.60]; mix = [-1.70, 4.58, 4.79, 3.70, 2.37, 6.45, 1.25, 1.30]
    fig, ax = plt.subplots(figsize=(6.3, 2.4)); x = np.arange(8); w = 0.26
    ax.bar(x - w, v1, w, color=G2, label="v1 (7일) 합산 1.58"); ax.bar(x, v2, w, color=G1, label="v2 (14일) 합산 1.92"); ax.bar(x + w, mix, w, color=INK, label="v1+v2 자본 50/50 합산 2.39")
    ax.axhline(0, color=INK, lw=0.6); ax.set_xticks(x); ax.set_xticklabels([f"ROS {l}" for l in labs], fontsize=6.5); ax.set_ylabel("Sharpe"); ax.grid(axis="y"); ax.legend(loc="upper left", ncol=3)
    fig.text(0.5, -0.03, "LightGBM 상대방향 전략의 분기 연속 walk-forward (2024-03~2026-02, 24개월 표본 밖, 폴드별 유동성 상위 200)", ha="center", fontsize=6.5, color=G1)
    save(fig, "fig11_wf")

# ---------- 데이터 요약 ----------
def data_summary():
    c = pd.read_csv("data/coverage.csv"); use = set(l.strip() for l in open("work/all_usable.txt") if l.strip())
    u = c[c.base.isin(use)]
    print("usable", len(u), "rows", int(u.rows.sum()), "delisted", int(u.delisted.sum()), "first", u.first_utc.min(), "src", u.source.value_counts().to_dict())
    print("all", len(c), "rows", int(c.rows.sum()), "delisted", int(c.delisted.sum()), "src", c.source.value_counts().to_dict())

if __name__ == "__main__":
    data_summary()
    for f in [fig_candle_settings, fig_momentum, fig_ladder, fig_seeds, fig_equity, fig_cnn_vs_gbm, fig_monthly, fig_event, fig_wf, fig_images, fig_mlp, fig_decay, fig_saliency, fig_btc, fig_live, fig_pipeline, fig_funding, fig_stack]:
        try: f()
        except Exception as e: print("FAIL", f.__name__, repr(e)[:300])

# ---------- 그림 12: MLP 대조군 ----------
def fig_mlp():
    mt = pd.read_csv("/home/arcosium/vault/HYFE_QTPA/paper/mlp_table.csv"); g = lambda n: mt[mt.model == n].iloc[0]
    series = [("m1 피처 37 MLP", g("m1 시드 3개 앙상블"), G2), ("m2 수치 행렬 MLP", g("m2 시드 3개 앙상블"), G1), ("LightGBM v2", g("LightGBM v2 (같은 피처 37개)"), INK), ("heatf 소형 CNN", g("heatf 소형 CNN 시드 3개 앙상블"), ACC)]
    fig, ax = plt.subplots(figsize=(6.3, 2.5)); x = np.arange(5); w = 0.2
    for i, (lab, r, c) in enumerate(series):
        v = [r[k] for k in "ABCD"] + [r.pooled]; ax.bar(x + (i - 1.5) * w, v, w, color=c, label=lab)
    ax.axhline(0, color=INK, lw=0.6); ax.set_xticks(x); ax.set_xticklabels(["A", "B", "C", "D", "합산"]); ax.set_ylabel("Sharpe"); ax.grid(axis="y"); ax.legend(loc="upper right", ncol=2)
    fig.text(0.5, -0.03, "시드 0·1·2 z-평균 앙상블, 코호트 롱숏(상·하위 10%, 14일, 왕복 0.2%). 같은 폴드·유니버스·라벨", ha="center", fontsize=6.5, color=G1)
    save(fig, "fig12_mlp")

# ---------- 그림 13: 신호 반감기(보유 기간별 Sharpe, 비용 전·후) ----------
def fig_decay():
    d = pd.read_csv("results/ext_decay.csv")
    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    ax.plot(d.hold_days, d.gross_sharpe, marker="o", ms=3.5, color=G1, lw=1.0, label="비용 0")
    ax.plot(d.hold_days, d.sharpe, marker="o", ms=3.5, color=ACC, lw=1.2, label="왕복 0.2% 차감")
    for x, y in zip(d.hold_days, d.sharpe): ax.text(x, y - 0.32, f"{y:.2f}", ha="center", fontsize=6.5, color=ACC)
    ax.axvline(14, color=INK, lw=0.5, ls=":"); ax.text(14.4, 4.3, "학습 지평 14일", fontsize=6.5, color=INK)
    ax.set_xscale("log"); ax.set_xticks([2, 4, 7, 14, 21, 28, 42]); ax.set_xticklabels(["2", "4", "7", "14", "21", "28", "42"]); ax.set_xlabel("보유 기간(일, 로그축)"); ax.set_ylabel("4폴드 합산 Sharpe"); ax.set_ylim(0, 4.7); ax.grid(axis="y"); ax.legend(loc="upper left", ncol=1)
    fig.text(0.5, -0.12, "heatf 시드 10 앙상블(14일 지평 학습)의 점수를 고정하고 코호트 보유 봉수만 바꿔 재판정. 상·하위 10%, 4h 판단.", ha="center", fontsize=6.5, color=G1)
    save(fig, "fig13_decay")

# ---------- 그림 14: 어디를 보는가 — saliency 지도 + 행군 가림 ----------
def fig_saliency():
    z = np.load("results/saliency_map.npz", allow_pickle=True); M = z["grad"]; rows = [str(r) for r in z["rows"]]
    occ = pd.read_csv("results/occlusion_table_mean.csv"); grp = occ[(occ.bars == 60) & (occ["rows"] > 1)].set_index("mask")
    names = {"price": "가격 4행", "vol": "거래량 2행", "time": "시각·요일 2행", "rank": "순위 3행", "feat27": "피처 27행", "xs10": "횡단면 피처 10행"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 3.4), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.55})
    a.imshow(M / M.max(), aspect="auto", cmap="Greys", vmin=0, vmax=1); a.set_xlabel("봉 (0 = 창 시작, 59 = 판단 시각)")
    bounds = [(0, 4, "가격"), (4, 6, "거래량"), (6, 8, "시각·요일"), (8, 11, "순위"), (11, 38, "피처 27"), (38, 48, "횡단면 10")]
    a.set_yticks([(s_ + e - 1) / 2 for s_, e, _ in bounds]); a.set_yticklabels([n for _, _, n in bounds], fontsize=7)
    for s_, e, _ in bounds[1:]: a.hlines(s_ - 0.5, -0.5, 59.5, color=ACC, lw=0.5)
    top = sorted(np.argsort(-M.sum(1))[:6]); ys = []
    for r in top:
        y = r if not ys else max(r, ys[-1] + 2.3); ys.append(y)
        a.annotate(rows[r], xy=(59.5, r), xytext=(62, y), fontsize=5.8, va="center", color=INK, arrowprops=dict(arrowstyle="-", color=G1, lw=0.4))
    a.set_xlim(-0.5, 78); a.set_xticks([0, 10, 20, 30, 40, 50, 59]); a.tick_params(axis="x", labelsize=7)
    order = ["xs10", "feat27", "rank", "price", "vol", "time"]; v = [1 - grp.at[k, "spearman"] for k in order]
    b.barh(range(len(order)), v, color=[ACC if k in ("xs10", "feat27") else INK for k in order], height=0.55)
    for i, x in enumerate(v): b.text(x + 0.01, i, f"{x:.2f}", va="center", fontsize=7)
    b.set_yticks(range(len(order))); b.set_yticklabels([names[k] for k in order], fontsize=7); b.invert_yaxis(); b.set_xlabel("가렸을 때 순위 상관 손실 (1 − ρ)"); b.set_xlim(0, max(v) * 1.3); b.grid(axis="x")
    fig.text(0.5, -0.04, "왼쪽: |∂점수/∂픽셀| 의 평균(진할수록 큼, 최대 = 1). 오른쪽: 행군을 표본 평균 그림으로 가렸을 때 원 점수와의 Spearman 손실. 홀드아웃 2026-03~05 창 5,000개, 최종 모델 시드 0~2.", ha="center", fontsize=6.5, color=G1)
    save(fig, "fig14_saliency")

# ---------- 그림 15: 전략 vs BTC 보유 (수익·MDD) ----------
def _mdd(eq): return float((eq / eq.cummax() - 1).min() * 100)
def fig_btc():
    def strat(files, cut=None):
        r = pd.concat([daily_returns(f)[0] for f in files]); return r[r.index <= cut] if cut else r
    def btc_daily(path, idx):
        b = pd.read_parquet(path, columns=["ts", "c"]); b.index = pd.to_datetime(b.ts, unit="ms"); d = b.c.resample("D").last(); return d.pct_change().reindex(idx).fillna(0.0)
    folds = [("A", "3"), ("B", "2"), ("C", "1"), ("D", "0")]; rows = []
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.7), gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.25})
    rs = strat([f"{R}/ens_heatf10_direct_4h_s{s}_cohort_H84.json" for _, s in folds]); rb = btc_daily("work/bars_full/4h/BTC.parquet", rs.index)
    es, eb = (1 + rs).cumprod(), (1 + rb).cumprod(); x = np.arange(len(rs))
    a.plot(x, es.values, color=ACC, lw=1.2, label=f"heatf 롱숏 ×{es.iloc[-1]:.2f}, MDD {_mdd(es):.1f}%"); a.plot(x, eb.values, color=G1, lw=1.0, label=f"BTC 보유 ×{eb.iloc[-1]:.2f}, MDD {_mdd(eb):.1f}%")
    a.axhline(1, color=INK, lw=0.5); n = 0
    for lab, s in folds:
        r = daily_returns(f"{R}/ens_heatf10_direct_4h_s{s}_cohort_H84.json")[0]; k = len(r); bt = btc_daily("work/bars_full/4h/BTC.parquet", r.index)
        e1, e2 = (1 + r).cumprod(), (1 + bt).cumprod(); rows.append({"period": lab, "days": k, "strat_ret": (e1.iloc[-1] - 1) * 100, "strat_mdd": _mdd(e1), "btc_ret": (e2.iloc[-1] - 1) * 100, "btc_mdd": _mdd(e2)})
        if n: a.axvline(n, color=G2, lw=0.6, ls=":")
        a.text(n + 3, 1.66, f"{lab} {r.index[0].strftime('%y-%m')}~{r.index[-1].strftime('%y-%m')}", fontsize=6, color=G1); n += k
    rows.append({"period": "합산 408일", "days": len(rs), "strat_ret": (es.iloc[-1] - 1) * 100, "strat_mdd": _mdd(es), "btc_ret": (eb.iloc[-1] - 1) * 100, "btc_mdd": _mdd(eb)})
    a.set_xlim(0, len(rs)); a.set_ylim(0.55, 1.75); a.set_ylabel("순자산 (각 구간 시작 = 이어붙임)"); a.set_xlabel("4폴드 시험 구간 일수 (A→D 순, 구간 사이는 불연속)"); a.grid(axis="y"); a.legend(loc="lower left", fontsize=6.5); a.set_title("롤링 4폴드 이어붙임", loc="left")
    rh = strat([f"{R}/ens_heatf10_hold_4h_s4_cohort_H84.json"], "2026-05-31"); bh = btc_daily("work/bars_hold/4h/BTC.parquet", rh.index); eh, ebh = (1 + rh).cumprod(), (1 + bh).cumprod()
    b.plot(eh.index, eh.values, color=ACC, lw=1.2, label=f"롱숏 ×{eh.iloc[-1]:.2f}, MDD {_mdd(eh):.1f}%"); b.plot(ebh.index, ebh.values, color=G1, lw=1.0, label=f"BTC ×{ebh.iloc[-1]:.2f}, MDD {_mdd(ebh):.1f}%"); b.axhline(1, color=INK, lw=0.5)
    b.set_title("홀드아웃 2026-03~05", loc="left"); b.grid(axis="y"); b.set_ylim(0.9, 1.26); b.legend(loc="lower right", fontsize=6.2); b.set_xticks([eh.index[0], eh.index[-1]]); b.set_xticklabels([eh.index[0].strftime("%y-%m-%d"), eh.index[-1].strftime("%y-%m-%d")], fontsize=6.5)
    rows.append({"period": "홀드아웃 2026-03~05", "days": len(rh), "strat_ret": (eh.iloc[-1] - 1) * 100, "strat_mdd": _mdd(eh), "btc_ret": (ebh.iloc[-1] - 1) * 100, "btc_mdd": _mdd(ebh)})
    pd.DataFrame(rows).to_csv("/home/arcosium/vault/HYFE_QTPA/paper/btc_compare.csv", index=False)
    fig.text(0.5, -0.14, "달러중립 코호트 롱숏(시드 10 앙상블, 왕복 0.2%)과 같은 날짜의 BTC 단순 보유. BTC 는 4시간봉 일별 종가.", ha="center", fontsize=6.5, color=G1)
    save(fig, "fig15_btc")

# ---------- 그림 16: 실전 페이퍼 장부 틱 ----------
def fig_live():
    rows = [json.loads(l) for l in open("work/live/dir_cnn_signals.jsonl") if l.strip()]
    t = pd.to_datetime([r["ts"] for r in rows], unit="ms") + pd.Timedelta(hours=9); nl = [sum(o.startswith("l:") for o in r.get("opened", [])) for r in rows]; ns = [sum(o.startswith("s:") for o in r.get("opened", [])) for r in rows]; no = [r.get("n_open", 0) for r in rows]
    fig, ax = plt.subplots(figsize=(6.3, 2.2)); x = np.arange(len(rows)); w = 0.38
    ax.bar(x - w / 2, nl, w, color=ACC, label="신규 롱"); ax.bar(x + w / 2, ns, w, color=G1, label="신규 숏"); ax.set_ylabel("틱당 신규 포지션"); ax.set_ylim(0, max(nl + ns) * 1.6)
    ax2 = ax.twinx(); ax2.plot(x, no, color=INK, lw=1.0, marker="o", ms=2.5, label="보유 포지션 수"); ax2.set_ylabel("보유 포지션 수"); ax2.spines["top"].set_visible(False)
    ax.set_xticks(x[::2]); ax.set_xticklabels([s.strftime("%m-%d %H시") for s in t[::2]], fontsize=6.5, rotation=0); ax.grid(axis="y")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=3)
    fig.text(0.5, -0.06, f"4시간봉 마감 판단(KST 표기). 시드 10 앙상블 점수 상·하위 10%, 14일 보유. 가동 {t[0].strftime('%Y-%m-%d %H시')} ~ {t[-1].strftime('%m-%d %H시')}, 틱 {len(rows)}회.", ha="center", fontsize=6.5, color=G1)
    save(fig, "fig16_live")

# ---------- 그림 17: 재현 파이프라인 ----------
def fig_pipeline():
    from matplotlib.patches import FancyBboxPatch
    steps = [("1분봉 저장소", "CryptoBars 6거래소\n무기한선물, 상장폐지 포함", "hyfe/bars.py"), ("4시간봉·유니버스", "판단 시점 직전 12개월\n거래대금 상위 200, 폴드별", "hyfe/universe.py"), ("라벨·피처", "14일 상대수익 부호\n피처 27 + 횡단면 순위 10", "hyfe/features.py"),
             ("heatf 렌더", "48행 × 60봉 → 96×180\nGPU 즉석 렌더", "train_cnn.py heatf()"), ("소형 CNN 학습", "75.5만 파라미터, 시드 0~9\n폴드 4 + 홀드아웃", "hyfe/train_cnn.py"), ("z-평균 앙상블", "시드별 점수 표준화 후 평균", "hyfe/zavg.py"),
             ("코호트 롱숏 판정", "상·하위 10% · 14일 · 왕복 0.2%\nSharpe · NW p · MDD · 셔플", "cohort.py · perf.py"), ("실전 장부", "4h 마감마다 렌더·추론\nAutoCrypto 화면", "live_cnn.py · live.py")]
    fig, ax = plt.subplots(figsize=(6.4, 3.2)); ax.axis("off"); ax.set_xlim(0, 4); ax.set_ylim(0, 2)
    for i, (t, d, m) in enumerate(steps):
        r, c = divmod(i, 4); x = c + 0.03; y = 1.08 - r * 1.0
        ax.add_patch(FancyBboxPatch((x, y), 0.9, 0.78, boxstyle="round,pad=0.01,rounding_size=0.02", fc="white", ec=ACC if i in (3, 4) else INK, lw=1.0 if i in (3, 4) else 0.7))
        ax.text(x + 0.05, y + 0.66, f"{i + 1}. {t}", fontsize=7.5, fontweight="bold", color=INK, va="center"); ax.text(x + 0.04, y + 0.42, d, fontsize=5.7, color=INK, va="center", linespacing=1.3); ax.text(x + 0.05, y + 0.12, m, fontsize=4.9, color=G1, va="center", family="monospace")
        if c < 3: ax.annotate("", xy=(x + 1.0, y + 0.39), xytext=(x + 0.9, y + 0.39), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7))
    ax.annotate("", xy=(3.5, 1.06), xytext=(3.5, 1.08), arrowprops=dict(arrowstyle="-", color=INK, lw=0.7)); ax.plot([3.5, 3.5, 0.5, 0.5], [1.08, 0.97, 0.97, 0.88], color=INK, lw=0.7); ax.annotate("", xy=(0.5, 0.86), xytext=(0.5, 0.97), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7))
    fig.text(0.5, 0.0, "저장소 results/ 에 시드·앙상블 코호트 json 과 표 CSV, 실험 전 과정은 실험일지.md. 학습은 Runpod A40, 판정·실전은 GB10.", ha="center", fontsize=6.5, color=G1)
    save(fig, "fig17_pipeline")

# ---------- 그림 18: 펀딩비 반영 ----------
def fig_funding():
    ft = pd.read_csv("/home/arcosium/vault/HYFE_QTPA/paper/funding_table.csv"); ft = ft[ft.fold.isin(["A", "B", "C", "D", "합산"])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.5), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.3}); x = np.arange(len(ft)); w = 0.2
    a.bar(x - 1.5 * w, ft.sharpe, w, color=G2, label="CNN 펀딩 전"); a.bar(x - 0.5 * w, ft.sharpe_fund, w, color=ACC, label="CNN 펀딩 후"); a.bar(x + 0.5 * w, ft.gbm_sharpe, w, color=G1, label="LightGBM 전"); a.bar(x + 1.5 * w, ft.gbm_sharpe_fund, w, color=INK, label="LightGBM 후")
    for xi, v in zip(x - 0.5 * w, ft.sharpe_fund): a.text(xi, v + 0.1, f"{v:.2f}", ha="center", fontsize=6, color=ACC)
    a.set_xticks(x); a.set_xticklabels(ft.fold); a.set_ylabel("Sharpe"); a.grid(axis="y"); a.legend(loc="upper right", ncol=2, fontsize=6); a.set_ylim(0, 7.2); a.set_title("코호트 롱숏 Sharpe", loc="left")
    f4 = ft[ft.fold != "합산"]; x4 = np.arange(len(f4)); b.bar(x4 - w, -f4.long_paid_bp_per_hold, 2 * w, color=G1, label="롱 다리 (음수 = 냄)"); b.bar(x4 + w, f4.short_received_bp_per_hold, 2 * w, color=ACC, label="숏 다리 (양수 = 받음)"); b.axhline(0, color=INK, lw=0.6)
    b.set_xticks(x4); b.set_xticklabels(f4.fold); b.set_ylabel("펀딩 수취 (bp / 14일 보유)"); b.grid(axis="y"); b.legend(loc="lower left", fontsize=6); b.set_title("다리별 펀딩", loc="left")
    fig.text(0.5, -0.05, "실측 바이낸스 8시간 정산 펀딩비를 보유 중 누적(롱 −, 숏 +). 자료 없는 종목은 같은 다리 평균 적용. 합산 = 4폴드 408일.", ha="center", fontsize=6.5, color=G1)
    save(fig, "fig18_funding")

# ---------- 그림 19: 가설 3 — 방향은 CNN, 크기는 발생 모델 ----------
def fig_stack():
    st = pd.read_csv("results/stack_table.csv"); pooled = st[st.fold == "pooled"].set_index("variant"); folds = st[st.fold != "pooled"]
    order = [("cnn", "CNN 단독"), ("gbm", "LightGBM 단독"), ("mix", "CNN·LightGBM\nz-합 50/50"), ("gate", "발생 확률\n상위 절반 게이트"), ("wiv", "CNN + 1/σ 비중"), ("wev", "CNN + 발생 확률\n순위 비중"), ("wevi", "CNN + 발생 확률\n× 1/σ 비중")]
    fig, ax = plt.subplots(figsize=(6.3, 2.7)); x = np.arange(len(order))
    for i, (k, lab) in enumerate(order):
        v = pooled.at[k, "sharpe"]; c = ACC if k == "wev" else (G2 if k in ("gbm", "mix") else INK)
        ax.bar(i, v, 0.55, color=c); ax.text(i, v + 0.12, f"{v:.2f}", ha="center", fontsize=7, color=c)
        fv = folds[folds.variant == k].sharpe.to_numpy(); ax.scatter(np.full(len(fv), i) + np.linspace(-0.12, 0.12, len(fv)), fv, s=9, color="white", edgecolors=INK, linewidths=0.5, zorder=3)
    ax.axhline(pooled.at["cnn", "sharpe"], color=G1, lw=0.6, ls=":"); ax.text(1.5, pooled.at["cnn", "sharpe"] + 0.1, "CNN 단독 3.53", fontsize=6.5, color=G1, ha="left")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in order], fontsize=6.5); ax.set_ylabel("4폴드 합산 Sharpe"); ax.set_ylim(0, 7.4); ax.grid(axis="y")
    fig.text(0.5, -0.06, "막대 = 4폴드 합산(408일), 점 = 폴드별 값. 발생 확률 = 15m·120·10 LightGBM 의 급등+급락 확률(판단 직전 75분 안 최신값)의 판단 시각별 순위.", ha="center", fontsize=6.5, color=G1)
    save(fig, "fig19_stack")
