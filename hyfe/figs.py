"""보고서 그림 — 폴드별 순자산 곡선(실제 신호 vs 점수 셔플 무작위), 작은 배수 4장, 같은 세로축.
usage: python3 -m hyfe.figs equity --stem work/results/ens_aligned --out work/figs/equity_ens_aligned.png
팔레트: dataviz 참조 인스턴스(실제 = 파랑 #2a78d6, 무작위 = 회색 #898781, 잉크 #0b0b0b, 괘선 #e1e0d9). 축 하나, 얇은 선, 직접 라벨.
"""
import argparse, json, os
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

BLUE, GRAY, INK, MUTED, GRID, SURFACE = "#2a78d6", "#898781", "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
FOLD_LABEL = {3: "폴드 3 · ROS 2024-06~08", 2: "폴드 2 · ROS 2024-12~25-02", 1: "폴드 1 · ROS 2025-09~11", 0: "폴드 0 · ROS 2025-12~26-02"}


def _font():
    c = [f for f in fm.findSystemFonts() if "NotoSansKR" in f or "NotoSansCJK" in f]
    return fm.FontProperties(fname=c[0]) if c else None


def equity(stem, out, K=20, cost=0.001, side="short", title=None):
    fp = _font()
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0), dpi=160, sharey=True)
    for ax, s in zip(axes, [3, 2, 1, 0]):
        real = json.load(open(f"{stem}_s{s}_sim_{side}_K{K}_c{cost}.json"))["daily"]
        rnd_p = f"{stem}_s{s}_sim_{side}_K{K}_c{cost}_shuf1.json"
        rnd = json.load(open(rnd_p))["daily"] if os.path.exists(rnd_p) else None
        r = pd.Series(real); r.index = pd.to_datetime(r.index)
        ax.plot(r.index, r.values, color=BLUE, lw=1.6)
        ax.text(r.index[-1], r.values[-1], f" ×{r.values[-1]:.2f}", color=INK, fontsize=8, va="center", fontproperties=fp)
        if rnd:
            q = pd.Series(rnd); q.index = pd.to_datetime(q.index)
            ax.plot(q.index, q.values, color=GRAY, lw=1.2)
            ax.text(q.index[-1], q.values[-1], f" ×{q.values[-1]:.2f}", color=MUTED, fontsize=8, va="center", fontproperties=fp)
        ax.axhline(1.0, color=GRID, lw=0.8, zorder=0)
        ax.set_title(FOLD_LABEL[s], loc="left", fontsize=8.5, color=INK, fontproperties=fp, pad=4)
        ax.set_facecolor(SURFACE); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color("#c3c2b7"); ax.spines[sp].set_linewidth(0.6)
        ax.tick_params(colors=MUTED, labelsize=7, length=2)
        ticks = [r.index[0], r.index[len(r) // 2], r.index[-1]]
        ax.set_xticks(ticks); ax.set_xticklabels([d.strftime("%m/%d") for d in ticks], fontproperties=fp)
        ax.margins(x=0.02)
    axes[0].set_ylabel("순자산 (시작 1.0)", fontsize=8, color=MUTED, fontproperties=fp)
    fig.legend(handles=[plt.Line2D([], [], color=BLUE, lw=1.6), plt.Line2D([], [], color=GRAY, lw=1.2)], labels=[title or "실제 신호(앙상블 하위 10% 공매도)", "점수 셔플 무작위 대조"],
               loc="lower center", ncol=2, frameon=False, fontsize=8, prop=fp, bbox_to_anchor=(0.5, -0.06))
    fig.patch.set_facecolor("white"); plt.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True); fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("kind", choices=["equity"]); ap.add_argument("--stem", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=20); ap.add_argument("--side", default="short"); ap.add_argument("--title")
    a = ap.parse_args(); equity(a.stem, a.out, a.K, side=a.side, title=a.title)
