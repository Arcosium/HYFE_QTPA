"""확장 ①b — 최종 heatf 모델이 그림의 어디를 보는가. 기울기 saliency(|∂점수/∂픽셀|, 입력×기울기)와 가림(occlusion: 행군×시간블록을 0 으로 채웠을 때 점수 변화).
홀드아웃 봉(work/bars_hold)에서 2026-03~05 창을 표집(기본 5,000개), 최종 모델 시드 0~2 평균. 학습 없음(추론만, GB10).
usage: HYFE_BARS=work/bars_hold python3 -m hyfe.saliency [--n 5000] [--seeds 0,1,2] → results/saliency_map.npz, results/occlusion_table.csv, work/figs/ext_saliency.png"""
import argparse, json, os
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from hyfe import train_cnn as T, features as F
from hyfe.pilot_gbm import XS_FEATS, add_xs

GROUPS = {"price": range(0, 4), "vol": (4, 5), "time": (6, 7), "rank": range(8, 11), "feat27": range(11, 38), "xs10": range(38, 48)}
ROWNAMES = ["o", "h", "l", "c", "v/창평균", "v/30일", "hour", "dow", "xs_r24", "xs_vol24", "xs_vrel"] + list(F.FEATURES) + XS_FEATS


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=5000); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--start", default="2026-03"); ap.add_argument("--end", default="2026-06")
    ap.add_argument("--fill", default="mean", choices=["mean", "zero"], help="가림 값: mean = 표본 평균 그림의 같은 픽셀(분포 안), zero = 0(피처 행엔 극단값이라 과대평가)")
    a = ap.parse_args(); dev = "cuda"; torch.manual_seed(0)
    bases = [l.strip() for l in open("work/liq12_s4.txt") if l.strip()]
    wins = T.Windows(bases, "4h", 60, 84, 2.0, dev, "2025-09", a.end, "relbin", need_feats=True, stride=1)
    ix = add_xs(wins.idx); ix = ix[(ix.ts >= T.ms(a.start)) & (ix.ts < T.ms(a.end))]
    ix = ix.sample(min(a.n, len(ix)), random_state=0).sort_values("ts"); print("windows", len(ix), "bases", ix.base.nunique(), flush=True)
    models = []
    for sd in a.seeds.split(","):
        meta = json.load(open(f"work/models/final_i1_heatf_4h_W60_H84_sd{sd}.json")); m = T.SmallCNN(in_ch=1).to(dev); m.load_state_dict(torch.load(f"work/models/final_i1_heatf_4h_W60_H84_sd{sd}.pt", map_location=dev)); m.eval(); models.append((m, meta))
    FEATS = models[0][1]["feat_cols"]
    g = torch.tensor(ix.g.to_numpy(), device=dev)
    def feats(meta):
        mu, sd = np.array(meta["feat_mu"]), np.array(meta["feat_sd"]); return torch.tensor(((ix[FEATS].to_numpy() - mu) / sd).astype(np.float32), device=dev)
    def score(m, x, f):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = m(x, f).float()
        return o[:, 1] - o[:, 2]
    bs = 256; sal = np.zeros((T.IMG_H, 60 * T.PX)); ixg = np.zeros_like(sal); s0 = np.zeros(len(ix)); xmean = torch.zeros(T.IMG_H, 60 * T.PX, device=dev)
    for mi, (m, meta) in enumerate(models):                   # 기울기 saliency(모델·창 평균) + 표본 평균 그림(가림 값)
        f = feats(meta)
        for i in range(0, len(ix), bs):
            x = wins.heatf(g[i:i + bs], f[i:i + bs]).detach().requires_grad_(True)
            sc = score(m, x, f[i:i + bs]); sc.sum().backward()
            gr = x.grad[:, 0]; sal += gr.abs().sum(0).cpu().numpy(); ixg += (gr * x[:, 0].detach()).abs().sum(0).cpu().numpy(); s0[i:i + bs] += sc.detach().cpu().numpy()
            if mi == 0:
                xmean += x.detach()[:, 0].sum(0)
    sal /= len(ix) * len(models); ixg /= len(ix) * len(models); s0 /= len(models); xmean /= len(ix)
    fill = xmean if a.fill == "mean" else torch.zeros_like(xmean)
    cell = lambda M: M.reshape(48, 2, 60, 3).mean((1, 3))     # 96×180 픽셀 → 48행×60봉 셀
    os.makedirs("results", exist_ok=True); np.savez_compressed("results/saliency_map.npz", grad=cell(sal), inputgrad=cell(ixg), rows=np.array(ROWNAMES), score0=s0, ts=ix.ts.to_numpy(), base=ix.base.to_numpy().astype(str))
    rows = []                                                  # 가림: 행군×시간블록(6×6) + 행 하나씩(48, 전체 시간)
    masks = [(f"{k}×t{b}", list(v), range(b * 10, b * 10 + 10)) for k, v in GROUPS.items() for b in range(6)] + [(f"{k}", list(v), range(60)) for k, v in GROUPS.items()] + [(f"row{r}:{ROWNAMES[r]}", [r], range(60)) for r in range(48)]
    q = np.quantile(s0, [0.1, 0.9]); top0, bot0 = s0 >= q[1], s0 <= q[0]
    with torch.no_grad():
        for name, rws, cols in masks:
            s1 = np.zeros(len(ix)); M = torch.zeros(T.IMG_H, 60 * T.PX, dtype=torch.bool, device=dev)
            ri = torch.tensor([2 * r + j for r in rws for j in range(2)], device=dev); ci = torch.tensor([3 * c + j for c in cols for j in range(3)], device=dev); M[ri[:, None], ci[None, :]] = True
            for m, meta in models:
                f = feats(meta)
                for i in range(0, len(ix), 512):
                    x = wins.heatf(g[i:i + 512], f[i:i + 512]); x = torch.where(M[None, None], fill[None, None], x)
                    s1[i:i + 512] += score(m, x, f[i:i + 512]).cpu().numpy()
            s1 /= len(models); q1 = np.quantile(s1, [0.1, 0.9])
            rows.append({"mask": name, "rows": len(rws), "bars": len(cols), "spearman": round(float(spearmanr(s0, s1)[0]), 4), "abs_delta_over_sd": round(float(np.abs(s1 - s0).mean() / s0.std()), 4),
                         "top_kept": round(float(((s1 >= q1[1]) & top0).sum() / top0.sum()), 3), "bot_kept": round(float(((s1 <= q1[0]) & bot0).sum() / bot0.sum()), 3)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"results/occlusion_table_{a.fill}.csv", index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 7)); os.makedirs("work/figs", exist_ok=True)
    for k, (t, M) in enumerate((("grad", cell(sal)), ("input×grad", cell(ixg)))):
        im = ax[k].imshow(M, aspect="auto", cmap="magma"); ax[k].set_title(t); ax[k].set_xlabel("bar (0 = oldest, 59 = decision)"); ax[k].set_yticks(range(48)); ax[k].set_yticklabels(ROWNAMES, fontsize=6); fig.colorbar(im, ax=ax[k], shrink=0.6)
    plt.tight_layout(); plt.savefig("work/figs/ext_saliency.png", dpi=150); print("saved")


if __name__ == "__main__":
    main()
