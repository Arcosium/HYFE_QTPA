"""확장 실험 판정표(①a 배치·④ 지평/격자·⑧ 모델) — pod 결과(work/results)가 모이는 대로 반복 실행. 없는 시드는 건너뛴다.
각 변형: 시드별 4폴드 코호트 Sharpe → 시드 합산 평균, 시드 z-평균 앙상블 합산(기준선은 같은 시드 수로 맞춘 앙상블), 홀드아웃(2026-03~05).
신호 반감기: H84 시드 10 앙상블 예측을 보유 봉수만 바꿔 재판정(학습 없음).
usage: PYTHONPATH=. python3 work/ext/eval.py → results/ext_{layout,horizon,decay,grid1h,model}.csv"""
import glob, json, os, re, subprocess
import numpy as np
import pandas as pd
from hyfe.perf import daily_returns, stats

R = "work/results"; os.makedirs("results", exist_ok=True)


def coh(pred, res, H, bars="work/bars_full", q=0.1):
    out = pred.replace("_pred.npz", f"_cohort_H{H}.json")
    if not os.path.exists(out) and os.path.exists(pred):
        subprocess.run(f"HYFE_BARS={bars} python3 -m hyfe.cohort --pred {pred} --res {res} --H {H} --cost 0.001 --q {q}", shell=True, capture_output=True)
    return out if os.path.exists(out) else None


def zavg(preds, out):
    if not os.path.exists(out) and all(os.path.exists(p) for p in preds):
        subprocess.run(f"python3 -m hyfe.zavg {' '.join(preds)} --out {out}", shell=True, capture_output=True)
    return out if os.path.exists(out) else None


def pooled(files, lag=14, end=None):
    rs = [daily_returns(f)[0] for f in files if f]
    if len(rs) < len(files) or not rs:
        return None
    r = pd.concat(rs); r = r[r.index < end] if end else r
    return stats(r, (1 + r).cumprod(), lag)


def stem(kind, seed, res="4h", W=60, H=84, model="i1", extra=""):
    """fulldir 이름 규칙: full_<model>_heatf[_extra][_sdN]_liq_<res>_W<W>_H<H>"""
    return f"{R}/full_{model}_heatf{extra}{'_sd%d' % seed if seed else ''}_liq_{res}_W{W}_H{H}"


def variant_row(name, stems, res="4h", H=84, bars="work/bars_full", hold_stems=(), tag=""):
    lag = max(1, H * {"4h": 240, "1h": 60, "1d": 1440}[res] // 1440); per = []; ens_files = []
    for s in stems:
        fs = [coh(f"{s}_s{k}_pred.npz", res, H, bars) for k in range(4)]
        st = pooled(fs, lag)
        if st:
            per.append(st["sharpe"]);
    have = [s for s in stems if all(os.path.exists(f"{s}_s{k}_pred.npz") for k in range(4))]
    row = {"variant": name, "n_seeds": len(have), "seed_pooled_mean": round(np.mean(per), 2) if per else None, "seed_pooled_min": round(min(per), 2) if per else None}
    if len(have) >= 2:
        for k in range(4):
            ens_files.append(coh(zavg([f"{s}_s{k}_pred.npz" for s in have], f"{R}/ens_{tag or re.sub(r'[^a-z0-9]', '', name.lower())}{len(have)}_{res}_H{H}_s{k}_pred.npz"), res, H, bars))
        st = pooled(ens_files, lag)
        if st:
            row.update(ens_sharpe=st["sharpe"], ens_p=st["p"], ens_mdd=st["mdd"], ens_folds=" · ".join(str(json.load(open(f))["summary"]["sharpe"]) for f in ens_files))
    if hold_stems:
        hf = [coh(f"{s}_s4_pred.npz", res, H, "work/bars_hold") for s in hold_stems]; hf = [f for f in hf if f]
        if hf:
            hs = [pooled([f], lag, "2026-06-01")["sharpe"] for f in hf]; row["hold_seed_mean"] = round(np.mean(hs), 2); row["hold_n"] = len(hf)
            if len(hf) >= 2:
                he = coh(zavg([f.replace(f"_cohort_H{H}.json", "_pred.npz") for f in hf], f"{R}/ens_{tag or re.sub(r'[^a-z0-9]', '', name.lower())}{len(hf)}_hold_{res}_H{H}_s4_pred.npz"), res, H, "work/bars_hold")
                st = pooled([he], lag, "2026-06-01") if he else None
                if st:
                    row.update(hold_ens_sharpe=st["sharpe"], hold_ens_p=st["p"], hold_ens_mdd=st["mdd"])
    return row


def main():
    base3 = [stem("", s) for s in (0, 1, 2)]; base2 = base3[:2]
    hold3 = [f"{R}/hold_i1_heatf{'_sd%d' % s if s else ''}_4h_W60_H84" for s in (0, 1, 2)]
    # ①a 배치: 행 순열·봉 순열(시드 0~2, 순열 시드 1~3) · 행군 제거(시드 0·1) — 기준선은 같은 시드 수
    lay = [variant_row("기준(시드 0~2)", base3, tag="base"), variant_row("행 순열", [stem("", s, extra=f"_rp{s + 1}") for s in (0, 1, 2)], tag="rp"), variant_row("봉 순열", [stem("", s, extra=f"_cp{s + 1}") for s in (0, 1, 2)], tag="cp"), variant_row("기준(시드 0·1)", base2, tag="base")]
    for g, nm in (("price", "가격 4행 제거"), ("vol", "거래량 2행 제거"), ("rank", "순위 3행 제거"), ("feat27", "피처 27행 제거"), ("xs10", "횡단면 피처 10행 제거"), ("time", "시각·요일 2행 제거(≈0 행, 대조)")):
        lay.append(variant_row(nm, [stem("", s, extra=f"_drop{g}") for s in (0, 1)], tag=f"drop{g}"))
    pd.DataFrame(lay).to_csv("results/ext_layout.csv", index=False); print("① 배치\n", pd.DataFrame(lay).to_string(index=False))
    # ④ 지평(4h): H42·126·168 시드 0~2 + 홀드아웃, 기준 H84 시드 0~2
    hor = [variant_row("H84 (14일) 기준", base3, hold_stems=hold3, tag="base")]
    for H, nm in ((42, "H42 (7일)"), (126, "H126 (21일)"), (168, "H168 (28일)")):
        hor.append(variant_row(nm, [stem("", s, H=H) for s in (0, 1, 2)], H=H, hold_stems=[f"{R}/hold_i1_heatf{'_sd%d' % s if s else ''}_4h_W60_H{H}" for s in (0, 1, 2)], tag=f"h{H}"))
    pd.DataFrame(hor).to_csv("results/ext_horizon.csv", index=False); print("④ 지평\n", pd.DataFrame(hor).to_string(index=False))
    # ④ 반감기: 시드 10 앙상블(H84 학습) 예측을 보유 봉수만 바꿔 재판정
    dec = []
    for k in range(4):   # 비용 0 판정용 사본 이름(코호트 결과 이름에 비용이 없어 충돌 방지)
        if not os.path.exists(f"{R}/ens_heatf10_direct_4h_s{k}_c0_pred.npz"):
            os.symlink(f"ens_heatf10_direct_4h_s{k}_pred.npz", f"{R}/ens_heatf10_direct_4h_s{k}_c0_pred.npz")
    for H in (12, 24, 42, 63, 84, 105, 126, 168, 252):
        st = pooled([coh(f"{R}/ens_heatf10_direct_4h_s{k}_pred.npz", "4h", H) for k in range(4)], max(1, H // 6))
        g0 = [f"{R}/ens_heatf10_direct_4h_s{k}_c0_cohort_H{H}.json" for k in range(4)]
        for k in range(4):
            if not os.path.exists(g0[k]):
                subprocess.run(f"HYFE_BARS=work/bars_full python3 -m hyfe.cohort --pred {R}/ens_heatf10_direct_4h_s{k}_c0_pred.npz --res 4h --H {H} --cost 0", shell=True, capture_output=True)
        sg = pooled(g0, max(1, H // 6))
        if st:
            dec.append({"hold_bars": H, "hold_days": H / 6, **st, "gross_sharpe": sg["sharpe"] if sg else None, "gross_final": sg["final"] if sg else None})
    pd.DataFrame(dec).to_csv("results/ext_decay.csv", index=False); print("④ 반감기(H84 모델, 보유 봉수만 변경)\n", pd.DataFrame(dec).to_string(index=False))
    # ④ 격자 1h(W240·H336, 판단 4봉마다) 시드 0·1 vs 4h 기준 시드 0·1
    g1 = [variant_row("4h·60·84 기준(시드 0·1)", base2, tag="base"), variant_row("1h·240·336 (같은 10일 창·14일 지평)", [stem("", s, res="1h", W=240, H=336) for s in (0, 1)], res="1h", H=336, tag="g1h")]
    pd.DataFrame(g1).to_csv("results/ext_grid1h.csv", index=False); print("④ 격자\n", pd.DataFrame(g1).to_string(index=False))
    # ⑧ 모델: z 회귀 손실 · 넓은 CNN(시드 0·1)
    mo = [variant_row("기준 i1·CE(시드 0·1)", base2, tag="base"), variant_row("i1·z 회귀 손실", [stem("", s, extra="_zreg") for s in (0, 1)], tag="zreg"), variant_row("i1w(채널 1.5배)·CE", [stem("", s, model="i1w") for s in (0, 1)], tag="i1w")]
    pd.DataFrame(mo).to_csv("results/ext_model.csv", index=False); print("⑧ 모델\n", pd.DataFrame(mo).to_string(index=False))


if __name__ == "__main__":
    main()
