"""Deflated Sharpe Ratio(Bailey & López de Prado 2014) — 탐색 시도 횟수를 반영한 유의확률.
승자 = 시드 10 앙상블 4폴드 합산 일수익(results/cohort/ens_heatf10_direct_4h_s[0-3]_cohort_H84.json).
시도 분포 = work/results 의 셔플·펀딩·롱온리 아닌 코호트 결과(폴드 단위 Sharpe, 앙상블 제외)의 분산. N 은 민감도표(10 ~ 전체 설정 수).
DSR = Φ[(SR − SR0)·√(T−1) / √(1 − γ3·SR + (γ4−1)/4·SR²)],  SR0 = √V·[(1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(N·e))],  일 단위 SR.
usage: python3 -m hyfe.dsr → results/dsr_table.csv"""
import glob, json, os, re
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
from hyfe.perf import daily_returns

R = "work/results"; G = 0.5772156649


def sr0(v, n):
    return np.sqrt(v) * ((1 - G) * norm.ppf(1 - 1 / n) + G * norm.ppf(1 - 1 / (n * np.e)))


def dsr(r, v, n):
    sr = r.mean() / r.std(); T = len(r); g3 = skew(r); g4 = kurtosis(r, fisher=False)
    return float(norm.cdf((sr - sr0(v, n)) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)))


def main():
    win = pd.concat([daily_returns(f"results/cohort/ens_heatf10_direct_4h_s{s}_cohort_H84.json")[0] for s in range(4)])
    by, fold_sh, fold_days = {}, [], []
    for f in glob.glob(f"{R}/*_cohort_H*.json"):
        b = os.path.basename(f)
        if re.search(r"shuf|fund|_long|^ens_|^krx_|_sd\d+_|gate|_w[a-z]+\.json", b):   # 셔플·펀딩·롱온리·앙상블·KRX(다른 자료)·시드 반복·가중 변형은 시도가 아니다 — 설정당 한 번(시드 0)
            continue
        s = json.load(open(f))["summary"]; fold_sh.append(s["sharpe"]); fold_days.append(s["days"]); by.setdefault(re.sub(r"_s\d_cohort", "_cohort", b), []).append(f)
    sr = win.mean() / win.std(); T = len(win)
    pooled = {k: pd.concat([daily_returns(f)[0] for f in v]) for k, v in by.items() if len(v) >= 3}   # 설정별 폴드 합산(승자와 같은 길이 단위)
    pool_sh = np.array([r.mean() / r.std() * np.sqrt(365) for r in pooled.values()]); v_pool = (pool_sh / np.sqrt(365)).var()
    fold_sh = np.array(fold_sh); v_fold = (fold_sh / np.sqrt(365)).var() * (np.mean(fold_days) / T)   # 폴드(≈90일) 추정오차를 승자 길이 T 로 축소한 대안
    stems = {re.sub(r"_(sd\d+)_", "_", re.sub(r"_s\d\.json$", "", os.path.basename(f))) for f in glob.glob(f"{R}/*.json") if not re.search(r"_cohort|_sim_|_stack|_blend|leaderboard", f)}
    n_img = sum(1 for s in stems if re.match(r"(full_|stack|student|krx_)?(i1|i1f|i1w|i2|i3|j2|f1|m1|m2)_", s))
    print(f"승자: 일 SR {sr:.4f} (연율 {sr*np.sqrt(365):.2f}), T {T}일, 왜도 {skew(win):.2f}, 첨도 {kurtosis(win, fisher=False):.2f}")
    print(f"시도 분포(합산): 설정 {len(pool_sh)}개, 연율 Sharpe 평균 {pool_sh.mean():.2f} 표준편차 {pool_sh.std():.2f} | 폴드 단위 {len(fold_sh)}개 표준편차 {fold_sh.std():.2f} → T 축소 {np.sqrt(v_fold)*np.sqrt(365):.2f}; 설정 수 전체 {len(stems)}, 이미지 모델 {n_img}")
    rows = []
    for name, n in [("사전등록 시드(참고)", 10), ("캔들 22설정", 22), ("코호트 판정 설정", len(pool_sh)), ("이미지 모델 설정", n_img), ("100", 100), ("전체 설정", len(stems)), ("1000", 1000)]:
        rows.append({"N": n, "기준": name, "SR0_annual": round(sr0(v_pool, n) * np.sqrt(365), 2), "DSR": round(dsr(win, v_pool, n), 4), "SR0_fold": round(sr0(v_fold, n) * np.sqrt(365), 2), "DSR_fold": round(dsr(win, v_fold, n), 4), "PSR": round(dsr(win, 0.0, 2), 4)})   # PSR = 시도 보정 없음(SR0 = 0)
    hold = daily_returns("results/cohort/ens_heatf10_hold_4h_s4_cohort_H84.json")[0]; hold = hold[hold.index < "2026-06-01"]   # 사전등록 홀드아웃(2026-03~05): 사용 2회(GBM·CNN)
    for n in (2, 22):
        rows.append({"N": n, "기준": f"홀드아웃 03~05 (N={n})", "SR0_annual": round(sr0(v_pool, n) * np.sqrt(365), 2), "DSR": round(dsr(hold, v_pool, n), 4), "SR0_fold": round(sr0(v_fold, n) * np.sqrt(365), 2), "DSR_fold": round(dsr(hold, v_fold, n), 4), "PSR": round(dsr(hold, 0.0, 2), 4)})
    df = pd.DataFrame(rows); print(f"홀드아웃: 연율 {hold.mean()/hold.std()*np.sqrt(365):.2f}, T {len(hold)}일")
    print(df.to_string(index=False)); os.makedirs("results", exist_ok=True); df.to_csv("results/dsr_table.csv", index=False)
    json.dump({"sr_daily": float(sr), "sharpe_annual": float(sr * np.sqrt(365)), "T": int(T), "skew": float(skew(win)), "kurt": float(kurtosis(win, fisher=False)),
               "trials_pooled": len(pool_sh), "pooled_sharpe_mean": float(pool_sh.mean()), "pooled_sharpe_sd": float(pool_sh.std()), "v_pool_daily": float(v_pool),
               "trials_fold": len(fold_sh), "fold_sharpe_sd": float(fold_sh.std()), "v_fold_daily_scaled": float(v_fold), "n_stems": len(stems), "n_img": n_img, "pooled_configs": sorted(pooled)},
              open("results/dsr_inputs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
