# results — 논문 결과의 원자료

`hyfe/cohort.py`(코호트 롱숏: 판단 시각별 상·하위 10%, 14일 보유, 달러중립, 왕복 0.2%, 단순수익)와
`hyfe/perf.py`(연율 Sharpe, Newey-West p, MDD)가 만든 파일을 그대로 두었다. 데이터 창은 2023-01~2026-05 이고
홀드아웃 파일은 연구 데이터 창(2026-05-31)까지만 담는다.

- `seed_table.csv` — heatf 소형 CNN 시드 0~9 하나짜리 결과. 폴드별·합산·홀드아웃 Sharpe/p/MDD (논문 표 4)
- `ensemble10_table.csv` — 시드 10개 z-평균 앙상블의 폴드별·합산·홀드아웃 Sharpe/p/MDD/순자산과 셔플 대조 (논문 표 3)
- `cohort/full_i1_heatf[_sdN]_liq_4h_W60_H84_s{0..3}_cohort_H84.json` — 시드 N, 폴드 s 의 일별 순자산(`daily`)과 요약. s3=2024-06~08, s2=2024-12~2025-02, s1=2025-09~11, s0=2025-12~2026-02
- `cohort/hold_i1_heatf[_sdN]_4h_W60_H84_s4_cohort_H84.json` — 홀드아웃 2026-03~05
- `cohort/ens_heatf10_*` — 시드 10개 앙상블 (`_shuf1`, `_shuf2` 는 점수 셔플 대조), `ens_heatf3_*` — 초기 시드 3개 앙상블
- `gbm/ens_v2_4h_s{0..3}_sim_ls_K100_c0.001*.json` — 대조군 LightGBM v2 같은 폴드의 K=100 롱숏 시뮬(`_shuf` 는 셔플)

재생성: 예측 파일(`*_pred.npz`, 용량 때문에 저장소에 없음)에서 `bash hyfe/eval_seeds.sh`.

## 9/9 추가 (심사 지적 반영)
- `mlp_table.csv` — MLP 대조군(m1 피처 37, m2 heatf 수치 행렬)과 heatf CNN·LightGBM 의 코호트 Sharpe (논문 표 11). 원자료 `cohort/full_m1_*`, `cohort/full_m2_*`, `cohort/ens_m13_*`, `cohort/ens_m23_*`
- `funding_table.csv` — 실측 펀딩비 반영 전후(시드 10 앙상블·LightGBM). 원자료 `cohort/ens_heatf10_*_fund.json`, `gbm/ens_v2_4h_s*_cohort_H84[_fund].json`
- `liquidity_legs.csv` — 롱·숏 다리 종목의 판단 전 30일 일평균 달러 거래대금(백만 달러)
