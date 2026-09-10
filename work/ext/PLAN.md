# 확장 실험 계획·상태 (9/10 사장 지시: 1~4 동시, 8 은 되면, 5 는 4 가 유의하면)

기준선 = heatf i1 4h·60·84 시드 0~9 앙상블(폴드 5.59·4.95·2.92·2.28, 합산 3.53, 홀드아웃 03~05 2.49). 잔고 시작 $45.46, 예상 지출 ≈ $37.

| # | 실험 | 어디서 | 산출물 | 상태 |
|---|------|--------|--------|------|
| 1a | 행 순열 rp1/2/3(시드 0/1/2) · 봉 순열 cp1/2/3 · 행군 제거 price/vol/rank/feat27/xs10(시드 0·1)+time(시드 0) — 4폴드 | pod | `full_i1_heatf_{rp,cp,drop*}_..._s[0-3]` | 학습 중 |
| 1b | 기울기 saliency + 가림(occlusion) 지도, 최종 모델 sd0~2, 홀드아웃 2026-03~05 창 5,000개 | GB10 | `results/saliency_map.npz`, `results/occlusion_table.csv`, `work/figs/ext_*.png` | `hyfe/saliency.py` |
| 2 | Deflated Sharpe(시도 횟수 N 민감도) | GB10 | `results/dsr_table.csv` | `hyfe/dsr.py` |
| 3 | CNN+GBM z-합(50/50) · 급등급락 확률 가중/게이트 · 1/σ 가중 | GB10 | `results/stack_table.csv` | `work/ext/stack.py`, cohort `--weight` |
| 4 | 지평 H42/126/168(시드 0~2, 4폴드+홀드아웃) · 격자 1h W240 H336 stride4(시드 0·1, 4폴드) | pod | `full_i1_heatf_[sdN_]liq_4h_W60_H{42,126,168}`, `hold_..._H{42,126,168}_s4`, `full_i1_heatf_[sd1_]liq_1h_W240_H336` | 학습 중 |
| 5 | 참여율 슬리피지·용량, 국면별 성과 — **4 가 유의할 때만** | GB10 | 미정 | 대기 |
| 8 | z 회귀 손실(zreg) · 넓은 CNN(i1w), 시드 0·1 4폴드 | pod | `full_i1_heatf_zreg_...`, `full_i1w_heatf_...` | 학습 중 |

## 운영
- 러너 `work/ext_launch.sh`(555 잠금, setsid) → `work/ext_run.log`(생성·준비·배정), 데몬 `work/daemon.log`, 감시 `work/ext_watch.log`(10분마다 잔고·pod 로그, 잔고 < $4 → 전부 종료).
- 끝나면: `pkill -f "ext_launc[h]"`, `python3 -m hyfe.fleet list` 로 pod 0 확인, `work/ext/eval.sh`(코호트·표) → 실험일지 → 논문 v1 갱신(vault).
- 시각·요일 행(6·7)은 학습 경로 버그로 ≈0 이라 `DROP=time` 은 무변화가 예상되는 대조(양성 대조 아님).
