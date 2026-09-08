#!/bin/bash
# 홀드아웃(2026-03~08, 첫 사용 구간) heatf 소형 CNN 1회 학습 — 시드만 바꿔 반복. usage: SEED=3 bash hyfe/hold.sh
set -u; cd "$(dirname "$0")/.."; export HYFE_BARS=work/bars_hold
SEED=${SEED:-0}; SUF=$([ "$SEED" != 0 ] && echo _sd$SEED)
echo "=== HOLD i1 heatf 4h_W60_H84 sd$SEED split4"
python -m hyfe.train_cnn --res 4h --W 60 --H 84 --model i1 --render heatf --label relbin --stride 1 --bases @work/liq12_s4.txt --seed $SEED --cap 5000000 --start 2023-01 --val 2025-12 --test 2026-03 --test_end 2026-09 --out work/results/hold_i1_heatf${SUF}_4h_W60_H84_s4.json --save_pred 2>&1 | grep --line-buffered -E "train|^ep|DONE"
echo HOLD_DONE
