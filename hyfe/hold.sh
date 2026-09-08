#!/bin/bash
# 홀드아웃/롤링 폴드 heatf 소형 CNN 1회 학습 — 시드만 바꿔 반복. usage: SEED=3 bash hyfe/hold.sh
#   기본 = 홀드아웃 s4(검증 2025-12~02, 시험 2026-03~08). 롤링 s5: S=5 VAL=2026-03 TEST=2026-06 TE=2026-09 LIQ=liq12_s5 SEED=n bash hyfe/hold.sh
set -u; cd "$(dirname "$0")/.."; export HYFE_BARS=work/bars_hold
SEED=${SEED:-0}; S=${S:-4}; VAL=${VAL:-2025-12}; TEST=${TEST:-2026-03}; TE=${TE:-2026-09}; LIQ=${LIQ:-liq12_s4}; SUF=$([ "$SEED" != 0 ] && echo _sd$SEED)
echo "=== HOLD i1 heatf 4h_W60_H84 sd$SEED split$S"
python -m hyfe.train_cnn --res 4h --W 60 --H 84 --model i1 --render heatf --label relbin --stride 1 --bases @work/$LIQ.txt --seed $SEED --cap 5000000 --start 2023-01 --val $VAL --test $TEST --test_end $TE --out work/results/hold_i1_heatf${SUF}_4h_W60_H84_s$S.json --save_pred 2>&1 | grep --line-buffered -E "train|^ep|DONE"
echo HOLD_DONE
