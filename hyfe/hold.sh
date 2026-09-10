#!/bin/bash
# 홀드아웃/롤링 폴드 heatf 소형 CNN 1회 학습 — 시드만 바꿔 반복. usage: SEED=3 bash hyfe/hold.sh
#   기본 = 홀드아웃 s4(검증 2025-12~02, 시험 2026-03~08). 롤링 s5: S=5 VAL=2026-03 TEST=2026-06 TE=2026-09 LIQ=liq12_s5 SEED=n bash hyfe/hold.sh
#   지평·격자·모델 변형: H=42|126|168, RES=1h W=240 H=336 STRIDE=4, MODEL=i1w, LOSS=zreg (파일명에 그대로 들어간다)
set -u; cd "$(dirname "$0")/.."; export HYFE_BARS=work/bars_hold
SEED=${SEED:-0}; S=${S:-4}; VAL=${VAL:-2025-12}; TEST=${TEST:-2026-03}; TE=${TE:-2026-09}; LIQ=${LIQ:-liq12_s4}
RES=${RES:-4h}; W=${W:-60}; H=${H:-84}; MODEL=${MODEL:-i1}; STRIDE=${STRIDE:-1}; LOSS=${LOSS:-}
SUF=$([ -n "$LOSS" ] && echo _$LOSS)$([ "$SEED" != 0 ] && echo _sd$SEED)
echo "=== HOLD $MODEL heatf$SUF ${RES}_W${W}_H${H} split$S"
python -m hyfe.train_cnn --res $RES --W $W --H $H --model $MODEL --render heatf --label relbin --stride $STRIDE ${LOSS:+--loss $LOSS} --bases @work/$LIQ.txt --seed $SEED --cap 5000000 --start 2023-01 --val $VAL --test $TEST --test_end $TE --out work/results/hold_${MODEL}_heatf${SUF}_${RES}_W${W}_H${H}_s$S.json --save_pred 2>&1 | grep --line-buffered -E "train|^ep|DONE|Error|Traceback"
echo HOLD_DONE
