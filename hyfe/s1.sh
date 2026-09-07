#!/bin/bash
# S1 이미지 핵심 검정 — 설정(res:W:H) 마다 GBM 기준선(같은 데이터·분할) + I1 CNN 2분할 + 플라시보 1런.
# usage: [LR=1e-4] [GBM=0] [PLACEBO=0] bash hyfe/s1.sh 15m:120:10 15m:60:5     (pod 의 /workspace/hyfe 에서)
set -u
cd "$(dirname "$0")/.."
TOP=${TOP:-200}; START=${START:-2023-01}; LR=${LR:-}; MODEL=${MODEL:-i1}; CAP=${CAP:-2000000}; GBM=${GBM:-1}; PLACEBO=${PLACEBO:-1}
LRARG=${LR:+--lr $LR}; SUF=${LR:+_lr$LR}
for cfg in "$@"; do
  IFS=: read res W H <<< "$cfg"; tag="${res}_W${W}_H${H}"
  if [ "$GBM" = 1 ]; then echo "=== $tag GBM baseline"; python -m hyfe.pilot_gbm --top $TOP --start $START --configs $cfg --save_pred --out work/results/gbm_${tag}.csv 2>&1 | grep --line-buffered -v Warning; fi
  for s in 0 1; do
    if [ $s = 0 ]; then V=2025-09; T=2025-12; TE=2026-03; else V=2025-06; T=2025-09; TE=2025-12; fi
    echo "=== $tag $MODEL$SUF split$s"; python -m hyfe.train_cnn --res $res --W $W --H $H --model $MODEL --cap $CAP --top $TOP --start $START --val $V --test $T --test_end $TE $LRARG --out work/results/${MODEL}_${tag}${SUF}_s${s}.json --save_pred 2>&1 | grep --line-buffered -E "train|^ep|DONE"
  done
  if [ "$PLACEBO" = 1 ]; then echo "=== $tag $MODEL$SUF placebo"; python -m hyfe.train_cnn --res $res --W $W --H $H --model $MODEL --cap $CAP --top $TOP --start $START --val 2025-09 --test 2025-12 --test_end 2026-03 --shuffle --epochs 3 $LRARG --out work/results/${MODEL}_${tag}${SUF}_shuffle.json 2>&1 | grep --line-buffered -E "train|^ep|DONE"; fi
done
echo ALL_DONE
