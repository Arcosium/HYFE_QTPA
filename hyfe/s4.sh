#!/bin/bash
# S4 종목군 — 설정(res:W:H) 의 I1 을 종목군별로 학습, 평가는 항상 전체 홀드아웃. 각 2분할 + GBM.
# usage: [LR=1e-4] [EXTRA="--k 1.5"] bash hyfe/s4.sh 15m:120:10 G1 G2 G3 G4 G5 G4out G5out
set -u
cd "$(dirname "$0")/.."
START=${START:-2023-01}; LR=${LR:-}; EXTRA=${EXTRA:-}; MODEL=${MODEL:-i1}; CAP=${CAP:-2000000}
LRARG=${LR:+--lr $LR}; SUF=${LR:+_lr$LR}
IFS=: read res W H <<< "$1"; shift; tag="${res}_W${W}_H${H}"
for v in "$@"; do
  g=${v%out}; d=in; [ "$v" != "$g" ] && d=out
  name="${tag}_${v}"
  echo "=== S4 $name GBM"; python -m hyfe.pilot_gbm --universe $g --delisted $d --start $START --configs $res:$W:$H $EXTRA --save_pred --out work/results/gbm_${name}.csv 2>&1 | grep --line-buffered -v Warning
  for s in 0 1; do
    if [ $s = 0 ]; then V=2025-09; T=2025-12; TE=2026-03; else V=2025-06; T=2025-09; TE=2025-12; fi
    echo "=== S4 $name $MODEL$SUF split$s"; python -m hyfe.train_cnn --res $res --W $W --H $H --model $MODEL --cap $CAP --universe $g --delisted $d $EXTRA --start $START --val $V --test $T --test_end $TE $LRARG --out work/results/${MODEL}_${name}${SUF}_s${s}.json --save_pred 2>&1 | grep --line-buffered -E "train|^ep|DONE"
  done
done
echo S4_DONE
