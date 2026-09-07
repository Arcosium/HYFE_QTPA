#!/bin/bash
# S3 표현 — 설정(res:W:H) 위에서 모델·렌더를 바꾼다. 각 2분할.
# usage: [LR=1e-4] [EXTRA="--k 1.5"] bash hyfe/s3.sh 15m:120:10 i2 j2 f1 candle line
set -u
cd "$(dirname "$0")/.."
TOP=${TOP:-200}; START=${START:-2023-01}; LR=${LR:-}; EXTRA=${EXTRA:-}; CAP=${CAP:-2000000}
SUF=${LR:+_lr$LR}
IFS=: read res W H <<< "$1"; shift; tag="${res}_W${W}_H${H}"
for v in "$@"; do
  case $v in
    i2|j2) name="${v}_${tag}"; args="--model $v --cap $CAP" ;;             # i2 는 lr 기본 1e-4, j2 5e-4 — LR 은 i1 계열(i1·f1)에만
    f1) name="${v}_${tag}"; args="--model f1 --cap $CAP ${LR:+--lr $LR}" ;;
    candle|line) name="${MODEL:-i1}_${tag}_${v}"; args="--model ${MODEL:-i1} --render $v --cap $CAP ${LR:+--lr $LR}" ;;
    *) echo "unknown $v"; continue ;;
  esac
  for s in 0 1; do
    if [ $s = 0 ]; then V=2025-09; T=2025-12; TE=2026-03; else V=2025-06; T=2025-09; TE=2025-12; fi
    echo "=== S3 $name split$s"; python -m hyfe.train_cnn --res $res --W $W --H $H $args $EXTRA --top $TOP --start $START --val $V --test $T --test_end $TE --out work/results/${name}${SUF}_s${s}.json --save_pred 2>&1 | grep --line-buffered -E "train|^ep|DONE"
  done
done
echo S3_DONE
