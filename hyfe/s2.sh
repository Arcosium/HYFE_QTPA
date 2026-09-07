#!/bin/bash
# S2 거래량·라벨·지평 — 설정(res:W:H) 위에서 한 축씩 바꾼다. 변형은 I1 2분할, 라벨·지평 변형은 GBM 기준선도 같이.
# usage: [LR=1e-4] bash hyfe/s2.sh 15m:120:10 novol k1.5 k3 fixed binary h4 h2   (변형 이름 생략 시 전부)
set -u
cd "$(dirname "$0")/.."
TOP=${TOP:-200}; START=${START:-2023-01}; LR=${LR:-}; MODEL=${MODEL:-i1}; CAP=${CAP:-2000000}
LRARG=${LR:+--lr $LR}; SUF=${LR:+_lr$LR}
IFS=: read res W H <<< "$1"; shift; tag="${res}_W${W}_H${H}"
VARS=${*:-novol k1.5 k3 fixed binary h4 h2}
run() {  # name, H, extra cnn args, extra gbm args ("" 이면 GBM 생략)
  name=$1; HH=$2; cnn=$3; gbm=$4
  if [ -n "$gbm" ]; then echo "=== S2 $name GBM"; python -m hyfe.pilot_gbm --top $TOP --start $START --configs $res:$W:$HH $gbm --save_pred --out work/results/gbm_${name}.csv 2>&1 | grep --line-buffered -v Warning; fi
  for s in 0 1; do
    if [ $s = 0 ]; then V=2025-09; T=2025-12; TE=2026-03; else V=2025-06; T=2025-09; TE=2025-12; fi
    echo "=== S2 $name $MODEL$SUF split$s"; python -m hyfe.train_cnn --res $res --W $W --H $HH --model $MODEL --cap $CAP --top $TOP --start $START --val $V --test $T --test_end $TE $cnn $LRARG --out work/results/${MODEL}_${name}${SUF}_s${s}.json --save_pred 2>&1 | grep --line-buffered -E "train|^ep|DONE"
  done
}
for v in $VARS; do
  case $v in
    novol)  run "${tag}_novol"  $H "--no_volume" "" ;;
    k1.5)   run "${tag}_k1.5"   $H "--k 1.5" "--k 1.5" ;;
    k3)     run "${tag}_k3"     $H "--k 3" "--k 3" ;;
    fixed)  FIXED=$(python -m hyfe.pick fixed $res $W $H 2>/dev/null | tail -1); run "${tag}_fixed" $H "--label fixed --fixed $FIXED" "--label fixed --fixed $FIXED" ;;
    binary) run "${tag}_binary" $H "--label binary" "--label binary" ;;
    h4)     run "${res}_W${W}_H$((W/4))" $((W/4)) "" "--k 2" ;;
    h2)     run "${res}_W${W}_H$((W/2))" $((W/2)) "" "--k 2" ;;
  esac
done
echo S2_DONE
