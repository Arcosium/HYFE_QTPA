#!/bin/bash
# 전수 본 학습 — 모델(i1|j2|f1) × 설정(res:W:H) 을 740종목·학습 500만 행으로 롤링 4폴드. 결과 work/results/full_<model>_<tag>_s<i>.json
# usage: bash hyfe/full.sh j2 15m:120:10 [0 1 2 3]
set -u
cd "$(dirname "$0")/.."
model=$1; IFS=: read res W H <<< "$2"; shift 2; folds=${*:-0 1 2 3}; tag="${res}_W${W}_H${H}"
CAP=${CAP:-5000000}; STRIDE=${STRIDE:-0}; CHANNELS=${CHANNELS:-gray}; LABEL=${LABEL:-ksigma}; TOPN=${TOPN:-}; NOVOL=${NOVOL:-}; DETREND=${DETREND:-}; UNI=${TOPN:+--top $TOPN}; UNI=${UNI:---universe G4}; EXTRA="${LR:+--lr $LR} ${PX:+--px $PX} ${IMGH:+--img_h $IMGH} ${EPOCHS:+--epochs $EPOCHS}"
SUF=$([ "$CHANNELS" != gray ] && echo _$CHANNELS)$([ "$LABEL" != ksigma ] && echo _$LABEL)$([ -n "$NOVOL" ] && echo _novol)$([ -n "$DETREND" ] && echo _detrend)$([ -n "$TOPN" ] && [ "$TOPN" != 200 ] && echo _top$TOPN)$([ -n "$LR" ] && echo _lr$LR)$([ -n "$PX" ] && echo _px$PX)$([ -n "$IMGH" ] && echo _h$IMGH)
for s in $folds; do
  case $s in 0) V=2025-09; T=2025-12; TE=2026-03;; 1) V=2025-06; T=2025-09; TE=2025-12;; 2) V=2024-09; T=2024-12; TE=2025-03;; 3) V=2024-03; T=2024-06; TE=2024-09;; esac
  echo "=== FULL $model$SUF $tag split$s"
  python -m hyfe.train_cnn --res $res --W $W --H $H --model $model --channels $CHANNELS --label $LABEL --stride $STRIDE ${NOVOL:+--no_volume} ${DETREND:+--detrend} $UNI $EXTRA --cap $CAP --start 2023-01 --val $V --test $T --test_end $TE --out work/results/full_${model}${SUF}_${tag}_s${s}.json --save_pred 2>&1 | grep --line-buffered -E "train|^ep|DONE"
done
echo FULL_DONE
