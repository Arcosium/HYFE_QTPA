#!/bin/bash
# 증류 학생 — 교사(피처 GBM walk-forward 표본 밖 점수)를 이미지에서 배운다. 분할 A: IS 2024-03~25-05 / OS 25-06~08 / ROS 25-09~26-02, B: IS ~25-02 / OS 25-03~05 / ROS 25-06~11
# usage: [RENDER=heat|bar CHANNELS=gray|multi] bash hyfe/student.sh i1 4h:60:84 v2
set -u; cd "$(dirname "$0")/.."
model=$1; IFS=: read res W H <<< "$2"; T=$3; RENDER=${RENDER:-heat}; CHANNELS=${CHANNELS:-gray}; CAP=${CAP:-3000000}; SEED=${SEED:-0}; TXS=${TXS:-}; LR=${LR:-}; TAILW=${TAILW:-}; tag="${res}_W${W}_H${H}"
for sp in A B; do
  if [ $sp = A ]; then V=2025-06; TE=2025-09; END=2026-03; else V=2025-03; TE=2025-06; END=2025-12; fi
  echo "=== STUDENT ${model}_${RENDER}_${CHANNELS} teacher=$T $tag split$sp"
  python -m hyfe.train_cnn --res $res --W $W --H $H --model $model --render $RENDER --channels $CHANNELS --label relbin --stride 1 --bases @work/liqwf_all.txt --teacher work/results/teacher_$T.npz ${TXS:+--teacher_xs} ${LR:+--lr $LR} ${TAILW:+--tail_w $TAILW} --cap $CAP --seed $SEED --start 2024-01 --val $V --test $TE --test_end $END --out work/results/student_${model}_${RENDER}_${CHANNELS}_${T}$([ "$SEED" != 0 ] && echo _sd$SEED)$([ -n "$TXS" ] && echo _txs)$([ -n "$LR" ] && echo _lr$LR)$([ -n "$TAILW" ] && echo _tw$TAILW)_${tag}_s${sp}.json --save_pred 2>&1 | grep --line-buffered -E "train|teacher|^ep|DONE"
done
echo STUDENT_DONE
