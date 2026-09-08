#!/bin/bash
# KRX 일봉 판 — heatf 이미지 소형 CNN 직접학습(상대부호 라벨), 폴드별 거래대금 상위 300(work/liqkrx_s<i>.txt), HYFE_BARS=work/bars_krx
# usage: [SEED=0 RENDER=heatf H=10] bash hyfe/fullkrx.sh i1 60 [0 1 2 3]
set -u; cd "$(dirname "$0")/.."; export HYFE_BARS=work/bars_krx
model=$1; W=$2; shift 2; folds=${*:-0 1 2 3}; H=${H:-10}; RENDER=${RENDER:-heatf}; SEED=${SEED:-0}; CAP=${CAP:-3000000}
for s in $folds; do
  case $s in 0) V=2025-09; T=2025-12; TE=2026-03;; 1) V=2025-06; T=2025-09; TE=2025-12;; 2) V=2025-03; T=2025-06; TE=2025-09;; 3) V=2024-12; T=2025-03; TE=2025-06;; esac
  echo "=== KRX $model $RENDER 1d_W${W}_H${H} sd$SEED split$s"
  python -m hyfe.train_cnn --res 1d --W $W --H $H --model $model --render $RENDER --label relbin --stride 1 --bases @work/liqkrx_s$s.txt --seed $SEED --cap $CAP --start 2023-05 --val $V --test $T --test_end $TE --out work/results/krx_${model}_${RENDER}$([ "$SEED" != 0 ] && echo _sd$SEED)_1d_W${W}_H${H}_s${s}.json --save_pred 2>&1 | grep --line-buffered -E "train|^ep|DONE"
done
echo KRX_DONE
