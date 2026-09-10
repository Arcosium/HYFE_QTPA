#!/bin/bash
# 방향 본 학습 — 폴드별 유동성 상위 종목(work/liq12_s<i>.txt, 판단 시점 이전 12개월 거래대금 = GBM 과 같은 유니버스)으로 롤링 4폴드.
# usage: [LABEL=relbin STRIDE=1 CHANNELS=gray NOVOL= DETREND= SEQCTX= LR= EPOCHS= CAP=5000000 ROWPERM= COLPERM= DROP= LOSS=] bash hyfe/fulldir.sh j2 4h:60:42 [0 1 2 3]
set -u
cd "$(dirname "$0")/.."
model=$1; IFS=: read res W H <<< "$2"; shift 2; folds=${*:-0 1 2 3}; tag="${res}_W${W}_H${H}"
CAP=${CAP:-5000000}; STRIDE=${STRIDE:-1}; CHANNELS=${CHANNELS:-gray}; LABEL=${LABEL:-relbin}; NOVOL=${NOVOL:-}; DETREND=${DETREND:-}; SEQCTX=${SEQCTX:-}; RENDER=${RENDER:-}; LR=${LR:-}; EPOCHS=${EPOCHS:-}; SEED=${SEED:-0}
ROWPERM=${ROWPERM:-}; COLPERM=${COLPERM:-}; DROP=${DROP:-}; LOSS=${LOSS:-}   # heatf 확장 실험(행 순열·봉 순열·행군 제거·z 회귀)
EXTRA="${LR:+--lr $LR} ${EPOCHS:+--epochs $EPOCHS} ${NOVOL:+--no_volume} ${DETREND:+--detrend} ${SEQCTX:+--seq_ctx} ${RENDER:+--render $RENDER} ${ROWPERM:+--row_perm $ROWPERM} ${COLPERM:+--col_perm $COLPERM} ${DROP:+--drop $DROP} ${LOSS:+--loss $LOSS} --seed $SEED"
SUF=$([ "$CHANNELS" != gray ] && echo _$CHANNELS)$([ -n "$NOVOL" ] && echo _novol)$([ -n "$DETREND" ] && echo _detrend)$([ -n "$SEQCTX" ] && echo _ctx)$([ -n "$RENDER" ] && echo _$RENDER)$([ -n "$ROWPERM" ] && echo _rp$ROWPERM)$([ -n "$COLPERM" ] && echo _cp$COLPERM)$([ -n "$DROP" ] && echo _drop$DROP)$([ -n "$LOSS" ] && echo _$LOSS)$([ "$SEED" != 0 ] && echo _sd$SEED)$([ -n "$LR" ] && echo _lr$LR)_liq
for s in $folds; do
  case $s in 0) V=2025-09; T=2025-12; TE=2026-03;; 1) V=2025-06; T=2025-09; TE=2025-12;; 2) V=2024-09; T=2024-12; TE=2025-03;; 3) V=2024-03; T=2024-06; TE=2024-09;; esac
  echo "=== FULLDIR $model$SUF $tag split$s"
  python -m hyfe.train_cnn --res $res --W $W --H $H --model $model --channels $CHANNELS --label $LABEL --stride $STRIDE --bases @work/liq12_s$s.txt $EXTRA --cap $CAP --start 2023-01 --val $V --test $T --test_end $TE --out work/results/full_${model}${SUF}_${tag}_s${s}.json --save_pred 2>&1 | grep --line-buffered -E "train|^ep|DONE|Error|Traceback"
done
echo FULL_DONE
