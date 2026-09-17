#!/bin/bash
# 페이퍼 트레이딩 틱 — cron 이 매시 12분에 부른다(10분 flush 뒤). flock 으로 중복 실행 방지, 로그 work/live/tick.log
cd /home/arcosium/projects/HYFE_QTPA || exit 1
mkdir -p work/live
export HYFE_BARS=work/bars_hold
(
  /usr/bin/flock -n 9 || exit 0
  if [ -f work/paper_trade/enabled.json ]; then
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /usr/bin/python3 -m hyfe.paper_live >> work/paper_trade/worker.log 2>&1
  fi
  /usr/bin/python3 -m hyfe.live tick >> work/live/tick.log 2>&1
) 9>work/live/tick.lock
