#!/bin/bash
# 페이퍼 트레이딩 틱 — cron 이 매시 12분에 부른다(10분 flush 뒤). flock 으로 중복 실행 방지, 로그 work/live/tick.log
cd /home/arcosium/projects/HYFE_QTPA || exit 1
mkdir -p work/live
export HYFE_BARS=work/bars_hold
/usr/bin/flock -n work/live/tick.lock /usr/bin/python3 -m hyfe.live tick >> work/live/tick.log 2>&1
