#!/bin/bash
# 확장 실험 착수(9/10): pod 20대 생성·준비·배정(ext_run) → 끝나면 수거·종료 데몬(HYFE_MAX_PODS=1) 기동. 실행 중 수정 금지(555).
cd /home/arcosium/projects/HYFE_QTPA
PYTHONPATH=. python3 work/ext_run.py work/ext_jobs.txt > work/ext_run.log 2>&1
grep -q PODS_DISPATCHED work/ext_run.log && HYFE_MAX_PODS=1 PYTHONPATH=. setsid nohup python3 -m hyfe.fleet daemon >> work/daemon.log 2>&1 < /dev/null &
# 잔고 감시: 10분마다 잔고·pod 로그 새 줄, 잔고 < $4 면 hyfe- pod 전부 종료(과금 방어). 끝나면 pkill -f "ext_launc[h]"
while true; do
  b=$(PYTHONPATH=. python3 -m hyfe.runpod balance 2>/dev/null | python3 -c "import sys,ast; print(ast.literal_eval(sys.stdin.read())['clientBalance'])" 2>/dev/null)
  n=$(python3 -c "import json;print(len(json.load(open('work/fleet.json'))))" 2>/dev/null)
  echo "$(date '+%m-%d %H:%M') balance $b pods $n queue $(grep -c -v '^#' work/queue.txt 2>/dev/null || echo 0)" >> work/ext_watch.log
  PYTHONPATH=. python3 -m hyfe.fleet tail >> work/ext_watch.log 2>/dev/null
  if [ -n "$b" ] && python3 -c "import sys; sys.exit(0 if float('$b') < 4 else 1)"; then echo "LOW BALANCE → term all" >> work/ext_watch.log; PYTHONPATH=. python3 -m hyfe.fleet term all >> work/ext_watch.log 2>&1; fi
  sleep 600
done
