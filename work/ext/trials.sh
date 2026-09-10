#!/bin/bash
# DSR 시도 분포용(확장 ②): 상대방향 설정(rel/relbin/liq/heat*/m1/m2/wf) 전부를 자기 (격자, 지평)의 코호트 롱숏으로 판정. 이미 있으면 건너뜀. 15m 발생·절대방향 모델은 다른 과제라 제외.
cd /home/arcosium/projects/HYFE_QTPA; export HYFE_BARS=work/bars_full
ls work/results/*_pred.npz | grep -v -E "ospred|/ens|/w_|gate|_sd[0-9]+_|krx_|hold_|teacher|shuf|15m|ens15h" | grep -E "rel|_liq_|heat|_m[12]_|/wf_" | while read f; do
  [[ $f =~ _(1h|4h|1d)_W([0-9]+)_H([0-9]+)_s[0-9]_pred\.npz$ ]] || continue
  res=${BASH_REMATCH[1]}; H=${BASH_REMATCH[3]}; out=${f/_pred.npz/_cohort_H$H.json}
  [ -f "$out" ] || echo "python3 -m hyfe.cohort --pred $f --res $res --H $H --cost 0.001 > /dev/null 2>&1"
done | xargs -P 4 -I{} bash -c "{}"
echo TRIALS_DONE
