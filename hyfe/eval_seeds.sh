#!/bin/bash
# 시드 검증표: heatf 소형 CNN 4h·60·84 — 시드 전부(사전 등록 0~9, 누락 없이) × 4폴드 코호트 Sharpe 분포 + 전 시드 z-평균 앙상블 + 홀드아웃(s4) 같은 표
cd /home/arcosium/projects/HYFE_QTPA; R=work/results; HS=${HS:-4}   # 홀드아웃 분할(4=2026-03~08, 5=롤링 06~08)
coh() { [ -f "${1/_pred.npz/_cohort_H84.json}" ] || python3 -m hyfe.cohort --pred $1 --res 4h --H 84 --cost 0.001 > /dev/null 2>&1; }
export HYFE_BARS=work/bars_full
for f in $R/full_i1_heatf_liq_4h_W60_H84_s[0-3]_pred.npz $R/full_i1_heatf_sd[0-9]_liq_4h_W60_H84_s[0-3]_pred.npz; do [ -f "$f" ] && coh $f; done
for s in 0 1 2 3; do fs=$(ls $R/full_i1_heatf_liq_4h_W60_H84_s${s}_pred.npz $R/full_i1_heatf_sd[0-9]_liq_4h_W60_H84_s${s}_pred.npz 2>/dev/null); n=$(echo $fs | wc -w)
  python3 -m hyfe.zavg $fs --out $R/ens_heatf${n}_direct_4h_s${s}_pred.npz > /dev/null 2>&1 && coh $R/ens_heatf${n}_direct_4h_s${s}_pred.npz; done
export HYFE_BARS=work/bars_hold
for f in $R/hold_i1_heatf_4h_W60_H84_s${HS}_pred.npz $R/hold_i1_heatf_sd[0-9]_4h_W60_H84_s${HS}_pred.npz; do [ -f "$f" ] && coh $f; done
fs=$(ls $R/hold_i1_heatf_4h_W60_H84_s${HS}_pred.npz $R/hold_i1_heatf_sd[0-9]_4h_W60_H84_s${HS}_pred.npz 2>/dev/null); n=$(echo $fs | wc -w)
python3 -m hyfe.zavg $fs --out $R/ens_heatf${n}_hold_4h_s${HS}_pred.npz > /dev/null 2>&1 && coh $R/ens_heatf${n}_hold_4h_s${HS}_pred.npz
python3 - <<'PY'
import json, glob, re, pandas as pd, numpy as np
from hyfe.perf import daily_returns, stats
R="work/results"; HS=__import__("os").environ.get("HS","4"); sd=lambda f:(re.search(r'_sd(\d+)_',f) or [None,'0'])[1]
def load(pat):
    return {(sd(f), re.search(r'_s(\d)_cohort',f)[1]): f for f in glob.glob(pat)}
full=load(f'{R}/full_i1_heatf*_liq_4h_W60_H84_s[0-3]_cohort_H84.json'); hold=load(f'{R}/hold_i1_heatf*_4h_W60_H84_s{HS}_cohort_H84.json')
seeds=sorted({k[0] for k in full}|{k[0] for k in hold}, key=int); rows=[]
for sdn in seeds:
    r={'seed':sdn}
    for s in '0123':
        f=full.get((sdn,s)); r[f's{s}']=json.load(open(f))['summary']['sharpe'] if f else np.nan
    fs=[full[(sdn,s)] for s in '0123' if (sdn,s) in full]
    if len(fs)==4: rr=pd.concat([daily_returns(f)[0] for f in fs]); st=stats(rr,(1+rr).cumprod(),14); r['pooled']=st['sharpe']; r['p']=st['p']; r['mdd']=st['mdd']
    f=hold.get((sdn,HS)); r['hold']=json.load(open(f))['summary']['sharpe'] if f else np.nan; rows.append(r)
df=pd.DataFrame(rows); pd.set_option('display.width',200); print(df.to_string(index=False))
num=df.drop(columns='seed').astype(float)
print('\n시드 분포 (n=%d): 평균 / 표준편차 / 최소 / 최대 / Sharpe>1 비율'%len(df))
print(pd.DataFrame({'mean':num.mean().round(2),'sd':num.std().round(2),'min':num.min().round(2),'max':num.max().round(2),'>1':(num>1).mean().round(2)}).T.to_string())
for pat,lag,name in [(f'{R}/ens_heatf*_direct_4h_s[0-3]_cohort_H84.json',14,'폴드 앙상블'),(f'{R}/ens_heatf*_hold_4h_s{HS}_cohort_H84.json',14,'홀드아웃 앙상블')]:
    g={}
    for f in glob.glob(pat): g.setdefault(re.search(r'ens_heatf(\d+)_',f)[1],[]).append(f)
    for n,fs in sorted(g.items(),key=lambda x:int(x[0])):
        fs=sorted(fs); per=[json.load(open(f))['summary'] for f in fs]; rr=pd.concat([daily_returns(f)[0] for f in fs])
        print(f'\n{name} 시드 {n}개: 폴드별 Sharpe', [p['sharpe'] for p in per], 'p', [p['p'] for p in per], '| 합산', stats(rr,(1+rr).cumprod(),lag))
PY
