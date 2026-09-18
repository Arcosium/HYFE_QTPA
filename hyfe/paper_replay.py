"""Private retrospective simulation with frozen weights and a fixed input snapshot.

This never writes the live SQLite database or pretends predictions were generated
in the past. Decision time and actual recomputation time are separate fields.
"""
import argparse
import copy
import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hyfe import bars as B, cohort as batch, live_cnn as LC
from hyfe.live import load_buffer, recent_1m
from hyfe.paper_live import infer_bars, mark_prices
from hyfe.paper_models import ROOT, RUNTIME, atomic_json, sha, validate_manifest
from hyfe.paper_rules import STEP_MS, HOLD, RULE_VERSION, advance_cohort, signal_positions
from hyfe.paper_store import new_state
from hyfe.paper_funding import collect


def write_frame(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.parquet')
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def source_hashes():
    names = ['paper_replay.py', 'paper_live.py', 'paper_rules.py', 'paper_funding.py',
             'live_cnn.py', 'features.py', 'bars.py', 'cohort.py']
    return {name: sha(ROOT/'hyfe'/name) for name in names}


def prepare(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out/'input_manifest.json'
    if manifest_path.exists():
        info = json.loads(manifest_path.read_text())
        if info['source_hashes'] != source_hashes():
            raise ValueError('replay source changed; use a new output directory')
        for name, digest in info['files'].items():
            if sha(out/name) != digest:
                raise ValueError('input snapshot changed: '+name)
        validate_manifest(info['models'])
        return info
    models = validate_manifest(json.loads((RUNTIME/'active_models.json').read_text()), True)
    legacy = [json.loads(line) for line in (ROOT/'work/live/dir_cnn_signals.jsonl').read_text().splitlines() if line.strip()]
    with sqlite3.connect(f'file:{(RUNTIME/"paper.sqlite3").resolve()}?mode=ro', uri=True) as con:
        observed = {str(t): json.loads(p)['universe'] for t, p in con.execute('SELECT ts,payload FROM decisions ORDER BY ts')}
    start, cutover, end = min(int(d['ts']) for d in legacy), min(map(int, observed)), max(map(int, observed))
    if pd.Timestamp(models['data_cutoff']).timestamp()*1000 > start:
        raise ValueError('model data cutoff is after replay start')
    first_month = pd.Timestamp(start, unit='ms', tz='UTC').strftime('%Y-%m')
    last_month = pd.Timestamp(end, unit='ms', tz='UTC').strftime('%Y-%m')
    if first_month != last_month:
        raise ValueError('this replay requires one frozen monthly universe per run')
    universe = json.loads((RUNTIME/f'universe-{first_month}.json').read_text())
    if universe['cutoff_ms'] > start:
        raise ValueError('universe cutoff is after replay start')
    atomic_json(out/'universe.json', universe)
    atomic_json(out/'observed_decisions.json', observed)
    atomic_json(out/'legacy_signal_snapshot.json', legacy)
    buf = load_buffer()
    coverage, files = {}, {}
    for index, base in enumerate(universe['bases']):
        raw = recent_1m(base, buf)
        if raw is None:
            continue
        raw = raw[raw.ts < end].copy()
        if raw.empty:
            continue
        coverage[base] = {'first_minute': int(raw.ts.min()), 'last_minute': int(raw.ts.max())}
        for resolution, minutes in [('4h', 240), ('15m', 15)]:
            bars = B.to_res(raw, minutes)
            if bars is None:
                continue
            path = out/'inputs'/resolution/(base+'.parquet')
            write_frame(path, bars[bars.ts + minutes*60_000 <= end])
            files[str(path.relative_to(out))] = sha(path)
        if (index+1) % 25 == 0:
            print(json.dumps({'phase':'snapshot','symbols':index+1,'total':len(universe['bases'])}), flush=True)
    for name in ['universe.json', 'observed_decisions.json', 'legacy_signal_snapshot.json']:
        files[name] = sha(out/name)
    info = {'kind':'retrospective_simulation', 'rule_version':RULE_VERSION,
            'created_ms':int(time.time()*1000), 'start_ms':start, 'cutover_ms':cutover, 'end_ms':end,
            'models':models, 'bases':universe['bases'], 'coverage':coverage, 'files':files,
            'source_hashes':source_hashes(), 'notes':[
                'Frozen existing weights; no training or parameter selection.',
                'All feature inputs end at or before each decision close.',
                'Consolidated historical data cannot recreate original feed delays or later corrections.',
                'Paper-close valuation, linear roundtrip fees and Binance funding proxy; no actual fills.',
                'Hypothetical decisions every 4h, including times when the old worker may have missed a tick.',
                'No retrospective decisions are inserted into the live ledger.']}
    atomic_json(manifest_path, info)
    return info


def simulate(signals, four, funding):
    times = sorted(signals)
    state = new_state(times[0])
    curve, archived = [], []
    for ts in times:
        if ts != times[0]:
            if ts != state['last_ts'] + STEP_MS:
                raise ValueError('replay decision grid has a gap')
            prices = mark_prices(four, ts)
            outcomes = [advance_cohort(c, prices, funding[ts]) for c in state['cohorts']]
            for variant, book in state['books'].items():
                for kind in ['net', 'funded']:
                    delta = float(np.mean([r[variant][kind] for r in outcomes])) if outcomes else 0.
                    if delta <= -1:
                        raise ValueError('replay portfolio insolvent')
                    book[kind] *= 1 + delta
                book['funding_coverage'] = float(np.mean([r[variant]['funding_coverage'] for r in outcomes])) if outcomes else 1.
            expired = [c for c in state['cohorts'] if c['age'] >= HOLD]
            archived.extend(expired)
            state['n_closed'] += len(expired)
            state['n_closed_positions'] += sum(len(c['positions']) for c in expired)
            state['cohorts'] = [c for c in state['cohorts'] if c['age'] < HOLD]
        state['cohorts'].append({'id':ts, 'age':0, 'exit_ts':ts + HOLD*STEP_MS,
                                 'positions':signal_positions(signals[ts])})
        state['last_ts'] = ts
        curve.append({'ts':ts, 'active_cohorts':len(state['cohorts']),
                      **{variant+'_'+key:value for variant, book in state['books'].items() for key,value in book.items()}})
    return state, pd.DataFrame(curve), archived


def validate_batch(out, signals, funding, curve):
    """An independent cohort-path aggregation must match incremental accounting."""
    all_signals = pd.concat(signals.values(), ignore_index=True)
    score = all_signals.score.to_numpy()
    open_ts = all_signals.ts.to_numpy() - STEP_MS
    pred = out/'replay_pred.npz'
    np.savez_compressed(pred, ts=open_ts, base=all_signals.base.to_numpy(str),
                        p=np.c_[np.zeros(len(score)), score.clip(0), (-score).clip(0)])
    weight = out/'replay_event_weights.npz'
    np.savez_compressed(weight, ts=open_ts, base=all_signals.base.to_numpy(str),
                        w=all_signals.groupby('ts').event.rank(pct=True).to_numpy())
    fr = pd.DataFrame([{'ts':ts-STEP_MS, 'base':base, 'rate':rate}
                       for ts, rates in funding.items() for base, rate in rates.items()])
    funding_file = out/'replay_funding.parquet'
    write_frame(funding_file, fr)
    old = B.OUT
    checks = {}
    try:
        B.OUT = str(out/'inputs')
        for variant in ['equal', 'event']:
            for kind in ['net', 'funded']:
                _, daily = batch.run(str(pred), '4h', HOLD,
                                     weight=str(weight) if variant == 'event' else '',
                                     funding=str(funding_file) if kind == 'funded' else '')
                expected = float(curve.iloc[-1][variant+'_'+kind])
                error = abs(float(daily.iloc[-1]) - expected)
                if error > 1e-11:
                    raise ValueError(f'batch/stream mismatch: {variant} {kind}, error={error}')
                checks[variant+'_'+kind] = error
    finally:
        B.OUT = old
    return checks


def run(out, prepare_only=False):
    import lightgbm as lgb
    out = Path(out).resolve()
    # Live data files can only ever be read by this command.
    if out == RUNTIME.resolve() or out == (ROOT/'work/live').resolve():
        raise ValueError('replay output must be a separate directory')
    torch.set_num_threads(2)
    info = prepare(out)
    if prepare_only:
        return {'status':'snapshot_ready', 'symbols':len(info['coverage'])}
    four = {p.stem:pd.read_parquet(p) for p in (out/'inputs/4h').glob('*.parquet')}
    quarter = {p.stem:pd.read_parquet(p) for p in (out/'inputs/15m').glob('*.parquet')}
    models = [LC.load_model(m['stem']) for m in info['models']['models']]
    event = lgb.Booster(model_file=info['models']['event_model']['path'])
    signals, records = {}, []
    times = list(range(info['start_ms'], info['end_ms']+1, STEP_MS))
    for index, ts in enumerate(times):
        path = out/'signals'/f'{ts}.parquet'
        start = time.time()
        if not path.exists():
            frame = infer_bars(ts, info['bases'], four, quarter, info['models'], models, event)
            write_frame(path, frame)
        else:
            frame = pd.read_parquet(path)
        if not (frame.ts == ts).all():
            raise ValueError('signal cache timestamp mismatch')
        signal_positions(frame)
        signals[ts] = frame
        records.append({'decision_ms':ts, 'recomputed_ms':int(path.stat().st_mtime*1000),
                        'kind':'retrospective_simulation', 'model_version':info['models']['version'],
                        'symbols':len(frame), 'sha256':sha(path)})
        print(json.dumps({'phase':'inference', 'done':index+1, 'total':len(times),
                          'decision':ts, 'symbols':len(frame), 'seconds':round(time.time()-start,2)}), flush=True)
    atomic_json(out/'decision_provenance.json', records)
    selected = {p['base'] for frame in signals.values() for p in signal_positions(frame)}
    print(json.dumps({'phase':'funding','symbols':len(selected)}), flush=True)
    funding = collect(out, times[1:], selected)
    atomic_json(out/'funding_by_close.json', funding)
    state, curve, archived = simulate(signals, four, funding)
    checks = validate_batch(out, signals, funding, curve)
    observed = json.loads((out/'observed_decisions.json').read_text())
    comparisons = []
    for when, universe in observed.items():
        ts = int(when)
        actual = pd.DataFrame(universe).set_index('base').sort_index()
        replay = signals[ts].set_index('base').sort_index()
        same_bases = actual.index.equals(replay.index)
        common = actual.index.intersection(replay.index)
        numeric = ['entry', 'score', 'event'] + [f'seed{i}' for i in range(10)]
        difference = float(np.max(np.abs(actual.loc[common,numeric].to_numpy()-replay.loc[common,numeric].to_numpy())))
        actual_positions = {(p['base'],p['side']) for p in signal_positions(pd.DataFrame(universe))}
        replay_positions = {(p['base'],p['side']) for p in signal_positions(signals[ts])}
        comparisons.append({'decision_ms':ts, 'same_universe':same_bases,
                            'same_selections':actual_positions == replay_positions,
                            'max_absolute_numeric_difference':difference})
    validate_manifest(info['models'])
    checkpoints = {}
    for name, ts in [('cutover',info['cutover_ms']), ('latest',info['end_ms'])]:
        part = curve[curve.ts <= ts]
        checkpoints[name] = {'ts':ts, 'variants':{}}
        for variant in ['equal','event']:
            values = part[variant+'_funded']
            checkpoints[name]['variants'][variant] = {
                'return_after_fees':float(part.iloc[-1][variant+'_net']-1),
                'return_after_fees_and_proxy_funding':float(values.iloc[-1]-1),
                'mdd':float((values/values.cummax()-1).min()),
                'minimum_funding_coverage':float(part[variant+'_funding_coverage'].min()),
                'sharpe':None}
    write_frame(out/'equity_curve.parquet', curve)
    curve.to_csv(out/'equity_curve.csv',index=False)
    atomic_json(out/'ending_state.json', state)
    atomic_json(out/'closed_cohorts.json', archived)
    result = {'kind':'retrospective_simulation', 'visibility':'internal_only', 'rule_version':RULE_VERSION,
              'start_ms':info['start_ms'], 'end_ms':info['end_ms'], 'cutover_ms':info['cutover_ms'],
              'completed_ms':int(time.time()*1000), 'decisions':len(signals),
              'active_cohorts':len(state['cohorts']), 'closed_cohorts':state['n_closed'],
              'model_version':info['models']['version'], 'models_retrained':False,
              'checkpoints':checkpoints, 'batch_stream_error':checks,
              'observed_live_comparison':comparisons, 'notes':info['notes']+[
                  'The interval is shorter than 14 days; all cohorts remain open at the endpoint.',
                  'No annualized Sharpe estimate is reported for this short interval.']}
    atomic_json(out/'summary.json', result)
    return {'status':'complete','decisions':len(signals),'active_cohorts':len(state['cohorts']),
            'live_comparison_same_selections':all(c['same_selections'] for c in comparisons),
            'batch_stream_checks':'passed','out':str(out)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.out,args.prepare_only)),flush=True)
