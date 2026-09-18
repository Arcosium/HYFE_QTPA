"""Explicit, audited import of a verified replay into the paper ledger.

The scheduled worker remains forward-only. Build and validate a separate SQLite
candidate first; apply replaces rows in one transaction, never the open DB file.
Original live decisions/execution observations and an immutable backup survive.
"""
import argparse
import fcntl
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd

from hyfe.paper_models import RUNTIME, ROOT, atomic_json, sha, validate_manifest
from hyfe.paper_replay import simulate
from hyfe.paper_live import mark_prices
from hyfe.paper_rules import STEP_MS, HOLD, RULE_VERSION, signal_positions
from hyfe.paper_store import connect, load_state, dumps, apply_decision, internal_report
from hyfe.paper_funding import collect

TABLES = ('state', 'decisions', 'marks', 'closed', 'executions', 'health')
REBUILT = ('state', 'decisions', 'marks', 'closed')


def fingerprint(con):
    return hashlib.sha256(dumps({t: con.execute(f'SELECT * FROM {t} ORDER BY 1,2,3,4' if t == 'executions'
                                               else f'SELECT * FROM {t} ORDER BY 1').fetchall()
                                for t in TABLES}).encode()).hexdigest()


def read_verified(replay):
    artifacts = json.loads((replay/'artifact_manifest.json').read_text())
    info = json.loads((replay/'input_manifest.json').read_text())
    for name, digest in {**artifacts['files'], **info['files']}.items():
        path = (replay/name).resolve()
        if not path.is_relative_to(replay.resolve()) or sha(path) != digest:
            raise ValueError('replay checksum mismatch: '+name)
    if info['rule_version'] != RULE_VERSION:
        raise ValueError('replay rule mismatch')
    validate_manifest(info['models'])
    verification = json.loads((replay/'verification.json').read_text())
    if verification['status'] != 'passed' or not verification['live_scores_exactly_equal']:
        raise ValueError('replay not verified')
    return info


def build_candidate(source, candidate, replay):
    """No changes to source. Replay first, then retain every later live boundary."""
    replay, candidate = Path(replay), Path(candidate)
    if candidate.exists() or candidate.resolve() == Path(source).resolve():
        raise ValueError('candidate must be a new file')
    info = read_verified(replay)
    with connect(source, readonly=True) as live:
        live.execute('BEGIN')
        before = fingerprint(live)
        original = load_state(live)
        rows = {t: live.execute(f'SELECT * FROM {t} ORDER BY 1').fetchall() for t in TABLES}
    if original.get('history_import'):
        raise ValueError('ledger already has a replay import')
    if original['started'] != info['cutover_ms'] or original['last_ts'] < info['end_ms']:
        raise ValueError('live/replay interval mismatch')
    live_decisions = {ts: (ts, generated, model, payload) for ts, generated, model, payload in rows['decisions']}
    provenance = {p['decision_ms']: p for p in json.loads((replay/'decision_provenance.json').read_text())}
    times = list(range(info['start_ms'], info['end_ms']+1, STEP_MS))
    signals = {ts: pd.read_parquet(replay/'signals'/f'{ts}.parquet') for ts in times}
    four = {p.stem: pd.read_parquet(p) for p in (replay/'inputs/4h').glob('*.parquet')}
    funding = {int(k): v for k, v in json.loads((replay/'funding_by_close.json').read_text()).items()}
    state, curve, closed = simulate(signals, four, funding)
    expected = pd.read_parquet(replay/'equity_curve.parquet')
    pd.testing.assert_frame_equal(curve, expected, check_exact=False, rtol=1e-12, atol=1e-12)
    decisions = []
    metadata = {}
    for ts, frame in signals.items():
        if ts in live_decisions:
            record = live_decisions[ts]
            actual = pd.DataFrame(json.loads(record[3])['universe']).set_index('base').sort_index()
            replay_frame = frame.set_index('base').sort_index()
            pd.testing.assert_frame_equal(actual[replay_frame.columns], replay_frame,
                                          check_dtype=False, check_exact=True)
            metadata[ts] = {'origin': 'forward', 'generated_ms': record[1], 'model_version': record[2]}
        else:
            p = provenance[ts]
            metadata[ts] = {'origin': 'retrospective', 'generated_ms': p['recomputed_ms'],
                            'model_version': p['model_version']}
            cohort = {'id': ts, 'age': 0, 'exit_ts': ts+HOLD*STEP_MS,
                      **metadata[ts], 'positions': signal_positions(frame)}
            record = (ts, p['recomputed_ms'], p['model_version'], dumps({
                'rule_version': RULE_VERSION, 'origin': 'retrospective',
                'universe': frame.to_dict(orient='records'), 'cohort': cohort}))
        decisions.append(record)
    for cohort in state['cohorts']+closed:
        cohort.update(metadata[cohort['id']])
    frame = signals[times[-1]]
    state.update(model_version=info['models']['version'], updated_ms=original['updated_ms'],
                 last_signal={'ts': times[-1], 'n': len(frame), 'n_positions': len(signal_positions(frame)),
                              'top': frame.nlargest(5, 'score')[['base','score']].values.tolist(),
                              'bottom': frame.nsmallest(5, 'score')[['base','score']].values.tolist()})
    state['history_import'] = {'kind': 'reconstructed_paper', 'replay_start_ms': times[0],
                               'replay_end_ms': times[-1], 'forward_since_ms': info['cutover_ms'],
                               'imported_ms': int(time.time()*1000),
                               'replay_sha256': sha(replay/'artifact_manifest.json'),
                               'retrospective_decisions': sum(m['origin']=='retrospective' for m in metadata.values())}
    con = connect(candidate)
    with con:
        con.execute('INSERT INTO state VALUES (1,?)', (dumps(state),))
        con.executemany('INSERT INTO decisions VALUES (?,?,?,?)', decisions)
        con.executemany('INSERT INTO closed VALUES (?,?)', [(c['id'],dumps(c)) for c in closed])
        for index, ts in enumerate(times[1:], 1):
            point = curve.iloc[index]
            mark = {'books': {v: {k: float(point[v+'_'+k]) for k in ('net','funded','funding_coverage')}
                              for v in ('equal','event')},
                    'n_active': min(index,HOLD), 'closed': int(index>=HOLD), 'signal_missing': False,
                    'origin': 'reconstructed', 'prices': mark_prices(four,ts), 'funding': funding[ts]}
            con.execute('INSERT INTO marks VALUES (?,?)', (ts,dumps(mark)))
    # A later live tick may have run since the original replay was calculated.
    # Settle the imported holdings too, using the recorded mark and funding cache.
    later = {ts: json.loads(p) for ts,p in rows['marks'] if ts > times[-1]}
    if later:
        selected = {p['base'] for c in state['cohorts'] for p in c['positions']}
        for ts, record in live_decisions.items():
            if ts > times[-1]:
                selected.update(p['base'] for p in json.loads(record[3])['cohort']['positions'])
        extra_funding = collect(candidate.parent/'funding', sorted(later), selected)
        for ts, mark in sorted(later.items()):
            if ts in live_decisions:
                _, generated, model, payload = live_decisions[ts]
                frame = pd.DataFrame(json.loads(payload)['universe'])
            else:
                generated, model, frame = ts+STEP_MS, state['model_version'], None
            apply_decision(con,ts,generated,model,frame,{ts:mark['prices']},{ts:extra_funding[ts]})
    with con:
        # Preserve every original inference timestamp, decision payload and
        # measured execution reference, including observations after the replay.
        con.executemany('INSERT OR REPLACE INTO decisions VALUES (?,?,?,?)',rows['decisions'])
        con.execute('DELETE FROM executions')
        con.executemany('INSERT INTO executions VALUES (?,?,?,?,?,?,?)',rows['executions'])
        con.executemany('INSERT INTO health VALUES (?,?)',rows['health'])
    final = load_state(con)
    if final['last_ts'] != original['last_ts'] or con.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
        raise ValueError('candidate boundary/integrity mismatch')
    result = {'before_sha256':before, 'candidate_sha256':fingerprint(con),
              'started_ms':final['started'], 'last_ms':final['last_ts'],
              'decisions':con.execute('SELECT count(*) FROM decisions').fetchone()[0],
              'retrospective_decisions':state['history_import']['retrospective_decisions'],
              'preserved_forward_decisions':len(rows['decisions']), 'books':final['books']}
    con.close()
    return result


def install(source, candidate, expected, backup):
    """Atomic row update with stale-source rejection and pre-change backup."""
    with connect(candidate, readonly=True) as staged:
        if fingerprint(staged) != expected['candidate_sha256']:
            raise ValueError('candidate changed after verification')
        data = {t: staged.execute(f'SELECT * FROM {t}').fetchall() for t in REBUILT}
    con = connect(source)
    try:
        with sqlite3.connect(backup) as dest:
            con.backup(dest)
        con.execute('BEGIN IMMEDIATE')
        if fingerprint(con) != expected['before_sha256']:
            raise ValueError('live ledger changed; rebuild candidate before applying')
        for table in REBUILT:
            con.execute(f'DELETE FROM {table}')
            if data[table]:
                slots = ','.join('?' for _ in data[table][0])
                con.executemany(f'INSERT INTO {table} VALUES ({slots})',data[table])
        if fingerprint(con) != expected['candidate_sha256']:
            raise ValueError('installed ledger differs from verified candidate')
        con.commit()
        return internal_report(con)
    except BaseException:
        con.rollback()
        raise
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    with open(ROOT/'work/live/tick.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        candidate = args.output/'candidate.sqlite3'
        result = build_candidate(RUNTIME/'paper.sqlite3',candidate,args.replay)
        if args.apply:
            report = install(RUNTIME/'paper.sqlite3',candidate,result,args.output/'original.sqlite3')
            atomic_json(RUNTIME/'internal_report.json',report)
        result['applied'] = args.apply
        atomic_json(args.output/'migration.json',result)
        print(dumps(result))


if __name__ == '__main__':
    main()
