import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from hyfe.paper_backfill import build_candidate, install, fingerprint, read_verified
from hyfe.paper_replay import simulate
from hyfe.paper_rules import STEP_MS
from hyfe.paper_store import connect, apply_decision, load_state, health


class BackfillTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.replay=self.root/'replay'
        (self.replay/'signals').mkdir(parents=True)
        (self.replay/'inputs/4h').mkdir(parents=True)
        self.db=self.root/'live.sqlite3'
        self.candidate=self.root/'candidate.sqlite3'
        self.bases=[f'B{i}' for i in range(20)]
        self.times=list(range(90*STEP_MS,180*STEP_MS,STEP_MS))
        self.four={b:pd.DataFrame({'ts':np.array(self.times)-STEP_MS,'c':100+np.arange(90)*(.01+i*.001)})
                   for i,b in enumerate(self.bases)}
        self.signals={t:pd.DataFrame({'base':self.bases,'ts':t,'score':np.linspace(-1,1,20),
                                     'entry':[self.four[b].c.iloc[k] for b in self.bases],
                                     'event':np.linspace(.1,.9,20)}) for k,t in enumerate(self.times)}
        self.prices={t:{b:float(self.four[b].c.iloc[k]) for b in self.bases} for k,t in enumerate(self.times)}
        self.funding={t:{b:(i-9)*.000001 for i,b in enumerate(self.bases)} for t in self.times}
        selected={t:self.signals[t] for t in self.times[:86]}
        _,curve,_=simulate(selected,self.four,self.funding)
        curve.to_parquet(self.replay/'equity_curve.parquet',index=False)
        for b,frame in self.four.items():
            frame.to_parquet(self.replay/'inputs/4h'/f'{b}.parquet',index=False)
        for t,frame in selected.items():
            frame.to_parquet(self.replay/'signals'/f'{t}.parquet',index=False)
        self.info={'start_ms':self.times[0],'end_ms':self.times[85],'cutover_ms':self.times[80],
                   'models':{'version':'test'}}
        (self.replay/'artifact_manifest.json').write_text('{}')
        (self.replay/'decision_provenance.json').write_text(json.dumps([
            {'decision_ms':t,'recomputed_ms':self.times[-1]+60_000,'model_version':'test'} for t in selected]))
        (self.replay/'funding_by_close.json').write_text(json.dumps(self.funding))
        con=connect(self.db)
        for t in self.times[80:88]:
            apply_decision(con,t,t+60_000,'test',self.signals[t],self.prices,self.funding)
        with con:
            con.execute('UPDATE executions SET price=100,observed_ms=? WHERE kind=?',(self.times[87],'entry'))
        health(con,'ok',self.times[87]+60_000)
        self.before=fingerprint(con)
        self.original_decisions=con.execute('SELECT * FROM decisions ORDER BY ts').fetchall()
        self.original_executions=con.execute('SELECT * FROM executions ORDER BY 1,2,3,4').fetchall()
        con.close()

    def tearDown(self):
        self.tmp.cleanup()

    def build(self):
        with patch('hyfe.paper_backfill.read_verified',return_value=self.info),patch('hyfe.paper_backfill.collect',return_value=self.funding):
            return build_candidate(self.db,self.candidate,self.replay)

    def test_import_preserves_live_records_and_continues_through_maturity(self):
        result=self.build()
        with connect(self.db,readonly=True) as con:
            self.assertEqual(fingerprint(con),self.before)
        backup=self.root/'backup.sqlite3'
        install(self.db,self.candidate,result,backup)
        with connect(backup,readonly=True) as con:
            self.assertEqual(fingerprint(con),self.before)
        with connect(self.db) as con:
            self.assertEqual(con.execute('SELECT * FROM decisions WHERE ts>=? ORDER BY ts',(self.times[80],)).fetchall(),self.original_decisions)
            self.assertEqual(con.execute('SELECT * FROM executions ORDER BY 1,2,3,4').fetchall(),self.original_executions)
            self.assertGreater(con.execute('SELECT generated_ms-ts FROM decisions ORDER BY ts LIMIT 1').fetchone()[0],45*60_000)
            state=load_state(con)
            self.assertEqual(state['n_closed'],4)
            self.assertEqual(json.loads(con.execute('SELECT payload FROM closed ORDER BY id LIMIT 1').fetchone()[0])['origin'],'retrospective')
            self.assertEqual(state['history_import']['retrospective_decisions'],80)
            for t in self.times[88:]:
                apply_decision(con,t,t+60_000,'test',self.signals[t],self.prices,self.funding)
            expected,_,_=simulate(self.signals,self.four,self.funding)
            actual=load_state(con)
            self.assertEqual(actual['n_closed'],expected['n_closed'])
            for v in ['event','equal']:
                for k in ['net','funded','funding_coverage']:
                    self.assertAlmostEqual(actual['books'][v][k],expected['books'][v][k],places=13)

    def test_changed_source_rejects_install_without_partial_writes(self):
        result=self.build()
        with connect(self.db) as con:
            health(con,'ok',self.times[-1])
            changed=fingerprint(con)
        with self.assertRaisesRegex(ValueError,'live ledger changed'):
            install(self.db,self.candidate,result,self.root/'backup.sqlite3')
        with connect(self.db,readonly=True) as con:
            self.assertEqual(fingerprint(con),changed)

    def test_repeat_import_is_rejected(self):
        result=self.build()
        install(self.db,self.candidate,result,self.root/'backup.sqlite3')
        with patch('hyfe.paper_backfill.read_verified',return_value=self.info):
            with self.assertRaisesRegex(ValueError,'already has'):
                build_candidate(self.db,self.root/'again.sqlite3',self.replay)

    def test_modified_replay_is_rejected(self):
        (self.replay/'artifact_manifest.json').write_text(json.dumps({'files':{'decision_provenance.json':'wrong'}}))
        (self.replay/'input_manifest.json').write_text(json.dumps({'files':{}}))
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):
            read_verified(self.replay)


if __name__=='__main__':
    unittest.main()
