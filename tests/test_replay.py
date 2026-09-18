import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hyfe import bars as B, features as F, live_cnn as LC
from hyfe.live import latest_features
from hyfe.paper_live import closed_features
from hyfe.paper_models import RUNTIME
from hyfe.paper_replay import simulate, run
from hyfe.paper_rules import STEP_MS
from hyfe.paper_store import connect, load_state, apply_decision
from hyfe.pilot_gbm import XS_FEATS


class ReplayTests(unittest.TestCase):
    def test_aggregated_features_match_live_and_ignore_future_bars(self):
        n = 320*240
        rng = np.random.default_rng(901)
        close = np.exp(4+np.cumsum(rng.normal(0,.0003,n)))
        raw = pd.DataFrame({'ts':np.arange(n)*60_000, 'open':close,
                            'high':close*1.001, 'low':close*.999, 'close':close,
                            'volume':rng.uniform(1,100,n), 'quote_volume':rng.uniform(100,10000,n)})
        decision = 300*STEP_MS
        for res,W,H in [('4h',60,84),('15m',120,10)]:
            bars = B.to_res(raw,B.RES_MIN[res])
            actual = closed_features(bars,res,W,H,decision)
            expected = latest_features(raw,res,W,H,decision)
            pd.testing.assert_series_equal(actual,expected)
            future = bars.ts+B.RES_MIN[res]*60_000 > decision
            bars.loc[future,['o','h','l','c','v','qv']] *= 1e8
            pd.testing.assert_series_equal(closed_features(bars,res,W,H,decision),expected)

    def test_prefiltering_renderer_context_preserves_every_pixel(self):
        n=400
        bars={}
        for i,base in enumerate(['A','B']):
            c=100+np.arange(n)*(.01+i*.02)
            bars[base]=pd.DataFrame({'ts':np.arange(n)*STEP_MS,'o':c,'h':c+1,
                                    'l':c-1,'c':c,'v':np.arange(n)+1+i})
        ctx=LC.bar_context(bars)
        cols=F.FEATURES+XS_FEATS
        meta={'feat_cols':cols,'feat_mu':[0.]*len(cols),'feat_sd':[1.]*len(cols)}
        row=pd.Series({c:.5 for c in cols})
        for base in bars:
            a=LC.render(ctx,base,n*STEP_MS,row,meta)
            b=LC.render(ctx[ctx.base==base],base,n*STEP_MS,row,meta)
            self.assertTrue(torch.equal(a,b))

    def test_replay_matches_forward_ledger_through_maturity_with_funding(self):
        bases=[f'B{i}' for i in range(20)]
        times=list(range(0,90*STEP_MS,STEP_MS))
        four={b:pd.DataFrame({'ts':np.array(times)-STEP_MS,
                              'c':100+np.arange(90)*(.01+i*.001)}) for i,b in enumerate(bases)}
        signals={t:pd.DataFrame({'base':bases,'ts':t,'score':np.linspace(-1,1,20),
                                  'entry':[four[b].c.iloc[k] for b in bases],
                                  'event':np.linspace(.1,.9,20)}) for k,t in enumerate(times)}
        funding={t:{b:(i-9)*.000001 for i,b in enumerate(bases) if i%3} for t in times[1:]}
        state,curve,closed=simulate(signals,four,funding)
        with tempfile.TemporaryDirectory() as root:
            con=connect(Path(root)/'test.sqlite3')
            for k,t in enumerate(times):
                prices={t:{b:float(four[b].c.iloc[k]) for b in bases}} if k else {}
                apply_decision(con,t,t+12*60_000,'test',signals[t],prices,{t:funding[t]} if k else {})
                current=load_state(con)
                for variant in ['equal','event']:
                    for kind in ['net','funded','funding_coverage']:
                        self.assertAlmostEqual(curve.iloc[k][variant+'_'+kind],current['books'][variant][kind],places=13)
            self.assertEqual(state['n_closed'],current['n_closed'])
            self.assertEqual(len(closed),6)
            self.assertEqual(len(state['cohorts']),84)
            con.close()

    def test_live_runtime_is_rejected_as_replay_output(self):
        with self.assertRaises(ValueError):
            run(RUNTIME)


if __name__=='__main__':
    unittest.main()
