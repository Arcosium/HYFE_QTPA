import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from hyfe import cohort as batch
from hyfe.paper_rules import STEP_MS, HOLD, COST, ensemble, select_legs, signal_positions, advance_cohort
from hyfe.paper_store import connect, load_state, apply_decision, resolve_executions, internal_report
from hyfe.paper_models import training_plan, retrain, prepare_snapshot
from hyfe.paper_funding import history, collect


def frame(ts=0, n=20):
    return pd.DataFrame({"base": [f"B{i}" for i in range(n)], "ts": ts,
                         "score": np.linspace(-1, 1, n), "entry": 100.,
                         "event": np.linspace(0.1, 0.9, n)})


class RulesTests(unittest.TestCase):
    def test_normalization_cannot_see_future_cross_sections(self):
        d = frame()
        d["s0"] = d.score
        d["s1"] = d.score**3
        expected = ensemble(d, ["s0", "s1"])
        future = d.assign(ts=STEP_MS, s0=d.s0 * 1e8, s1=d.s1 * -1e5)
        actual = ensemble(pd.concat([d, future], ignore_index=True), ["s0", "s1"])
        np.testing.assert_allclose(actual.iloc[:len(d)], expected, atol=1e-12)

    def test_nan_seed_and_duplicate_symbol_are_rejected(self):
        d = frame().assign(s0=1., s1=2.)
        d.loc[0, "s0"] = np.nan
        with self.assertRaises(ValueError): ensemble(d, ["s0", "s1"])
        d.loc[0, "s0"] = 1.
        with self.assertRaises(ValueError): ensemble(pd.concat([d, d.iloc[:1]]), ["s0", "s1"])

    def test_rank_boundaries_and_balanced_legs(self):
        d = frame(n=200)
        l, s = select_legs(d.score)
        self.assertEqual((int(l.sum()), int(s.sum())), (21, 20))
        pp = signal_positions(d)
        for v in ["equal", "event"]:
            self.assertAlmostEqual(sum(p[v] for p in pp if p["side"] == "long"), .5)
            self.assertAlmostEqual(sum(p[v] for p in pp if p["side"] == "short"), .5)
        with self.assertRaises(ValueError): signal_positions(d.assign(score=0.))

    def test_flat_prices_charge_exact_roundtrip_cost(self):
        c = {"positions": signal_positions(frame()), "age": 0}
        px = {p["base"]: 100. for p in c["positions"]}
        total = 0.
        for _ in range(HOLD):
            total += advance_cohort(c, px)["equal"]["net"]
        self.assertAlmostEqual(total, -0.002, places=14)
        self.assertEqual(c["age"], HOLD)

    def test_long_short_funding_sign_and_missing_coverage(self):
        c = {"positions": [
            {"base": "L", "side": "long", "entry": 100., "equal": .5, "event": .5},
            {"base": "S", "side": "short", "entry": 100., "equal": .5, "event": .5}]}
        r = advance_cohort(c, {"L": 110., "S": 90.}, {"L": .01, "S": .02})["event"]
        self.assertAlmostEqual(r["gross"], .1)
        self.assertAlmostEqual(r["funding"], -.005)
        self.assertAlmostEqual(r["funded"], .105 - .002/HOLD)
        self.assertEqual(r["funding_coverage"], 1.)
        r = advance_cohort(c, {"L": 110., "S": 90.}, {"L": .01})["event"]
        self.assertEqual(r["funding_coverage"], .5)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.con = connect(self.root / "paper.sqlite3")

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def open_first(self):
        self.assertTrue(apply_decision(self.con, 0, 12*60_000, "v1", frame(), {}, {}))

    def test_restart_idempotence_and_same_symbol_multiple_cohorts(self):
        self.open_first()
        before = load_state(self.con)
        self.assertFalse(apply_decision(self.con, 0, 13*60_000, "v1", frame(), {}, {}))
        self.assertEqual(load_state(self.con), before)
        self.con.close(); self.con = connect(self.root / "paper.sqlite3")
        prices = {STEP_MS: {f"B{i}": 100. for i in range(20)}}
        apply_decision(self.con, STEP_MS, STEP_MS+12*60_000, "v1", frame(STEP_MS), prices, {STEP_MS: {}})
        state = load_state(self.con)
        self.assertEqual(len(state["cohorts"]), 2)
        self.assertEqual(state["cohorts"][0]["positions"][0]["base"], state["cohorts"][1]["positions"][0]["base"])

    def test_error_rolls_back_marks_both_books_and_signal(self):
        self.open_first(); before = load_state(self.con)
        with self.assertRaises(ValueError):
            apply_decision(self.con, STEP_MS, STEP_MS+1, "v1", frame(STEP_MS), {STEP_MS: {}}, {STEP_MS: {}})
        self.assertEqual(load_state(self.con), before)
        self.assertEqual(self.con.execute("SELECT count(*) FROM marks").fetchone()[0], 0)
        self.assertEqual(self.con.execute("SELECT count(*) FROM decisions").fetchone()[0], 1)

    def test_missed_ticks_mark_holdings_without_backdating_signals(self):
        self.open_first()
        prices = {i*STEP_MS: {f"B{j}": 100. for j in range(20)} for i in range(1, 4)}
        apply_decision(self.con, 3*STEP_MS, 3*STEP_MS+12*60_000, "v1", frame(3*STEP_MS), prices, {t: {} for t in prices})
        state = load_state(self.con)
        self.assertEqual([c["id"] for c in state["cohorts"]], [0, 3*STEP_MS])
        self.assertEqual(self.con.execute("SELECT count(*) FROM marks").fetchone()[0], 3)
        with self.assertRaises(ValueError):
            apply_decision(self.con, 4*STEP_MS, 4*STEP_MS+46*60_000, "v1", frame(4*STEP_MS), {}, {})

    def test_execution_reference_strictly_after_signal(self):
        self.open_first()
        base, target = self.con.execute("SELECT base,target_ms FROM executions WHERE kind='entry' LIMIT 1").fetchone()
        self.assertGreater(target, 12*60_000)
        older = {base: pd.DataFrame({"ts": [target-60_000], "open": [99.]})}
        self.assertEqual(resolve_executions(self.con, older, target+120_000), 0)
        exact = {base: pd.DataFrame({"ts": [target], "open": [101.]})}
        self.assertEqual(resolve_executions(self.con, exact, target+120_000), 1)
        self.assertEqual(resolve_executions(self.con, exact, target+180_000), 0)

    def test_forward_ledger_matches_batch_backtest_and_84_bar_maturity(self):
        rng = np.random.default_rng(41)
        n, T, opens = 20, 94, 10
        prices = 100*np.exp(np.cumsum(rng.normal(0, .01, (T, n)), axis=0))
        grid = np.arange(T)*STEP_MS
        bases = [f"B{i}" for i in range(n)]
        signals, weights = [], []
        for i, ts in enumerate(grid):
            d = None
            if i < opens:
                d = frame(int(ts)).assign(entry=prices[i], score=rng.normal(size=n), event=rng.uniform(.05, .95, n))
                signals.append(d)
                weights.append(d.assign(w=d.event.rank(pct=True)))
            px = {int(ts): dict(zip(bases, prices[i]))} if i else {}
            apply_decision(self.con, int(ts), int(ts)+12*60_000, "v1", d, px, {int(ts): {}})
        state = load_state(self.con)
        self.assertEqual(state["n_closed"], opens)
        self.assertEqual(state["cohorts"], [])
        sig = pd.concat(signals, ignore_index=True)
        pred = self.root / "test_pred.npz"
        s = sig.score.to_numpy()
        np.savez(pred, ts=sig.ts, base=sig.base, p=np.c_[np.zeros(len(sig)), s.clip(0), (-s).clip(0)])
        w = pd.concat(weights)
        wf = self.root / "weights.npz"
        np.savez(wf, ts=w.ts, base=w.base, w=w.w)
        for j, base in enumerate(bases):
            pd.DataFrame({"ts": grid, "c": prices[:, j]}).to_parquet(self.root / (base+".parquet"))
        with patch.object(batch.B, "path", lambda res, base: str(self.root/(base+".parquet"))):
            _, eq_equal = batch.run(pred, "4h", HOLD)
            _, eq_event = batch.run(pred, "4h", HOLD, weight=str(wf))
        self.assertAlmostEqual(state["books"]["equal"]["net"], float(eq_equal.iloc[-1]), places=12)
        self.assertAlmostEqual(state["books"]["event"]["net"], float(eq_event.iloc[-1]), places=12)
        report = internal_report(self.con)
        self.assertEqual(report["decisions"], opens)
        self.assertIsNone(report["variants"]["event"]["sharpe"])


class FundingTests(unittest.TestCase):
    def test_full_page_is_split_without_dropping_settlements(self):
        events = [{"symbol": "AUSDT", "fundingTime": i, "fundingRate": ".0001"} for i in range(1200)]
        def request(endpoint, params):
            return [x for x in events if params["startTime"] <= x["fundingTime"] <= params["endTime"]][:1000]
        result = history(0, 1199, ["AUSDT"], request)
        self.assertEqual(len(result), 1200)
        self.assertEqual(len({x["fundingTime"] for x in result}), 1200)

    def test_known_zero_unknown_coverage_and_boundary_assignment(self):
        def request(endpoint, params=None):
            if endpoint == "exchangeInfo":
                return {"symbols": [{"baseAsset": "A", "symbol": "AUSDT", "quoteAsset": "USDT", "contractType": "PERPETUAL"}]}
            return [{"symbol": "AUSDT", "fundingTime": STEP_MS, "fundingRate": ".001"}]
        with tempfile.TemporaryDirectory() as root:
            rates = collect(root, [STEP_MS, 2*STEP_MS], ["A", "UNKNOWN"], request)
        self.assertEqual(rates[STEP_MS], {"A": .001})
        self.assertEqual(rates[2*STEP_MS], {"A": 0.})


class TrainingTests(unittest.TestCase):
    def test_fixed_splits_mature_labels_and_quarter_rollover(self):
        p = training_plan("2026-09-17T12:00:00Z")
        self.assertEqual(p["data_cutoff"][:10], "2026-09-01")
        self.assertEqual(p["validation_start"][:10], "2026-07-01")
        self.assertEqual(p["training_window_end_exclusive"][:10], "2026-06-07")
        self.assertEqual(p["next_quarter"][:10], "2026-10-01")
        self.assertEqual(training_plan("2026-12-31")["next_quarter"][:10], "2027-01-01")

    def test_busy_resources_cannot_launch_training_or_change_active_models(self):
        with tempfile.TemporaryDirectory() as root, patch("hyfe.paper_models.bootstrap", return_value={"version": "baseline-test"}), \
             patch("hyfe.paper_models.resource_status", return_value={"ready": False}), \
             patch("hyfe.paper_models.subprocess.run") as run:
            result = retrain(Path(root), run=True)
            self.assertEqual(result["status"], "waiting_resources")
            run.assert_not_called()
            self.assertFalse((Path(root)/"active_models.json").exists())

    def test_running_training_keeps_its_status_and_lock(self):
        import fcntl
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root/"training_status.json").write_text('{"status":"training","seed":4}')
            with (root/"training.lock").open("a") as lock, \
                 patch("hyfe.paper_models.bootstrap", return_value={"version": "baseline-test"}), \
                 patch("hyfe.paper_models.resource_status", return_value={"ready": False}):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(retrain(root, run=True)["status"], "already_running")
                self.assertEqual(json.loads((root/"training_status.json").read_text())["seed"], 4)

    def test_snapshot_resume_rejects_changed_input_and_ignores_new_history(self):
        d = pd.DataFrame({"ts": [0], "o": [1.], "h": [1.], "l": [1.], "c": [1.], "v": [1.], "qv": [1.]})
        with tempfile.TemporaryDirectory() as root, patch("hyfe.paper_data.refresh_bars", return_value=d) as refresh:
            self.assertEqual(prepare_snapshot(root, ["A"], {}, STEP_MS), ["A"])
            refresh.reset_mock()
            self.assertEqual(prepare_snapshot(root, ["B"], {}, STEP_MS), ["A"])
            refresh.assert_not_called()
            (Path(root)/"bars/4h/A.parquet").write_bytes(b"modified")
            with self.assertRaises(ValueError): prepare_snapshot(root, ["A"], {}, STEP_MS)


class LiveTests(unittest.TestCase):
    def test_dry_run_leaves_no_database_or_runtime_artifacts(self):
        from hyfe import paper_live
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            manifest = {"models": list(range(10)), "version": "test"}
            (root/"active_models.json").write_text(json.dumps(manifest))
            before = {p.name:p.read_bytes() for p in root.iterdir()}
            with patch.object(paper_live, "validate_manifest", return_value=manifest), \
                 patch.object(paper_live, "bootstrap") as boot, \
                 patch.object(paper_live, "load_buffer", return_value={}), \
                 patch.object(paper_live, "liquid_bases", return_value=[]), \
                 patch.object(paper_live, "infer", return_value=frame()), \
                 patch.object(paper_live, "collect") as fund:
                result = paper_live.tick(root, dry=True, now_ms=12*60_000)
            self.assertEqual(result["status"], "dry")
            boot.assert_not_called(); fund.assert_not_called()
            self.assertEqual({p.name:p.read_bytes() for p in root.iterdir()}, before)


if __name__ == "__main__":
    unittest.main()
