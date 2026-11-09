import ast
import copy
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_protocol import Protocol, PairStream, Metrics, canonical, digest
from paper_state import Checkpoints
from paper_train import Trainer, Aggregate, StopRequested, evaluate_final, final_checkpoint

BINDING = {k+"SHA256": digest(["synthetic-fixture", k]) for k in ("protocol", "source", "data", "runtime")}


class FakeEngine:
    """Explicit control-flow fixture; these are NOT model/GPU measurements."""
    def __init__(self):
        self.completed_updates, self.rng, self.weight = 0, 17, 0.
        self.ids, self.purposes = [], []

    def step(self, batch):
        self.ids.extend(p.identity for p in batch["pairs"])
        self.rng = (1664525*self.rng+1013904223) % 2**32
        self.weight += self.rng/2**32
        self.completed_updates += 1
        n = len(batch["pairs"])
        return {"pairs": n, "classificationSum": n*self.weight, "reconstructionSum": n*.1}

    def evaluate(self, rows, purpose, stop=lambda: False):
        aggregate = Aggregate()
        for batch in rows:
            if stop():
                raise StopRequested()
            self.purposes.append((purpose, [p.identity for p in batch["pairs"]]))
            metrics = Metrics()
            for pair in batch["pairs"]:
                metrics.add([.1]*10, pair.labels)
            report = metrics.result()
            aggregate.add({k: report[k] for k in aggregate.state if k not in
                           ("labelConditionedSSE", "predictedMaskPermutationSSE", "groundTruthConservationSSE", "predictedConservationSSE")} |
                          {"labelConditionedSSE": len(batch["pairs"])*(self.completed_updates-2)**2/100,
                           "predictedMaskPermutationSSE": 0., "groundTruthConservationSSE": 0.,
                           "predictedConservationSSE": 0.})
        return aggregate.state

    def snapshot(self):
        return {"updates": self.completed_updates, "rng": self.rng, "weight": self.weight}

    def restore(self, state):
        self.completed_updates, self.rng, self.weight = state["updates"], state["rng"], state["weight"]

    def encode(self, payload):
        return canonical(payload)

    def decode(self, raw):
        return json.loads(raw)


def fixture(root, engine=None):
    p = Protocol(pairs_per_anchor=2, batch_size=3, validation_anchors=2, validation_every_updates=2)
    labels = bytes([0, 1, 2, 3, 0, 1, 2, 3])
    images = bytes((i % 256 for i in range(8*784)))
    engine = engine or FakeEngine()
    store = Checkpoints(Path(root)/"train", BINDING)
    run = Trainer(p, PairStream(labels, "train", p, fixture=True), images, engine, store,
                  fixture=True, checkpoint_every=2)
    return run, engine, store, p


class TrainerTests(unittest.TestCase):
    def test_isolated_cli_help_needs_no_native_library_or_dataset(self):
        child = subprocess.run([sys.executable, "-I", "-B", "-S", str(Path(__file__).with_name("paper_train.py")), "--help"],
                               capture_output=True, timeout=10)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertIn(b"--phase", child.stdout)

    def test_explicit_primary_budget_and_no_native_import(self):
        p = Protocol().describe()
        self.assertEqual((p["trainPairs"], p["testPairs"], p["updatesPerExposure"]), (60000000, 10000000, 468750))
        import paper_engine
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)
        for name in ("paper_train.py", "paper_engine.py"):
            tree = ast.parse(Path(__file__).with_name(name).read_text())
            self.assertFalse(any(isinstance(n, (ast.Import, ast.ImportFrom)) and
                                 (getattr(n, "module", "") or "").startswith("torch") for n in tree.body))

    def test_exact_finite_tail_and_train_only_diagnostics(self):
        with tempfile.TemporaryDirectory() as root:
            run, engine, store, p = fixture(root)
            result = run.run()
            self.assertEqual((result["completedPairs"], result["completedUpdates"]), (16, 6))
            self.assertEqual(run.state["completedExposures"], 1)
            self.assertEqual(len(engine.ids), 16)
            self.assertEqual(len(set(engine.ids)), 16)
            self.assertTrue(all(kind == "train-diagnostic" for kind, _ in engine.purposes))
            self.assertEqual([r["update"] for r in run.state["diagnostics"]], [2, 4, 6])
            self.assertEqual(result["testPairsEvaluated"], 0)
            self.assertEqual((result["diagnosticEvaluations"], result["uniqueDiagnosticPairs"],
                              result["diagnosticPairObservations"]), (3, 2, 6))
            self.assertFalse(result["onlineTraining"]["isFinalModelEvaluation"])
            self.assertEqual(run.state["online"]["pairs"], 16)
            self.assertEqual(store.read("best")[1]["completedUpdates"], 2)
            self.assertEqual(len(list(store.root.glob("payload-*.bin"))), 2)

    def test_graceful_stop_and_resume_match_uninterrupted_state(self):
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            full, a, _, _ = fixture(one)
            full.run()
            partial, b, _, _ = fixture(two)
            stopped = partial.run(lambda: b.completed_updates >= 3)
            self.assertEqual(stopped["committedPairs"], 9)
            resumed, c, _, _ = fixture(two)
            resumed.run()
            self.assertEqual(a.snapshot(), c.snapshot())
            self.assertEqual(full.state, resumed.state)
            self.assertEqual(b.ids+c.ids, a.ids)

    def test_interrupted_diagnostic_retains_prior_checkpoint_no_fake_score(self):
        with tempfile.TemporaryDirectory() as root:
            run, engine, store, _ = fixture(root)
            stopped = run.run(lambda: engine.completed_updates >= 2)
            self.assertEqual((stopped["committedPairs"], stopped["uncommittedPairs"]), (0, 6))
            self.assertIsNone(store.read("best"))
            restored, _, _, _ = fixture(root)
            restored.run()
            self.assertEqual(len(restored.state["diagnostics"]), 3)

    def test_final_test_is_latest_complete_model_not_diagnostic_best(self):
        with tempfile.TemporaryDirectory() as root:
            run, engine, store, p = fixture(root)
            run.run()
            entry = final_checkpoint(store, engine, p, fixture=True, train_count=16)
            self.assertEqual(entry["completedUpdates"], 6)
            self.assertNotEqual(entry["sha256"], store.read("best")[1]["sha256"])
            self.assertEqual(engine.completed_updates, 6)

    def test_test_progress_resume_preserves_denominators_and_indices(self):
        with tempfile.TemporaryDirectory() as root:
            run, _, store, p = fixture(root)
            run.run()
            stream = PairStream(bytes([0, 1, 2, 3]), "test", p, fixture=True)
            images = bytes([3])*(4*784)
            evaluation = Checkpoints(Path(root)/"test", BINDING)
            a = FakeEngine()
            stopped = evaluate_final(p, stream, images, a, store, evaluation, fixture=True,
                                     train_count=16, stop=lambda: len(a.purposes) >= 1)
            self.assertEqual(stopped["testPairsEvaluated"], 3)
            b = FakeEngine()
            result = evaluate_final(p, stream, images, b, store, evaluation, fixture=True, train_count=16)
            ids = [i for _, values in a.purposes+b.purposes for i in values]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(ids), 8)
            self.assertEqual(result["metrics"]["denominators"],
                             {"pairs": 8, "digits": 16, "pairedPixels": 20736, "canvasPixels": 10368})
            self.assertEqual(b.completed_updates, 6)
            before = len(b.purposes)
            self.assertEqual(evaluate_final(p, stream, images, b, store, evaluation, fixture=True,
                                           train_count=16), result)
            self.assertEqual(before, len(b.purposes))

    def test_corrupt_cursor_protocol_and_premature_test_refused(self):
        with tempfile.TemporaryDirectory() as root:
            run, engine, store, p = fixture(root)
            run.run(lambda: engine.completed_updates >= 1)
            with self.assertRaises(ValueError):
                final_checkpoint(store, engine, p, fixture=True, train_count=16)
            state = copy.deepcopy(run.state)
            state["completedPairs"] += 1
            with self.assertRaises(ValueError):
                run.validate(state)
            state = copy.deepcopy(run.state)
            state["diagnostics"] = [{"update": 2, "metrics": {}}]
            with self.assertRaises(ValueError):
                run.validate(state)
            with self.assertRaises(ValueError):
                Trainer(p, run.stream, run.images, engine, store)  # fixture protocol cannot become production

    def test_post_commit_callback_failure_keeps_authentic_resumable_prefix(self):
        class FailAfterCommit(Checkpoints):
            def commit(self, *values, **options):
                super().commit(*values, **options)
                raise RuntimeError("synthetic logging callback failed after publication")

        with tempfile.TemporaryDirectory() as root:
            run, _, training, p = fixture(root)
            run.run()
            stream = PairStream(bytes([0, 1, 2, 3]), "test", p, fixture=True)
            images = bytes([3])*(4*784)
            interrupted = FailAfterCommit(Path(root)/"interrupted", BINDING)
            first = FakeEngine()
            with self.assertRaisesRegex(RuntimeError, "after publication"):
                evaluate_final(p, stream, images, first, training, interrupted,
                               fixture=True, train_count=16, checkpoint_batches=1)
            payload, entry = interrupted.read()
            self.assertEqual((entry["completedPairs"], entry["completedUpdates"]), (3, 1))
            self.assertEqual(json.loads(payload)["metrics"]["pairs"], 3)
            self.assertTrue((interrupted.root/"SESSION.lock").exists())
            self.assertFalse((interrupted.root/"WRITER.lock").exists())
            # Unexpected failures never steal or silently clear a live/session lock.
            with self.assertRaises(FileExistsError):
                evaluate_final(p, stream, images, FakeEngine(), training, interrupted,
                               fixture=True, train_count=16)
            # Explicit reconciliation is simulated ONLY in this disposable fixture:
            # copy the two authenticated committed files; retain the original lock.
            reconciled = Checkpoints(Path(root)/"reconciled", BINDING)
            for name in ("INDEX.json", entry["name"]):
                (reconciled.root/name).write_bytes((interrupted.root/name).read_bytes())
            second = FakeEngine()
            result = evaluate_final(p, stream, images, second, training, reconciled,
                                    fixture=True, train_count=16, checkpoint_batches=1)
            ids = [i for _, values in first.purposes+second.purposes for i in values]
            self.assertEqual((len(ids), len(set(ids))), (8, 8))
            self.assertTrue((interrupted.root/"SESSION.lock").exists())
            control = evaluate_final(p, stream, images, FakeEngine(), training,
                                     Checkpoints(Path(root)/"control", BINDING),
                                     fixture=True, train_count=16, checkpoint_batches=1)
            self.assertEqual(result, control)
            self.assertEqual(second.completed_updates, 6)

    def test_failed_update_preserves_checkpoint_and_session_lock(self):
        class WrongEngine(FakeEngine):
            def step(self, batch):
                result = super().step(batch)
                self.completed_updates += 1
                return result
        with tempfile.TemporaryDirectory() as root:
            run, _, store, _ = fixture(root, WrongEngine())
            with self.assertRaises(ValueError):
                run.run()
            self.assertEqual(store.read()[1]["completedPairs"], 0)
            self.assertTrue((store.root/"SESSION.lock").exists())
            with self.assertRaises(FileExistsError):
                fixture(root)[0].run()

    def test_metric_numerators_reject_nonfinite_partial_and_impossible(self):
        engine = FakeEngine()
        run_labels = [type("P", (), {"labels": (0, 1), "identity": 0})()]
        row = engine.evaluate(iter([{"pairs": run_labels}]), "train-diagnostic")
        report = Aggregate(row).report()
        self.assertEqual(report["pairError"], 0)
        self.assertEqual(report["perDigitError"], 0)
        for key, value in (("labelConditionedSSE", float("nan")), ("missedDigits", 1), ("pairs", 0)):
            bad = copy.deepcopy(row)
            bad[key] = value
            with self.assertRaises(ValueError):
                Aggregate().add(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
