"""Tiny synthetic/reference controls. These are not MNIST or model results."""
import ast
from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import random
import sys
import tracemalloc
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_protocol import (Metrics, Pair, PairStream, Permutation, Protocol,
                            learning_rate, margin, overlap_stats, reconstruction_loss,
                            reconstruction_metrics, render,
                            route, squash)


class ProtocolTests(unittest.TestCase):
    def test_every_new_python_file_compiles_without_execution(self):
        paths = [*HERE.glob("paper_*.py"), HERE / "cpu_reference_gate.py", HERE / "cpu_trainer_gate.py"]
        for path in paths:
            compile(path.read_bytes(), str(path), "exec")

    def test_frozen_shape_and_real_cardinality(self):
        spec = Protocol().describe()
        self.assertEqual((spec["trainPairs"], spec["testPairs"], spec["updatesPerExposure"]),
                         (60000000, 10000000, 468750))
        # Independent layer-parameter arithmetic, not a framework/model import.
        parameters = 256 * (81 + 1) + 256 * (256 * 81 + 1)
        parameters += 3200 * 10 * 16 * 8 + 10 * 16
        parameters += (160 + 1) * 512 + (512 + 1) * 1024 + (1024 + 1) * 1296
        self.assertEqual(parameters, spec["parameters"])
        self.assertEqual(Protocol(profile="paper-equations").describe()["parameters"], parameters - 160)
        self.assertFalse(spec["executionApproved"])
        for keyword in ({"seed": True}, {"profile": "hybrid"}, {"pairs_per_anchor": 1001},
                        {"selection": "best-test"}, {"batch_size": 0}):
            with self.assertRaises(ValueError):
                Protocol(**keyword)
        self.assertNotEqual(Protocol().sha256, Protocol(seed=1).sha256)

    def test_permutations_are_bijections_and_reversible(self):
        for n in range(1, 129):
            for key in (0, "test", [9, "train"]):
                permutation = Permutation(n, key)
                ordered = [permutation.at(i) for i in range(n)]
                self.assertEqual(sorted(ordered), list(range(n)))
                self.assertEqual([permutation.at(i, inverse=True) for i in ordered], list(range(n)))
        for count in (0, True, 2**62 + 1):
            with self.assertRaises(ValueError):
                Permutation(count, 0)

    def test_streams_cover_all_ids_without_global_rng(self):
        state = random.getstate()
        protocol = Protocol(pairs_per_anchor=3, exposures=2)
        labels = bytes(i % 10 for i in range(30))
        stream = PairStream(labels, "train", protocol, fixture=True)
        first = [stream.ordered(i) for i in range(stream.count)]
        second = [stream.ordered(i, 1) for i in range(stream.count)]
        self.assertEqual({p.identity for p in first}, set(range(90)))
        self.assertNotEqual([p.identity for p in first], [p.identity for p in second])
        self.assertEqual(sorted(first, key=lambda p: p.identity), sorted(second, key=lambda p: p.identity))
        for anchor in range(30):
            rows = [p for p in first if p.anchor == anchor]
            self.assertEqual(len(rows), 3)
            self.assertEqual(len({p.partner for p in rows}), 3)
            self.assertEqual(len({p.anchor_shift for p in rows}), 1)
            self.assertTrue(all(p.labels[0] != p.labels[1] for p in rows))
        # Resume is an index operation, with no skipped or duplicated rows.
        self.assertEqual(first[37:], [stream.ordered(i) for i in range(37, stream.count)])
        other = PairStream(labels, "test", protocol, fixture=True)
        self.assertNotEqual(first, [other.ordered(i) for i in range(other.count)])
        for anchor in range(30):
            diagnostic = stream.validation(anchor)
            self.assertNotIn(diagnostic.partner, {p.partner for p in first if p.anchor == anchor})
            self.assertEqual(diagnostic.split, "train-diagnostic")
            self.assertGreaterEqual(diagnostic.identity, stream.count)
        with self.assertRaises(ValueError):
            other.validation(0)
        self.assertEqual(random.getstate(), state)
        with self.assertRaises(ValueError):
            PairStream(labels, "train", protocol)
        with self.assertRaises(ValueError):
            PairStream([0] * 30, "train", protocol, fixture=True)
        with self.assertRaises(ValueError):
            stream.ordered(stream.count)

    def test_thousand_unique_partners_and_large_index_memory(self):
        stream = PairStream(bytes(i % 10 for i in range(1200)), "train", Protocol(), fixture=True)
        pairs = [stream.pair(i) for i in range(1000)]
        self.assertEqual(len({p.partner for p in pairs}), 1000)
        tracemalloc.start()
        permutation = Permutation(60000000, [1, "large"])
        for i in range(0, 60000000, 60001):
            self.assertEqual(permutation.at(permutation.at(i), inverse=True), i)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertLess(peak, 1024 * 1024)

    def test_all_shifts_preserve_complete_glyph_and_clip_addition(self):
        glyph = bytes([173]) * 784
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                pair = Pair("train", 0, 0, 1, (2, 3), (dy, dx), (dy, dx))
                composite, targets = render(glyph, glyph, pair)
                self.assertEqual(sum(targets[0]), 173 * 784)
                self.assertEqual(targets[0].count(173), 784)
                self.assertEqual(composite.count(255), 784)
                self.assertNotEqual(composite, targets[0])  # catches pixelwise max
                self.assertEqual(len(composite), 1296)
        with self.assertRaises(ValueError):
            render(glyph, glyph, replace(pair, anchor_shift=(5, 0)))
        with self.assertRaises(ValueError):
            render(glyph[:-1], glyph, pair)

    def test_asymmetric_marker_axes_signs_and_overlap_diagnostics(self):
        glyph = bytearray(784)
        glyph[2 * 28 + 5] = 177
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                pair = Pair("train", 0, 0, 1, (2, 3), (dy, dx), (dy, dx))
                _, targets = render(bytes(glyph), bytes(glyph), pair)
                marker = (4 + dy + 2) * 36 + 4 + dx + 5
                self.assertEqual(targets[0][marker], 177)
                self.assertEqual(sum(targets[0]), 177)
                stats = overlap_stats(targets)
                self.assertEqual((stats["inkPixels"], stats["inkIoU"], stats["bboxIoU"]), ([1, 1], 1., 1.))
                self.assertIsNone(stats["admissionThreshold"])
        empty = overlap_stats([bytes(1296)] * 2)
        self.assertIsNone(empty["inkIoU"])
        self.assertIsNone(empty["bboxIoU"])

    def test_reconstruction_mask_provenance_and_assignment(self):
        a, b = [0.] * 1296, [0.] * 1296
        a[7], b[100] = 1., 1.
        composite = [min(x + y, 1) for x, y in zip(a, b)]
        conditioned = reconstruction_metrics([b, a], [a, b], composite, mask_kind="ground-truth")
        blind = reconstruction_metrics([b, a], [a, b], composite, mask_kind="predicted-top2")
        self.assertEqual(conditioned["labelConditionedSSE"], 4.)
        self.assertIsNone(conditioned["predictedMaskPermutationMSE"])
        self.assertEqual(blind["predictedMaskPermutationMSE"], 0.)
        self.assertIsNone(blind["labelConditionedMSE"])
        self.assertEqual(blind["clippedMixtureConservationMSE"], 0.)
        with self.assertRaises(ValueError):
            reconstruction_metrics([b, a], [a, b], composite, mask_kind="unspecified")

    def test_routing_closed_form_and_permutation_invariance(self):
        output, coefficients = route([[[3., 4.]]])
        for actual, expected in zip(output[0], (15 / 26, 20 / 26)):
            self.assertAlmostEqual(actual, expected, places=14)
        self.assertEqual(coefficients, [[1.]])
        votes = [[[.2, -.1], [.1, .4]], [[-.4, .3], [.2, -.2]], [[.6, .1], [-.1, .5]]]
        original, coefficients = route(votes)
        permuted, _ = route(list(reversed(votes)))
        for a, b in zip(original, permuted):
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y, places=14)
            self.assertLess(math.hypot(*a), 1.)
        for row in coefficients:
            self.assertAlmostEqual(sum(row), 1., places=14)
        self.assertEqual(route([[[0., 0.], [0., 0.]]])[0], [[0., 0.], [0., 0.]])
        self.assertEqual(squash([0., 0.]), [0., 0.])
        self.assertAlmostEqual(squash([1e300])[0], 1.)
        for bad in ([], [[[]]], [[[float("nan")]]]):
            with self.assertRaises(ValueError):
                route(bad)

    def test_loss_reductions_and_denominators(self):
        scores = [0.] * 10
        self.assertAlmostEqual(margin(scores, [2, 3], "paper-equations"), 1.62)
        self.assertAlmostEqual(margin(scores, [2, 3], "author2017"), .081)
        self.assertAlmostEqual(reconstruction_loss([[0.] * 1296] * 2, [[1.] * 1296] * 2), 1.296)
        self.assertEqual(learning_rate(0), .001)
        self.assertAlmostEqual(learning_rate(20000), .00096)
        self.assertEqual(learning_rate(10**9), 1e-6)
        metrics = Metrics()
        with self.assertRaises(ValueError):
            metrics.result()
        # predicted {0,1}: both, one, none; ties deliberately use lower class IDs
        for truth in ([0, 1], [0, 2], [2, 3]):
            metrics.add([.5] * 10, truth)
        result = metrics.result()
        self.assertEqual((result["pairs"], result["bothCorrect"], result["atLeastOneCorrect"], result["missedDigits"]), (3, 1, 2, 3))
        self.assertAlmostEqual(result["pairError"], 2 / 3)
        self.assertEqual(result["perDigitError"], .5)
        self.assertEqual(sum(result["truePerClass"]), 6)
        self.assertEqual(sum(result["predictedPerClass"]), 6)
        self.assertEqual(sum(result["truePositivePerClass"]), 3)
        self.assertEqual(result["top2BoundaryTies"], 3)
        with self.assertRaises(ValueError):
            metrics.add([.5] * 10, [1, 1])

    def test_model_module_imports_without_torch_or_native_packages(self):
        path = HERE / "paper_capsnet.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom)) and
                             (getattr(node, "module", "") or "").startswith("torch") for node in tree.body))
        before = set(sys.modules)
        spec = importlib.util.spec_from_file_location("paper_capsnet", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertFalse(any(name.split(".")[0] in ("torch", "numpy") for name in set(sys.modules) - before))


if __name__ == "__main__":
    unittest.main(verbosity=2)
