"""Recheck published aggregate arithmetic; no model, data or device is loaded."""
import json
import math
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent


class MeasuredReport(unittest.TestCase):
    def setUp(self):
        self.actual = json.loads((HERE / 'MEASURED_TEST_RESULT.json').read_bytes())
        self.report = json.loads((HERE / 'PAPER_RUN.json').read_bytes())

    def test_counts_and_denominators(self):
        m = self.actual['metrics']
        n, d = m['numerators'], m['denominators']
        pairs = self.actual['testPairsEvaluated']
        self.assertEqual(pairs, 10_000_000)
        self.assertEqual(self.actual['testBatchesEvaluated'], 78_125)
        self.assertEqual(d, dict(pairs=pairs, digits=2*pairs,
                                 pairedPixels=2592*pairs, canvasPixels=1296*pairs))
        both = n['bothCorrect']
        exactly_one = n['atLeastOneCorrect'] - both
        neither = pairs - n['atLeastOneCorrect']
        self.assertEqual((both, exactly_one, neither), (9_392_128, 599_314, 8_558))
        self.assertEqual(both + exactly_one + neither, pairs)
        self.assertEqual(n['missedDigits'], exactly_one + 2*neither)
        self.assertEqual(sum(n['truePerClass']), d['digits'])
        self.assertEqual(sum(n['predictedPerClass']), d['digits'])
        self.assertEqual(sum(n['truePositivePerClass']), 2*both + exactly_one)
        self.assertEqual(m['pairError'], (pairs-both)/pairs)
        self.assertEqual(m['perDigitError'], n['missedDigits']/d['digits'])
        self.assertEqual(m['bothAccuracy'], both/pairs)
        self.assertEqual(m['atLeastOneAccuracy'], n['atLeastOneCorrect']/pairs)

    def test_reconstruction_reductions(self):
        m = self.actual['metrics']
        for numerator, denominator, result in [
            ('labelConditionedSSE', 'pairedPixels', 'labelConditionedMSE'),
            ('predictedMaskPermutationSSE', 'pairedPixels', 'predictedMaskPermutationMSE'),
            ('groundTruthConservationSSE', 'canvasPixels', 'groundTruthConservationMSE'),
            ('predictedConservationSSE', 'canvasPixels', 'predictedConservationMSE'),
        ]:
            self.assertTrue(math.isclose(m['numerators'][numerator]/m['denominators'][denominator],
                                         m[result], rel_tol=1e-14, abs_tol=1e-16))

    def test_report_identity_and_comparison(self):
        r, a = self.report, self.actual
        self.assertEqual(r['finalEvaluation']['metrics'], a['metrics'])
        self.assertEqual(r['finalTrainingCheckpointSHA256'], a['modelCheckpointSHA256'])
        self.assertEqual(r['training']['optimizerUpdates'], a['trainingOptimizerUpdates'])
        self.assertEqual(a['trainingOptimizerUpdates'], 468_750)
        self.assertEqual(a['selection'], 'final-checkpoint-only')
        self.assertFalse(a['syntheticOnly'])
        self.assertTrue(r['finalEvaluation']['complete'])
        self.assertEqual(r['finalEvaluation']['continuation']['newOptimizerSteps'], 0)
        self.assertTrue(r['finalEvaluation']['continuation']['unknownPriorUncommittedInferenceTail'])
        self.assertEqual(r['paperComparison']['paperPairError'], 0.052)
        self.assertEqual(r['paperComparison']['section6_2ProsePairError'], 0.05)
        self.assertAlmostEqual(100*(a['metrics']['pairError']-0.052),
                               r['paperComparison']['differencePercentagePoints'], places=12)
        self.assertFalse(r['paperComparison']['exactMatchedConditions'])
        self.assertFalse(r['paperComparison']['paperReconstructionMSEComparatorAvailable'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
