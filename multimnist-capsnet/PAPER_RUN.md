# Full-width run: complete training and final evaluation

This single-seed `author2017` condition completed one 60M-pair training exposure
and its final-checkpoint-only 10M-pair test. Pair error is **6.07872%**,
compared with **5.2%** for the three-routing/reconstruction CapsNet row in
[Table 1 of the paper](https://arxiv.org/pdf/1710.09829v2). That is **0.87872
percentage points higher**. Section 6.2 separately states 5.0%; it is not the
selected table comparator. The outcome remains **partial**, rather than an
exact reproduction under matched conditions or a general failure of CapsNet.

The historical 48.6% reduced NumPy result is unchanged. This report advances
[issue #46](https://github.com/cybertronai/hinton-problems/issues/46), but does not
close its entire 25-stub/A100 request. It supplies quantitative reconstruction
measurements, not a claim of matching a paper reconstruction MSE: no numerical
paper MSE comparator was established. Prepared with Codex.

## Classification: counts before percentages

| Measurement | Numerator / denominator | Result |
|---|---|---|
| Both labels correct | 9,392,128 / 10,000,000 pairs | 93.92128% |
| Exactly one correct | 599,314 / 10,000,000 pairs | 5.99314% |
| Neither correct | 8,558 / 10,000,000 pairs | 0.08558% |
| Pair error | 607,872 / 10,000,000 pairs | 6.07872% |
| Per-digit error | 616,430 / 20,000,000 digits | 3.08215% |
| At least one correct | 9,991,442 / 10,000,000 pairs | 99.91442% |
| Top-two boundary ties | 0 / 10,000,000 pairs | 0 |

These denominators are different. Per-digit error must not be compared with
the paper's pair error. The complete per-class counts are in
[MEASURED_TEST_RESULT.json](MEASURED_TEST_RESULT.json), copied byte-for-byte
from the genuine native aggregate, and [PAPER_RUN.json](PAPER_RUN.json).

## Reconstruction: four separately defined statistics

| Measurement | SSE | Denominator | MSE |
|---|---|---|---|
| Label-conditioned paired reconstruction | 253665617.7847454 | 25,920,000,000 branch-pixels | 0.00978648216762135 |
| Predicted-mask/permutation-matched reconstruction | 260895937.93180078 | 25,920,000,000 branch-pixels | 0.01006542970415898 |
| Ground-truth-conditioned mixture conservation | 152747352.75525546 | 12,960,000,000 canvas-pixels | 0.011786061169386996 |
| Predicted-mask mixture conservation | 153228204.0437238 | 12,960,000,000 canvas-pixels | 0.011823163892262638 |

Correct labels supplied to the decoder are not blind segmentation. These
aggregate errors do not replace a qualitative visual reconstruction assessment.
No source images, reconstructions, dataset payloads or trained weights are
redistributed in this code contribution.

## Fixed condition and measured resources

[PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) specifies the released-author profile:
256 Conv1 channels, 32 primary capsule types, 11,361,968 parameters, three routing
iterations and a shared 512/1024 decoder; clipped-addition 36x36 MultiMNIST;
seed 20261107, batch 128, FP32/no AMP/TF32 disabled, explicit Adam and decay.
Training committed 468,750 optimizer updates / 60,000,000 pairs. It resumed a
verified 256-update/32,768-pair prefix and made 468,494 new updates, 469 new
checkpoint commits and 46 diagnostic rounds over the same 1,000 extra ordered
training pairs. Those 46,000 observations are neither new distinct examples nor
optimizer exposures or held-out selection.

| Timing or memory | Actual observation |
|---|---|
| Prefix training wall time | 25.747s |
| Continuation training wall time | 18,930.454s including checkpoints, diagnostics and rendering |
| Continuation native / owned-job wall time | 18,941.415s /19,120.188s |
| Continuation peak Torch allocated / reserved | 1,585,392,640 /2,684,354,560 bytes |
| Continuation peak whole owned-job memory | 6,310,838,272 bytes |
| Last test continuation | 2,115,200 pairs / 16,525 batches; 351.696s evaluation / 363.392s native |
| Last test continuation peak Torch allocated / reserved | 893,412,864 /1,166,016,512 bytes |

The last continuation's timing is not the total test wall time. Runtime was
Python 3.13.14, PyTorch 2.14.0+cu130, Windows x86_64, RTX 5090/SM120, driver 621.14,
under an unchanged external 410W limit. Wall time is not integrated GPU-busy
time. CPU numerical/reference gates, full-width CPU/GPU parity and a 13-step
serialized-continuation pilot passed before the recorded training.

## One frozen experiment, with preserved continuation history

The original final-evaluation wrapper failed after committing 12,800 pairs /
100 batches, when an external checkpoint-log callback raised. Its single-use
admission, exception and checkpoint remain preserved. A separately reviewed
continuation reached 7,884,800 pairs / 61,600 batches before a gaming stop. The
last independently admitted child resumed that exact committed aggregate and
processed only the remaining 2,115,200 pairs / 16,525 batches. The final aggregate
contains 10,000,000 distinct ordered pairs and 78,125 batches. It is not a new
training trial, diagnostic-best selection, reset, fresh test seed or cumulative
optimizer update counter. No optimizer steps occurred during evaluation.

Previously uncommitted inference at interruption may have been performed and
recomputed. Its exact size is unknown; the committed cursor prevents adding
those pair numerators twice. Thus 10M committed pairs does not prove exactly 10M
physical forward attempts over all interrupted processes. No unseen-tail or
uninterrupted-execution claim is made.

The final checkpoint digest is
`5b5f200c03482fe949d47f493f09eca6b1ef1e4fb343f076a59a9679e686eea4`.
Native before/after model/optimizer-state digests match; final model identity,
full aggregate, zero optimizer changes and outer runtime identity passed.
Exact owned-process/container zero cleanup and original-token release passed
and were independently read back. Receipt digests and source/data/runtime
bindings are in PAPER_RUN.json; public aggregate arithmetic is independently
checked by `test_paper_report.py`. Private execution logs are not published.

## Limits on comparison and rerunning

This is one seed and one explicit 60M-pair exposure; the exact historical total
training-update budget was not recovered. Deterministic pairing changes partner
identities/shifts as well as order relative to the released generator. The
`author2017` and `paper-equations` profiles differ in primary scaling/bias and
classification-loss normalization; the latter was not trained. PyTorch/RTX 5090
is not a matched TensorFlow/A100 comparison. Neither multi-seed reproducibility
nor literal completion of issue #46 is established.

The measured six scientific files used Windows CRLF. Public LF files have
identical normalized text/AST but a different byte-level source binding.
Observed native results belong to the measured source; the public LF files
were not run as a second GPU experiment. Existing measured checkpoints cannot
be relabeled to the public LF digest. [RERUN.md](RERUN.md) gives the actual
public CLI, pinned input materializer, runtime identity and external supervision
requirements. Public graceful-resume support does not authorize a new measured
test admission or erase an unexpected session lock.
