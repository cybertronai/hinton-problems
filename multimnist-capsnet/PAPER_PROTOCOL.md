# Full-width MultiMNIST protocol

This separate full-width condition completed one 60M-pair exposure and a
final-checkpoint-only 10M-pair evaluation. The [measured report](PAPER_RUN.md)
records 6.07872% pair error, reconstruction statistics and limitations. The
historical reduced NumPy program, figures and scores remain unchanged. Its
smaller widths, pixelwise-max overlay, attempted IoU conditioning, stopped
routing coefficients and reconstruction reduction differ from this condition.
Its IoU sampler can exhaust its attempts and return a lower-overlap pair.

## Primary sources and scientifically different conditions

The primary references are Sabour, Frosst and Hinton's
[paper, version 2](https://arxiv.org/pdf/1710.09829v2), sections 2–6, and
[Sara Sabour's released code](https://github.com/Sarasra/models/tree/984fbc754943c849c55a57923f4223099a1ff88c/research/capsules)
at commit `984fbc754943c849c55a57923f4223099a1ff88c`. The implementation here
is independent; the TensorFlow implementation is a reference, not a runtime
dependency. The two named profiles must never be silently mixed.

| Detail | `author2017` | `paper-equations` comparison |
|---|---|---|
| Conv1 | 256 channels, 9x9, stride 1, valid, ReLU | same |
| PrimaryCaps | 32 types x 8 atoms, 9x9, stride 2; 3200 capsules on 36x36 | same widths |
| Primary preactivation | raw convolution / 32 + learned bias | raw convolution + learned bias |
| DigitCaps | 10 x 16 atoms; separate transform per primary/output pair; learned 10x16 bias | no DigitCaps bias |
| Routing | 3 iterations, softmax over output classes, full gradient through agreement | same |
| Decoder | shared 160 -> 512 -> 1024 -> 1296; ReLU/ReLU/sigmoid | same |
| Parameter count | 11,361,968 | 11,361,808 |
| Classification loss | one twentieth of summed-class Eq. 4 margin loss | summed-class Eq. 4 margin loss |
| Reconstruction loss | `.0005 * mean_batch(SSE_a + SSE_b)` | same |

The released `conv_slim_capsule` uses one routing iteration from one input
capsule type into 32 outputs. Zero routing logits give coefficients 1/32,
**before** the bias and squash. `layers.py` also applies a factor .5 to
classification terms and then averages over all 10 classes and the batch.
That changes the classification/reconstruction balance by a factor of 20
relative to the paper's summed-class expression. Both are implemented
explicitly. The preregistered primary condition is `author2017`; the recorded run executed
this condition. The
`paper-equations` profile changes the listed mathematical details while
retaining the released initialization scheme; it is not a claim of recovering
every unreported paper implementation choice.

Weights use independent truncated-normal draws within two standard deviations
(.05 for Conv1, .1 elsewhere), and learned biases start at .1. Random seeds do
not make PyTorch's draw sequence identical to TensorFlow's. Primary flattening
order is type, row, column, atom. Squash uses the continuous zero extension,
avoiding the original expression's division by zero. No stop-gradient is used
in the new routing implementation. Native forward/gradient equivalence is a
separate required gate; parsing source and scalar tests cannot establish it.

## Data identities and streaming

Only the original MNIST train split supplies training pairs; only the original
test split supplies official test pairs. Each of 60,000/10,000 anchors has
exactly 1000 different-label, same-split partners: 60,000,000/10,000,000 pairs.
Each anchor's partners are unique. Its shift is fixed across those 1000 pairs,
as in the released generator; partner shifts vary per pair. Both axis shifts
are signed integers in [-4,4]. A 28x28 image occupies origin `(4+dy,4+dx)` in
the 36x36 target. Neither glyph is cropped. This condition uses integer translations only: no
rotation, scaling or elastic distortion is applied. Composite bytes are
`min(a+b,255)`, then normalized by 255 for the model. There is no IoU rejection.
The paper's overlap description does not define an IoU admission threshold.
The released README's `--shift 6 --pad 4` example is not our signed-four-pixel
condition.

Pair identity is `anchor * 1000 + counter`. A keyed, six-round Feistel
permutation with bounded cycle walking selects unique eligible partners;
a separate permutation orders all pair identities for each exposure.
Unbiased hash/rejection choices select shifts. No global RNG state or
60-million-image array is needed. This changes the selected partner identities and shift realizations as well
as their ordering relative to the released NumPy random generator; it is not
merely a reshuffling of the author's exact image pairs. Cycle-walk exhaustion fails rather than
dropping/replacing a row. Batches retain a final partial batch and explicit
start/next cursors. The same generator, source bytes and cursors reproduce
the remaining stream after a restart.

The pinned [MNIST provider card](https://huggingface.co/datasets/ylecun/mnist/blob/77f3279092a1c1579b2250db8eafed0ad422088c/README.md)
declares MIT. Retain the MNIST attribution to Yann LeCun, Corinna Cortes and
Christopher J. C. Burges. `paper_data.py` accepts only hash-bound, complete
raw 28x28 uint8 projections from that revision, in original IDX order. Admission
must compare **every** provider image/label to original IDX before the raw
manifest is trusted. Merely matching parquet counts is insufficient. The
reader authenticates the same buffer it returns and never downloads data.
Images, reconstructed samples and checkpoints stay local; submit only code,
protocol, configuration and aggregate measurements.

## Preregistered primary budget and selection

The selected explicit protocol uses seed 20261107, one complete 60M-pair exposure,
batch 128, hence 468,750 updates. This is an explicit approximation: neither
the generic released `max_steps=1000` default nor its MNIST example establishes
the MultiMNIST paper's full training budget. Freeze the actual exposure/update
budget before viewing official test outcomes. This project has frozen exactly
that single exposure and final-only 10M test; it does not infer a paper duration.

The explicit Adam formula uses beta1=.9, beta2=.999 and epsilon=1e-8 added
after the uncorrected second-moment square root:
`alpha = lr * sqrt(1-beta2^t)/(1-beta1^t)` and
`parameter -= alpha * m/(sqrt(v)+epsilon)`.
The learning rate is `max(1e-6, .001 * .96^(completed_updates/20000))`.
The 20,000 period is the paper's ten-times-MNIST decay period applied to the
released 2,000-step MNIST setting. No staircase, weight decay or gradient
clipping is silently added. The actual pinned-Torch CPU gate and full-width GPU parity/serialized-continuation
controls passed before the recorded training run.

The proposal uses the **final** checkpoint for official test. Optional
diagnostics use 1000 deterministically selected training anchors, every 10,000
updates, each with partner counter 1000 (outside that anchor's training partner
set). These glyphs were still used in training; reversed-anchor image pairs
can recur. This is an ordered-pair diagnostic, **not a source-image holdout**.
Retain its best checkpoint separately, ranked by pair error then
paired reconstruction MSE, without using it to select the official test
checkpoint. The exact diagnostic schedule/anchor list must be frozen with the
experiment. The deterministic anchor ordering is keyed by seed and
`train-diagnostic-anchors`; these 1000 anchors and the extra partner counter
are identical at every diagnostic: 46 rounds yield 46,000 diagnostic pair
observations of 1000 distinct ordered diagnostic pairs, not 46,000 new training
pairs or optimizer exposures. No test-source image is used for selection.

`paper_train.py` now implements the finite control flow. The production CLI
accepts only the explicit default protocol: one 60M exposure/468,750 updates.
It streams batches, preserves tails, checkpoints every 1000 updates and at a new
diagnostic best, and always commits the final state. Checkpoints contain model,
complete Adam moments/update count, CPU/CUDA/Python RNG, pair/update/exposure
cursors, online loss sums and exact diagnostic/selection history. Source,
data, runtime and protocol identities must match before restoration. Model
keys/shapes/dtypes, optimizer formula/moments and RNG states are validated.

`paper_state.py` authenticates payloads before deserialization and retains only
latest/best plus bounded transient bytes. Native loading uses weights_only=True.
An exclusive session lock prevents concurrent trainers; unexpected exceptions or
kills preserve the lock and require explicit reconciliation, never stale-lock
stealing. An ordinary empty STOP file in the owned output requests a graceful
checkpoint. A stop inside a diagnostic preserves the prior authenticated
checkpoint and reports discarded uncommitted pairs, rather than publishing a
partial score. Resuming recomputes only the uncommitted trajectory.

Train and official test are separate CLI phases. The final training checkpoint
is authenticated and its complete budget checked **before** the test source is
opened. Diagnostic-best weights cannot select official test weights. Test
progress uses a separate bounded journal tied to the final checkpoint digest;
completed test batches resume without double-counted indices. Its generic
checkpoint-store `completedUpdates` column counts evaluation batches (78,125
for 10M pairs), never optimizer updates; results report those separately from
the frozen 468,750 training optimizer updates. A completed
journal returns its existing result without model reevaluation. An interrupted
uncommitted evaluation batch may be recomputed, but never added twice.

## Source identity and execution boundaries

The executed algorithm-source digest is
`b5b0f41285cdbf113304f797b42c4db45d412a35f479f3085d24db65d5670a18`.
The public LF Git blobs instead produce
`f5256dba596ba1ba9e075c1459fe69201b4f668c9a90edbd72b13307db528d9a`.
Each is SHA256 of canonical JSON mapping every `paper_*.py` filename to its
SHA256. Only line endings in `paper_engine.py` and `paper_train.py` differ;
they are not byte-identical source bindings. The public checkout therefore
needs its own run/checkpoint identity; do not relabel or transfer the executed
checkpoint to the LF digest.

The actual training continuation resumed an authenticated 256-update,
32,768-pair prefix, rather than restarting at pair zero. The public CLI supports
graceful checkpoint/cursor resume in both train and test phases. In contrast,
the measured final test used separately reviewed, single-use continuation
admissions after an executor exception and a gaming stop. All retained the same
final model, seed, pair ordering and cumulative aggregate; no committed indices
were counted twice and no fresh experiment was admitted. The unknown prior
uncommitted inference tail is disclosed in [the run record](PAPER_RUN.md).
Public CLI resumability is not permission to reset or repeat the measured test.

## Reproduce this condition

See [RERUN.md](RERUN.md) for pinned public inputs, manifest construction, private
environment identity, source-only/CPU controls, and the actual separate
train/test CLI commands. The ordinary commands use `python -s -B` inside the
private environment; bare `-I -S` does not supply its site-packages. The native
CPU gate uses an explicit isolated package-directory wrapper instead.

No package installation, data download, watchdog, thermal/power setting change
or automatic GPU choice is performed by the trainer. The outer operator must
authenticate inputs, limit resources and supervise authority/thermal/stop and
exact-owned cleanup conditions. The trainer is not an OS sandbox or a resource
grant. The measured run used the unchanged external 410W limit.

## Metrics and validation gates

Top-2 predictions are a set of distinct labels. Report raw counts and:

* pair error = pairs missing at least one true label / number of pairs;
* per-digit error = missed true labels / (2 * number of pairs);
* both-correct accuracy and at-least-one-correct accuracy separately;
* per-class true/predicted/true-positive counts, plus top-2 boundary ties.

The released author evaluation counts a pair correct only when **both** labels
are found. Table 1's MultiMNIST 5.2% and section 6.2's prose 5.0% must stay
distinct. Report the selected table comparator explicitly; do not relabel
the existing repository's 48.6% small-condition both-correct accuracy as a
paper-scale error result. Exact score ties use lower class index in this new
condition and are counted, rather than relying on an unspecified device tie.

Reconstruction must report label-conditioned paired SSE/MSE, separately from
predicted-mask reconstruction (permutation-matched target assignment) and
clipped-mixture conservation. Correct labels supplied to the decoder are not
evidence of successful label prediction or blind segmentation.

From this directory, current executable light checks use only tiny synthetic
bytes/scalars:

```sh
python -I -B -S -m unittest discover -s . -p 'test_paper_*.py' -v
python -I -B -S paper_protocol.py
```

`cpu_reference_gate.py` is the native CPU gate used before the recorded run.
In a separately approved private CPU environment it checks scalar-vs-tensor routing, finite
differences through agreement, both loss reductions, explicit Adam and serialized
optimizer/RNG/cursor continuation, plus actual full-width forward/backward.
It requires `CUDA_VISIBLE_DEVICES=-1` before importing Torch.

The finite trainer has stdlib synthetic control-flow tests. Before the recorded
run, the actual native CPU gate, full-width GPU CPU/parity checks and a 13-step
serialized GPU-continuation pilot passed. The real training then resumed its
accepted 256-update prefix and reached exactly 468,750 updates/60M pairs. See
[PAPER_RUN.md](PAPER_RUN.md) for measured timings, memory and remaining limits.
These controls establish tested numerical/continuation behavior, not paper
accuracy or generalization. A new run still needs independently authenticated
inputs and external resource/thermal/authority supervision; this module grants
no execution permission.
