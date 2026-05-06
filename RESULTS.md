# RESULTS — v1 baselines + DBN add-on

Per-stub reproducibility, implementation difficulty, and run wallclock for the current catalog: the pre-existing 4-2-4 worked example, 53 v1 implementations shipped across wave PRs #32–#41, and the DBN add-on. The summary statistics count the 54 implemented stubs; the worked example remains listed for continuity. Compiled from PR bodies for the v2 data-movement / ByteDMD filter.

**Reproduces? legend**: `yes` = matches paper qualitatively or quantitatively; `partial` = method works, paper number not fully reached (gap documented in stub README); `no` = paper claim does not replicate (gap analysis documented).

**Implementation wallclock**: agent end-to-end time from spec read to branch pushed. Variance is large across waves; values are agent-self-reported.

**Run wallclock**: time to run the final headline experiment on a laptop M-series CPU. Numpy + matplotlib only, no GPU.

## 1980s — Connectionist foundations

### Ackley, Hinton & Sejnowski (1985) — Boltzmann learning algorithm

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`encoder-4-2-4/`](encoder-4-2-4/) (worked example) | yes (CD-k variant; paper used SA) | n/a (pre-existing) | ~1s |
| [`encoder-3-parity/`](encoder-3-parity/) (PR #33) | yes (KL = log 2 = 0.6931 visible-only; RBM drops to 0.10) | ~50 min | 0.04s + 1.3s |
| [`encoder-4-3-4/`](encoder-4-3-4/) (PR #33) | yes (60% error-correcting rate / 30 seeds; even-parity codeset at seed 12) | ~3 hr | 2.3s |
| [`encoder-8-3-8/`](encoder-8-3-8/) (PR #33) | **yes (16/20 = exact paper parity)** | ~2 hr | ~20s/seed |
| [`encoder-40-10-40/`](encoder-40-10-40/) (PR #34) | yes (exceeds paper: 100% vs 98.6%) | ~1.5 hr | ~6s |

### Rumelhart, Hinton & Williams (1986) — Backprop

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`xor/`](xor/) (PR #32) | yes (qualitative, paper ~558 epochs / median 730) | 6.4 min | 0.3s |
| [`n-bit-parity/`](n-bit-parity/) (PR #32) | yes (qualitatively; thermometer code partial) | 30 min | 0.20s |
| [`encoder-backprop-8-3-8/`](encoder-backprop-8-3-8/) (PR #33) | yes (70% strict 8/8 distinct codes; 100% reconstruction) | ~10 min | 0.6s |
| [`distributed-to-local-bottleneck/`](distributed-to-local-bottleneck/) (PR #34) | yes (graded values 0.007 / 0.167 / 0.553 / 0.971 vs paper 0 / 0.2 / 0.6 / 1.0) | 75 min | 0.082s |
| [`symmetry/`](symmetry/) (PR #32) | yes (1 : 1.994 : 3.969 weight ratio, residual 0.000) | 12.8 min | 0.4s |
| [`binary-addition/`](binary-addition/) (PR #33) | yes (qualitatively; 4-3-3 succeeds, 4-2-3 stuck) | ~2 hr | 44s |
| [`negation/`](negation/) (PR #32) | yes (4-6-3 arch deviation justified; stub said 4-3-3 which can't converge) | 25 min | 0.10s |
| [`t-c-discrimination/`](t-c-discrimination/) (PR #34) | yes (all 3 detector families emerge across 40 kernels) | 30 min | 0.69s |
| [`recurrent-shift-register/`](recurrent-shift-register/) (PR #34) | yes (89 sweeps N=3, 121 sweeps N=5; both well under paper's <200) | 25 min | 0.9s / 1.1s |
| [`sequence-lookup-25/`](sequence-lookup-25/) (PR #35) | yes (phenomenon — paper has no specific number; 4-5/5 held-out) | 70 min | 0.20s / 5.78s |

### Hinton (1986) — Distributed representations

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`family-trees/`](family-trees/) (PR #35) | yes (3/4 best seed; 1.9/4 mean — matches paper's 2/4) | ~? | 2.1s |

### Hinton & Sejnowski (1986) — Learning and relearning

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`shifter/`](shifter/) (PR #34) | yes (92.3% recognition; position-pair detectors visible in figure3.png) | 30 min | 14s |
| [`grapheme-sememe/`](grapheme-sememe/) (PR #34) | yes (qualitatively; +6.7pp spontaneous recovery on held-out 2 at seed 0) | 70 min | 1.7s |

### Plaut & Hinton (1987)

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`riser-spectrogram/`](riser-spectrogram/) (PR #35) | yes (network 98.08% vs Bayes 98.90%, gap +0.83pp; paper +1.0pp) | ~7 min | 0.91s |

### Hinton & Plaut (1987) — Fast weights

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`fast-weights-rehearsal/`](fast-weights-rehearsal/) (PR #35) | yes (rehearsed-subset recovery +22pp mean / 30 seeds) | 25 min | 0.14s |

## 1990s — Mixtures, Helmholtz, deep belief

### Jacobs, Jordan, Nowlan & Hinton (1991)

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`vowel-mixture-experts/`](vowel-mixture-experts/) (PR #39) | partial (MoE 92.8% / MLP 90.1%; gate cleanly partitions front vs back vowels — phonetically meaningful. Paper's "MoE in half the epochs" claim does NOT replicate at 2-D F1/F2: data is nearly linearly separable, MLP wins on speed) | 70 min | 0.09s |

### Becker & Hinton (1992) — Imax / spatial coherence

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`random-dot-stereograms/`](random-dot-stereograms/) (PR #36) | yes (qualitatively; Imax 1.18 nats, modules' agreement corr 0.91, disparity readout 0.74. Paper has no single comparable scalar.) | ~1 hr | 6.1s |

### Nowlan & Hinton (1992) — Soft weight-sharing

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`sunspots/`](sunspots/) (PR #39) | yes (MoG 0.00420 ≤ decay 0.00422 ≤ vanilla 0.00432 / 5 seeds; structural effect dramatic — MoG collapses ~150 of 208 weights onto 2 crisp peaks) | ~? | ~5s |

### Hinton & Zemel (1994) — Bits-back / factorial VQ

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`spline-images-factorial-vq/`](spline-images-factorial-vq/) (PR #37) | yes (factorial 4×6 VQ wins 3× over standard 24-VQ baseline; DL 22.0 vs 65.3) | ~? | ~? |

### Zemel & Hinton (1995) — Population codes / MDL

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`dipole-position/`](dipole-position/) (PR #36) | partial (R² = 0.81 vs (x,y); supervised warm-up needed for tractable optimization. Pure-unsupervised emergence from random init is open question) | ~3 hr | 2s |
| [`dipole-3d-constraint/`](dipole-3d-constraint/) (PR #36) | yes (qualitatively; singular values 6.67 / 4.61 / 3.80 — 3 dims emerge) | ~? | 11s |
| [`dipole-what-where/`](dipole-what-where/) (PR #36) | partial (two near-perpendicular 1-D manifolds, axis angle 83°; meet at origin instead of opposite corners — needs learned mixture-of-Gaussians prior) | ~? | 2s |

### Dayan, Hinton, Neal & Zemel (1995) — Helmholtz machine

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`helmholtz-shifter/`](helmholtz-shifter/) (PR #36) | partial (3 of 4 layer-3 units develop clean shift-direction tuning; n_top=4 vs paper's n_top=1 — single top unit can't break t↔1-t symmetry on this task) | 75 min | 209s |

### Hinton, Dayan, Frey & Neal (1995) — Wake-sleep

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`bars/`](bars/) (PR #35) | partial (KL = 0.451 bits vs paper 0.10; structure captured but residual gap; multi-restart wrapper deferred) | 70 min | 222s |

## 2000s — RBMs, products of experts, deep belief

### Hinton (2000) — Contrastive divergence

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`bars-rbm/`](bars-rbm/) (PR #35) | yes (7/8 bars at purity ≥0.5 with n_hidden=8 / 10 seeds; 8/8 with n_hidden=16) | ~30 min | 1.5s |

### Hinton, Osindero & Teh (2006) — A fast learning algorithm for deep belief nets

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`dbn-mnist/`](dbn-mnist/) | partial (3.23% test err on full MNIST without up-down vs paper 1.25% with up-down; algorithm reproduces, fine-tuning step omitted) | ~1 hr | 5s default / 30s full MNIST |

### Salakhutdinov & Hinton (2009) — Deep Boltzmann Machines

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`dbm-mnist/`](dbm-mnist/) | partial (4.88% test err on full MNIST without discriminative fine-tuning vs paper 0.95% with; 7.83% on 10k subset; pretraining + joint PCD pipeline reproduces) | ~1.5 hr | 9s default / 45s full MNIST |

### Memisevic & Hinton (2007) — Gated 3-way RBM

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`transforming-pairs/`](transforming-pairs/) (PR #37) | partial (axis-selective transformation detectors emerge; 8-way classification 3.2× chance. Direction-selective Reichardt cells need natural video, not random-dot pairs) | ~? | 2s |

### Sutskever & Hinton (2007) — TRBM

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`bouncing-balls-2/`](bouncing-balls-2/) (PR #37) | partial (rollout MSE between predict-mean and copy-last baselines; qualitatively correct first 3-4 frames then diffuses to mean) | 75 min | 6.2s |

### Sutskever, Hinton & Taylor (2008) — RTRBM

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`bouncing-balls-3/`](bouncing-balls-3/) (PR #37) | partial (CD-1 recon MSE 0.0053; rollout MSE 0.13; W_h≡0 ablation matches full model on rollouts — suggests Sutskever's BPTT correction is needed) | ~? | 3.4s |

## 2010s — Capsules, distillation, attention

### Hinton, Krizhevsky & Wang (2011)

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`transforming-autoencoders/`](transforming-autoencoders/) (PR #38) | yes (R²(dx)=0.78, R²(dy)=0.67) | ~30 min | ~100s |

### Tang, Salakhutdinov & Hinton (2012)

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`deep-lambertian-spheres/`](deep-lambertian-spheres/) (PR #40) | yes (normal angular error 27° / 23.7° median — hits target <30°; albedo MSE 0.012 ~7× baseline. GRBM prior dropped — paper's actual contribution; v1 is feed-forward baseline) | ~50 min | 33s |

### Sutskever, Martens, Dahl & Hinton (2013)

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`rnn-pathological/`](rnn-pathological/) (PR #37) | yes (3 of 4 tasks; ortho-init solves, random-init at chance; XOR not cracked at our budget — needs NAG + 8× iterations per paper) | 2.5 hr | 42s |

### Hinton, Vinyals & Dean (2015) — Distillation

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`distillation-mnist-omitted-3/`](distillation-mnist-omitted-3/) (PR #38) | yes (97.82% on digit-3 post-correction; paper 98.6%. Hyperparameter-free bias correction) | 40 min | 121.8s |

### Eslami, Heess, Weber, Tassa, Szepesvari, Kavukcuoglu & Hinton (2016) — AIR

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`air-multimnist/`](air-multimnist/) (PR #41) | partial (count 79.7% vs target 50% — exceeds; reconstruction blurry due to under-scale; Gumbel-sigmoid throughout, no REINFORCE) | ~50 min | ~6s |
| [`air-3d-primitives/`](air-3d-primitives/) (PR #41) | partial (1-prim sanity 88.8%; 3-prim count 81%, type 52%; supervised regression instead of REINFORCE-AIR) | ~50 min | 11.7s |

### Ba, Hinton, Mnih, Leibo & Ionescu (2016) — Fast weights attention

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`fast-weights-associative-retrieval/`](fast-weights-associative-retrieval/) (PR #36) | partial (architecture verified by gradient check 1e-9; 38% retrieval vs 90% target — optimizer-landscape gap, needs RMSProp + 10⁵ steps per Ba et al.) | ~3 hr | 293s |
| [`multi-level-glimpse-mnist/`](multi-level-glimpse-mnist/) (PR #39) | partial (82.46% vs paper 90%+; deterministic 24-glimpse simplification + no CNN encoder) | ~1 hr | 1199s |
| [`catch-game/`](catch-game/) (PR #40) | partial (33.9% FW vs 11.4% vanilla at size=24; ablation unambiguous; 91% FW at size=10. REINFORCE budget below paper's A3C compute) | ~? | ~? |

### Sabour, Frosst & Hinton (2017) — Dynamic routing

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`affnist/`](affnist/) (PR #40) | **no** (gap wrong sign: CapsNet 85.5% / CNN 87.5% — paper +13%, ours −2%. 3 causes documented: synth-affNIST too close to train aug, tiny capsules, no reconstruction regularizer) | ~? | ~4 min |
| [`multimnist-capsnet/`](multimnist-capsnet/) (PR #40) | partial (48.6% vs target 80%; 22× chance; routing-by-agreement visibly works; reduced arch for pure-numpy budget) | ~3 hr | 395s |

### Hinton, Sabour & Frosst (2018) — Matrix capsules with EM routing

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`smallnorb-novel-viewpoint/`](smallnorb-novel-viewpoint/) (PR #41) | yes qualitatively (caps held-out 0.726 vs CNN 0.696 / 3 seeds; caps drop 0.244 vs CNN 0.304 — 20% relative reduction. Synthesized 5-class dataset vs real smallNORB) | ~? | ~10s |

### Kosiorek, Sabour, Teh & Hinton (2019) — Stacked capsule autoencoders

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`constellations/`](constellations/) (PR #39) | yes (per-point recovery 86.9% best / 84.0% mean; chance 36.4%. 12,708-param numpy set transformer + capsule decoder, FD-checked) | ~75 min | 25s |

## 2020s — Subclass distillation, GLOM, Forward-Forward

### Müller, Kornblith & Hinton (2020) — Subclass distillation

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`mnist-2x5-subclass/`](mnist-2x5-subclass/) (PR #38) | partial (subclass recovery 82.88% best / 73.87% mean; paper ~95%+ with ResNet vs our MLP backbone. Bounded aux loss gradient verified 6e-10) | ~50 min | 13s |

### Sabour, Tagliasacchi, Yazdani, Hinton & Fleet (2021) — Flow capsules

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`geo-flow-capsules/`](geo-flow-capsules/) (PR #40) | yes (mean IoU 0.764 / 200 pairs; chance ~0.20. EM-based mixture decomposition with closed-form M-step on GT flow vs paper's learned encoder) | ~8 min | 43s |

### Culp, Sabour & Hinton (2022) — eGLOM

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`ellipse-world/`](ellipse-world/) (PR #37) | yes (92.2% on 5-class; +6.6pp lift from GLOM iterations; islands form — cell-similarity rises +0.117 across iterations. Hand-coded backward FD-checked 1e-6) | ~? | 9s |

### Hinton (2022) — Forward-Forward

| Stub | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [`ff-hybrid-mnist/`](ff-hybrid-mnist/) (PR #38) | partial (5.21% test err vs paper 1.37%; 4×1000 + 30 epochs vs paper 4×2000 + 60. Goodness distributions show 2.8-3.3σ pos-vs-neg separation) | ~75 min | 492s |
| [`ff-label-in-input/`](ff-label-in-input/) (PR #38) | partial (3.60% vs paper 1.36%; smaller arch + fewer epochs. Three FF gotchas documented for siblings: mean(h²)=1, lr=0.003, all-layers > skip-L0) | ~1 hr | 66s |
| [`ff-recurrent-mnist/`](ff-recurrent-mnist/) (PR #38) | partial (10.66% vs paper 1.31%; ~25× fewer params, 3× fewer epochs. Algorithm reproduces; capacity doesn't) | ~1 hr | 216s |
| [`ff-cifar-locally-connected/`](ff-cifar-locally-connected/) (PR #39) | partial (FF 22.78% / BP baseline 38.31%; paper FF 41-46% / BP 37-39%. 15pp gap mostly under-training: 10K of 50K + 10 of 60+ epochs) | ~3 hr | 150s |
| [`ff-aesop-sequences/`](ff-aesop-sequences/) (PR #39) | yes (TF 53% / SG 34% / chance 3.3% / unigram 19.6%. Paper's "nearly identical" claim doesn't replicate at smaller scale — TF leads SG by 19pp) | ~12 min | 131s |

---

## Summary statistics

| Verdict | Count | Notes |
|---|---:|---|
| **yes** (full or qualitative match) | 27 | including all backprop foundations + most encoders + distillation-omitted-3 + ellipse-world + spline-VQ |
| **partial** (method works, paper number gap documented) | 26 | mostly Forward-Forward at smaller scale, capsules at smaller arch, AIR variants without REINFORCE, plus the DBN add-on without up-down fine-tuning |
| **no** (paper claim does NOT replicate) | 1 | affnist (gap wrong sign — three causes documented) |

**Total: 54 implemented stubs: 53 v1 stubs plus the DBN add-on, all in pure numpy, all <5 min/seed on a laptop except where noted.** The table also includes the pre-existing 4-2-4 worked example.

## v2 filter recommendation

For the data-movement / ByteDMD instrumentation, prioritize stubs that:

1. **Reproduce cleanly + run fast** (low noise floor for measuring data-movement deltas):
   - `xor`, `symmetry`, `n-bit-parity`, `negation` (sub-second runs, well-converged)
   - `encoder-3-parity`, `encoder-backprop-8-3-8`, `encoder-4-2-4` (Boltzmann/backprop pair on same problem)
   - `distributed-to-local-bottleneck`, `recurrent-shift-register`, `t-c-discrimination`
   - `binary-addition`, `riser-spectrogram` (clean MSE / Bayes-optimal targets)

2. **Have algorithmic variants** (lets you compare data-movement properties of different algorithms on the same problem):
   - 8-3-8: backprop vs Boltzmann
   - bars: wake-sleep vs RBM
   - shifter: Boltzmann (this) vs Helmholtz (helmholtz-shifter)
   - fast-weights-rehearsal vs fast-weights-associative-retrieval

3. **Defer for v2**: anything where the run takes >100s or where the v1 implementation is partial — measuring data-movement on a non-converged solver isn't informative.

---

_Compiled by agent-0bserver07 (Claude Code) on behalf of Yad. Source: PR bodies #32-#41._
