# Hinton Problems

A reproducible-baseline catalog of the synthetic learning problems that appear in Geoffrey Hinton's experimental papers from 1981 through 2022 — implemented in pure numpy, runnable on a laptop CPU, with paper-comparison metrics per stub.

**Site**: https://cybertronai.github.io/hinton-problems/ • **Catalog**: [RESULTS.md](RESULTS.md) • **55 of 55 stubs implemented** (PRs #32–#41 + DBN + DBM add-ons)

## Introduction

> The field has standardized on backprop by the end of the '80s, and Hinton gives a sample of problems that were used at the time. In the last 20 years, we have transitioned to GPUs, and the math has changed considerably. Instead of being bottlenecked by arithmetic, the shrinking of transistors means that arithmetic is essentially free, and all of the work comes from data movement. **Backprop is inefficient in terms of "commute to compute ratio"** because it requires fetching all of the activations for each gradient add.
>
> So a natural experiment would be to redo key experiments of this time with a focus on data movement. The first step is to get a baseline — to establish the list of problems which are famous (made by Hinton), reasonable to implement, and easy to run/reproduce.
>
> — Yaroslav, [issue #1](https://github.com/cybertronai/hinton-problems/issues/1#issuecomment-4363088986) (Sutro Group)

This repository **is that baseline**. v1 ships 53 implementations covering the lineage from the 4-2-4 encoder (1985) through the shifter (1986), bars (1995), MultiMNIST (2017), Constellations (2019), Ellipse World (2022), and the Forward-Forward suite (2022). Each stub is a self-contained folder with model + train + eval + visualization + animated GIF, all in numpy, all runnable in <5 min per seed on an M-series laptop.

The next step ([#45 v2](https://github.com/cybertronai/hinton-problems/issues/45)) instruments these 53 baselines with [ByteDMD](https://github.com/cybertronai/ByteDMD) — Yaroslav's data-movement cost tracer — to measure the actual "commute" each algorithm pays.

## What's here

| 27 reproduce paper claims | 27 partial reproductions | 1 non-replication |
| :---: | :---: | :---: |
| full or qualitative match | algorithm works, paper-config gap documented | gap analysed in 3 causes |

Pure numpy + matplotlib throughout. Every stub runs on a laptop CPU. Each problem lives in its own folder with `<slug>.py` (model + train + eval), `README.md`, `make_<slug>_gif.py`, `visualize_<slug>.py`, an animated `<slug>.gif`, and a `viz/` folder of training curves and weight visualizations.

## Development

This repository includes a minimal Nix development shell with Python and
NumPy:

```bash
nix develop
python v2-bytedmd/validate_implementations.py
python v2-bytedmd/total_cost_comparison.py
```

Or run one command directly:

```bash
nix develop -c python v2-bytedmd/validate_implementations.py
```

## Visual tour

| ![encoder-4-2-4](encoder-4-2-4/encoder.gif) | ![spline-images-factorial-vq](spline-images-factorial-vq/spline_images_factorial_vq.gif) |
| :---: | :---: |
| [`encoder-4-2-4`](encoder-4-2-4/) — Ackley/Hinton/Sejnowski 1985, the worked example. Bipartite RBM, 2-bit code emerges. | [`spline-images-factorial-vq`](spline-images-factorial-vq/) — Hinton/Zemel 1994, factorial VQ wins 3× over standard 24-VQ baseline. |
| ![ellipse-world](ellipse-world/ellipse_world.gif) | ![ff-recurrent-mnist](ff-recurrent-mnist/ff_recurrent_mnist.gif) |
| [`ellipse-world`](ellipse-world/) — Culp/Sabour/Hinton 2022, eGLOM islands form across iterations (5-class, 92.2%). | [`ff-recurrent-mnist`](ff-recurrent-mnist/) — Hinton 2022, top-down recurrent Forward-Forward. |

For the long-form picture-first walk through **all 53 stubs** — every GIF, organized by year, with notes on what each visualization is meant to show — see [`VISUAL_TOUR.md`](VISUAL_TOUR.md).

## Catalog

Each table shows the v1 result per stub. Full per-stub metrics (compile-time, GIF size, headline numbers) are in [`RESULTS.md`](RESULTS.md).

**Reproduces?** legend: `yes` = matches paper qualitatively or quantitatively; `partial` = method works, paper number not fully reached (gap documented in stub README); `no` = paper claim does not replicate.

### 1980s — Connectionist foundations

**Ackley, Hinton & Sejnowski (1985)** — A learning algorithm for Boltzmann machines

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [encoder-4-2-4](encoder-4-2-4/) ★ | yes (CD-k variant) | n/a (worked example) | ~1s |
| [encoder-3-parity](encoder-3-parity/) | yes (KL = log 2 visible-only; RBM drops to 0.10) | ~50 min | 0.04s + 1.3s |
| [encoder-4-3-4](encoder-4-3-4/) | yes (60% error-correcting rate / 30 seeds) | ~3 hr | 2.3s |
| [encoder-8-3-8](encoder-8-3-8/) | **yes (16/20 = exact paper parity)** | ~2 hr | ~20s/seed |
| [encoder-40-10-40](encoder-40-10-40/) | yes (exceeds paper: 100% vs 98.6%) | ~1.5 hr | 6s |

**Rumelhart, Hinton & Williams (1986)** — Learning internal representations by error propagation

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [xor](xor/) | yes (qualitative) | 6.4 min | 0.3s |
| [n-bit-parity](n-bit-parity/) | yes (qualitative; thermometer code partial) | 30 min | 0.20s |
| [encoder-backprop-8-3-8](encoder-backprop-8-3-8/) | yes (70% strict 8/8 distinct codes) | ~10 min | 0.6s |
| [distributed-to-local-bottleneck](distributed-to-local-bottleneck/) | yes (graded values 0.007/0.167/0.553/0.971) | 75 min | 0.082s |
| [symmetry](symmetry/) | yes (1 : 1.994 : 3.969 weight ratio) | 12.8 min | 0.4s |
| [binary-addition](binary-addition/) | yes (qualitatively; 4-3-3 succeeds, 4-2-3 stuck) | ~2 hr | 44s |
| [negation](negation/) | yes (4-6-3 deviation justified) | 25 min | 0.10s |
| [t-c-discrimination](t-c-discrimination/) | yes (all 3 detector families emerge) | 30 min | 0.69s |
| [recurrent-shift-register](recurrent-shift-register/) | yes (89 sweeps N=3, 121 sweeps N=5) | 25 min | 0.9s / 1.1s |
| [sequence-lookup-25](sequence-lookup-25/) | yes (4-5/5 held-out generalization) | 70 min | 0.20s / 5.78s |

**Hinton (1986)** — Distributed representations of concepts

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [family-trees](family-trees/) | yes (3/4 best, 1.9/4 mean — matches paper) | ~1 hr | 2.1s |

**Hinton & Sejnowski (1986)** — Learning and relearning in Boltzmann machines

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [shifter](shifter/) | yes (92.3% recognition; position-pair detectors) | 30 min | 14s |
| [grapheme-sememe](grapheme-sememe/) | yes (qualitative; +6.7pp spontaneous recovery) | 70 min | 1.7s |

**Plaut & Hinton (1987)** — Learning sets of filters using back-propagation

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [riser-spectrogram](riser-spectrogram/) | yes (98.08% net vs 98.90% Bayes; gap +0.83pp) | ~7 min | 0.91s |

**Hinton & Plaut (1987)** — Using fast weights to deblur old memories

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [fast-weights-rehearsal](fast-weights-rehearsal/) | yes (rehearsed-subset recovery +22pp / 30 seeds) | 25 min | 0.14s |

### 1990s — Unsupervised learning, mixtures, the Helmholtz machine

**Jacobs, Jordan, Nowlan & Hinton (1991)** — Adaptive mixtures of local experts

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [vowel-mixture-experts](vowel-mixture-experts/) | partial (MoE 92.8% / MLP 90.1%; gate partitions vowels) | 70 min | 0.09s |

**Becker & Hinton (1992)** — A self-organizing neural network that discovers surfaces in random-dot stereograms

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [random-dot-stereograms](random-dot-stereograms/) | yes (Imax 1.18 nats; disparity readout 0.74) | ~1 hr | 6.1s |

**Nowlan & Hinton (1992)** — Simplifying neural networks by soft weight-sharing

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [sunspots](sunspots/) | yes (MoG ≤ decay ≤ vanilla; weight peaks at 0 + 0.27) | ~1 hr | 5s |

**Hinton & Zemel (1994)** — Autoencoders, MDL and Helmholtz free energy

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [spline-images-factorial-vq](spline-images-factorial-vq/) | yes (factorial wins 3× over 24-VQ baseline) | ~1 hr | ~5s |

**Zemel & Hinton (1995)** — Learning population codes by minimizing description length

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [dipole-position](dipole-position/) | partial (R² = 0.81; supervised warm-up needed) | ~3 hr | 2s |
| [dipole-3d-constraint](dipole-3d-constraint/) | yes (qualitatively; 3 dims emerge) | ~1 hr | 11s |
| [dipole-what-where](dipole-what-where/) | partial (perpendicular manifolds, lin-sep 0.58) | ~1 hr | 2s |

**Dayan, Hinton, Neal & Zemel (1995)** — The Helmholtz machine

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [helmholtz-shifter](helmholtz-shifter/) | partial (3 of 4 layer-3 units shift-selective; n_top=4) | 75 min | 209s |

**Hinton, Dayan, Frey & Neal (1995)** — The wake-sleep algorithm

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [bars](bars/) | partial (KL = 0.451 bits vs paper 0.10) | 70 min | 222s |

### 2000s — Products of experts, contrastive divergence, deep belief nets

**Hinton (2000)** — Training products of experts by minimizing contrastive divergence

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [bars-rbm](bars-rbm/) | yes (7/8 bars at purity ≥0.5; 8/8 with n_hidden=16) | ~30 min | 1.5s |

**Hinton, Osindero & Teh (2006)** — A fast learning algorithm for deep belief nets

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [dbn-mnist](dbn-mnist/) | partial (3.23% w/o up-down vs paper 1.25% w/ up-down; 6.04% on 10k subset) | ~1 hr | 5s / 30s |

**Salakhutdinov & Hinton (2009)** — Deep Boltzmann Machines

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [dbm-mnist](dbm-mnist/) | partial (4.88% w/o discriminative fine-tuning vs paper 0.95% with; 7.83% on 10k subset) | ~1.5 hr | 9s / 45s |

**Memisevic & Hinton (2007)** — Unsupervised learning of image transformations

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [transforming-pairs](transforming-pairs/) | partial (axis-selective transformation detectors) | ~1 hr | 2s |

**Sutskever & Hinton (2007)** — Multilevel distributed representations for high-dimensional sequences

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [bouncing-balls-2](bouncing-balls-2/) | partial (rollout MSE between baselines) | 75 min | 6.2s |

**Sutskever, Hinton & Taylor (2008)** — The recurrent temporal RBM

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [bouncing-balls-3](bouncing-balls-3/) | partial (CD-1 recon 0.005; rollout 0.13) | ~1 hr | 3.4s |

### 2010s — Capsules, distillation, attention

**Hinton, Krizhevsky & Wang (2011)** — Transforming auto-encoders

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [transforming-autoencoders](transforming-autoencoders/) | yes (R²(dx)=0.78, R²(dy)=0.67) | ~30 min | 100s |

**Tang, Salakhutdinov & Hinton (2012)** — Deep Lambertian Networks

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [deep-lambertian-spheres](deep-lambertian-spheres/) | yes (normal angular err 27°; albedo 7× baseline) | ~50 min | 33s |

**Sutskever, Martens, Dahl & Hinton (2013)** — On the importance of initialization and momentum

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [rnn-pathological](rnn-pathological/) | yes (3 of 4 tasks; ortho beats random init) | 2.5 hr | 42s |

**Hinton, Vinyals & Dean (2015)** — Distilling the knowledge in a neural network

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [distillation-mnist-omitted-3](distillation-mnist-omitted-3/) | yes (97.82% on digit-3 post-correction; paper 98.6%) | 40 min | 121.8s |

**Eslami, Heess, Weber, Tassa, Szepesvari, Kavukcuoglu & Hinton (2016)** — Attend, Infer, Repeat

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [air-multimnist](air-multimnist/) | partial (count 79.7%; reconstructions blurry) | ~50 min | 6s |
| [air-3d-primitives](air-3d-primitives/) | partial (1-prim 88.8%; 3-prim count 81%) | ~50 min | 11.7s |

**Ba, Hinton, Mnih, Leibo & Ionescu (2016)** — Using fast weights to attend to the recent past

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [fast-weights-associative-retrieval](fast-weights-associative-retrieval/) | partial (architecture verified; 38% retrieval) | ~3 hr | 293s |
| [multi-level-glimpse-mnist](multi-level-glimpse-mnist/) | partial (82.46% vs paper 90%+) | ~1 hr | 1199s |
| [catch-game](catch-game/) | partial (FW 33.9% vs vanilla 11.4%; 91% at size=10) | ~2 hr | ~50s |

**Sabour, Frosst & Hinton (2017)** — Dynamic routing between capsules

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [affnist](affnist/) | **no** (gap wrong sign: −2% vs paper +13%) | ~3 hr | 4 min |
| [multimnist-capsnet](multimnist-capsnet/) | partial (48.6% vs target 80%; 22× chance) | ~3 hr | 395s |

**Hinton, Sabour & Frosst (2018)** — Matrix capsules with EM routing

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [smallnorb-novel-viewpoint](smallnorb-novel-viewpoint/) | yes qualitatively (caps 0.726 vs CNN 0.696 held-out) | ~1 hr | 10s |

**Kosiorek, Sabour, Teh & Hinton (2019)** — Stacked capsule autoencoders

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [constellations](constellations/) | yes (per-point recovery 86.9% best / 84% mean) | ~75 min | 25s |

### 2020s — Subclass distillation, GLOM, Forward-Forward

**Müller, Kornblith & Hinton (2020)** — Subclass distillation

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [mnist-2x5-subclass](mnist-2x5-subclass/) | partial (subclass recovery 82.88% best / 73.87% mean) | ~50 min | 13s |

**Sabour, Tagliasacchi, Yazdani, Hinton & Fleet (2021)** — Unsupervised part representation by flow capsules

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [geo-flow-capsules](geo-flow-capsules/) | yes (mean IoU 0.764 / chance 0.20) | ~8 min | 43s |

**Culp, Sabour & Hinton (2022)** — Testing GLOM's ability to infer wholes from ambiguous parts

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [ellipse-world](ellipse-world/) | yes (92.2% on 5-class; islands form +0.117) | ~1 hr | 9s |

**Hinton (2022)** — The forward-forward algorithm: some preliminary investigations

| Problem | Reproduces? | Implementation | Run wallclock |
|---|---|---:|---:|
| [ff-hybrid-mnist](ff-hybrid-mnist/) | partial (5.21% test err vs paper 1.37%) | ~75 min | 492s |
| [ff-label-in-input](ff-label-in-input/) | partial (3.60% vs paper 1.36%) | ~1 hr | 66s |
| [ff-recurrent-mnist](ff-recurrent-mnist/) | partial (10.66% vs paper 1.31%) | ~1 hr | 216s |
| [ff-cifar-locally-connected](ff-cifar-locally-connected/) | partial (FF 22.78% / BP 38.31%) | ~3 hr | 150s |
| [ff-aesop-sequences](ff-aesop-sequences/) | yes (TF 53% / SG 34%; baselines 3-20%) | ~12 min | 131s |

## Structure

```
problem-folder/
├── README.md                  source paper, problem, results, deviations
├── <slug>.py                  dataset + model + train + eval
├── visualize_<slug>.py        training curves + weight viz
├── make_<slug>_gif.py         animated GIF
├── <slug>.gif                 committed animation
└── viz/                       committed PNGs
```

## Roadmap

- [**#45 v2: ByteDMD instrumentation**](https://github.com/cybertronai/hinton-problems/issues/45) — measure data-movement cost per stub on these baselines (the actual research goal)
- [**#46 v1.5: paper-scale reruns**](https://github.com/cybertronai/hinton-problems/issues/46) — close the 25 partial reproductions on Modal/GPU
- See `Open questions / next experiments` section in each stub README for stub-specific follow-ups

## Contributing

Implementations follow the [v1 spec](https://github.com/cybertronai/hinton-problems/issues/1):

- Each stub fills in `<slug>.py` (model + train + eval), an 8-section `README.md`, `make_<slug>_gif.py`, `visualize_<slug>.py`, an animated `<slug>.gif`, and `viz/` PNGs.
- Acceptance: reproduces in <5 min on a laptop; final accuracy with seed in Results table; GIF illustrates problem AND learning dynamics; "Deviations from the original" section honest; at least one open question.
- v1 metrics in PR body: `"Paper reports X; we got Y. Reproduces: yes/no."` + run wallclock + implementation wallclock.

The v1.5 reruns ([#46](https://github.com/cybertronai/hinton-problems/issues/46)) and v2 ByteDMD work ([#45](https://github.com/cybertronai/hinton-problems/issues/45)) welcome contributions.

## License

The hinton-problems source and documentation are released into the public domain under the [Unlicense](LICENSE).
