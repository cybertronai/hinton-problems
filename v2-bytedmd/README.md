# v2 ByteDMD instrumentation

ByteDMD cost measurements for selected algorithm pairs from the
`hinton-problems` v1 baselines. Companion to
[issue #45](https://github.com/cybertronai/hinton-problems/issues/45).

## Measurement contract

Yaroslav's rule for ByteDMD/DALI comparisons is:

> Compare total data-movement cost to reach the agreed reference accuracy or
> solve criterion, not isolated per-step cost.

The v2 contract is therefore:

1. Compare algorithms on the same task, data, and reference criterion.
2. Implement a pure-Python kernel for one full training update.
3. Validate the pure-Python kernel against the NumPy reference where practical.
4. Measure ByteDMD for one full update.
5. Count how many full updates each algorithm needs to reach the reference
   criterion under the documented seeds.
6. Report `total_cost = median_steps_to_reference * bytedmd_per_update`.

Per-step ByteDMD and second-pass penalties are useful diagnostics, but they are
not the headline comparison. They can flip conclusions when one algorithm needs
many more updates to reach the same result.

## Current instrumentation semantics

`bytedmd.py` traces Python-level scalar reads through hand-written pure-Python
kernels. The score is the sum of `ceil(sqrt(reuse_distance))` over tracked
reads under its LRU-stack model with eager argument initialization and liveness
compaction.

This is honest ByteDMD instrumentation for algorithm-shaped scalar code. It is
not a DALI trace, not a BLAS trace, and not a hardware cache measurement. NumPy
reference stubs in the parent folders are used for correctness and convergence
counts; they are not directly traced, because vectorized operations hide the
scalar read order that ByteDMD needs.

## Canonical first example

`total_cost_comparison.py` is the canonical first v2 slice. It compares:

- `encoder-backprop-8-3-8`: full-batch MLP + gradient descent.
- `encoder-8-3-8`: RBM/Boltzmann encoder trained with CD-k.

Both solve the same 8 one-hot pattern task through a 3-bit bottleneck and have
48 traced weight values. The reference criterion is 100% reconstruction
accuracy and 8/8 distinct hidden codes.

Run:

```bash
nix develop -c python v2-bytedmd/validate_implementations.py
nix develop -c python v2-bytedmd/total_cost_comparison.py
```

Verified on 2026-05-06:

```text
Per full-batch update:
  Backprop 8-3-8 total          :     18,327
  Boltzmann CD-1 total          :     24,641
  Boltzmann CD-5 total          :     67,581

Steps to reference criterion, 10 seeds:
  Backprop solved               :       7/10
  Backprop median steps         :        982 full-batch updates
  Boltzmann solved              :       8/10
  Boltzmann median steps        :     35,936 full-batch CD updates

Total ByteDMD to reference criterion:
  Backprop                      :  17,997,114
  Boltzmann CD-1                : 885,498,976  (49.2x backprop)
  Boltzmann CD-5                : 2,428,590,816  (134.9x backprop)
```

The diagnostic second-pass penalties are still informative:

```text
Backprop backward/forward       : 1.82x
Boltzmann CD-1 negative/positive: 2.29x
Boltzmann CD-5 negative/positive: 8.03x
```

The headline result is the total-cost comparison, not those per-update ratios.

## Legacy single-step diagnostic

`encoder_pair_comparison.py` is kept as a small diagnostic and validation
target. It compares one single-pattern step of `encoder-backprop-8-3-8`
against `encoder-3-parity` and helped expose the second-pass penalty issue.

It is not a canonical v2 comparison because it mixes different tasks and does
not count steps to the reference criterion.

## Second comparison: bars-rbm (CD-1) vs bars (wake-sleep)

`bars_cost_comparison.py` compares CD-1 RBM against the Helmholtz machine on
bar specialization. Reference criterion: 7/8 bars covered (cosine purity >= 0.5).

Note: the two stubs use different data distributions (independent vs hierarchical
bars), so convergence counts are indicative rather than strictly comparable. The
per-step ByteDMD costs are distribution-independent and fully valid.

Run:

```bash
python3 v2-bytedmd/bars_cost_comparison.py
```

Verified on 2026-05-11:

```text
Per single-sample step:
  bars-rbm  CD-1  (W: 16×8 = 128 weights)
    positive phase  (v→h)                  :     2,177
    negative phase  (h→v→h)                :     4,751
    total                                   :     6,928
    2nd-pass penalty (neg/pos)             :      2.18x

  bars wake-sleep  (gen: 161 + rec: 145 = 306 weights)
    wake recognition (v→h→t)              :     2,236
    wake gen + sleep                       :    11,574
    total                                   :    13,810
    wake-sleep / recognition               :      6.18x

Convergence to 7/8 bars covered, 10 seeds:
  bars-rbm   solved : 9/10,  median   30,000 samples
  bars (WS)  solved : 3/10,  median  300,000 samples
  (wake-sleep lr=0.1; lr=0.01 rarely converges within budget)

Total ByteDMD to reference criterion:
  bars-rbm  CD-1     :    207,840,000
  bars wake-sleep    :  4,143,000,000  (19.9x bars-rbm)
```

Wake-sleep costs ~20× more total data movement than CD-1 to reach comparable bar
specialization. The penalty comes from two effects: (1) ~2× higher per-step cost
(more weights, wake+sleep both touch them), and (2) ~10× more samples to converge
(3/10 solve rate vs 9/10, median 10× more samples when solved).

## Next pairs

Per [issue #45](https://github.com/cybertronai/hinton-problems/issues/45),
recommended follow-up pairs remain:

- `shifter` vs `helmholtz-shifter` (Boltzmann vs Helmholtz on the same structure).
- `encoder-4-2-4` (Boltzmann) to close the size gap in earlier encoder diagnostics.

Do not add a new pair until its reference criterion and convergence-count method
are explicit enough to satisfy the contract above.
