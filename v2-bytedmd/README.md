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

## Next pairs

Per [issue #45](https://github.com/cybertronai/hinton-problems/issues/45),
recommended follow-up pairs remain:

- `bars` vs `bars-rbm` (wake-sleep vs CD-1 on the same data).
- `shifter` vs `helmholtz-shifter` (Boltzmann vs Helmholtz on the same structure).
- `encoder-4-2-4` (Boltzmann) to close the size gap in earlier encoder diagnostics.

Do not add a new pair until its reference criterion and convergence-count method
are explicit enough to satisfy the contract above.
