# v2 ByteDMD instrumentation

ByteDMD cost measurements for selected algorithm pairs from the hinton-problems v1 baselines. Companion to [issue #45](https://github.com/cybertronai/hinton-problems/issues/45).

ByteDMD measures data-movement cost as the sum of ceil(sqrt(reuse_distance)) over all memory reads, modelling the energy cost of fetching data through a 2D-laid-out LRU cache hierarchy. See [cybertronai/ByteDMD](https://github.com/cybertronai/ByteDMD) for the full spec.

Because ByteDMD traces Python-level operations, all kernels here are pure Python (nested lists, no numpy). The stubs in the parent folders use numpy and cannot be directly wrapped.

## Encoder pair — backprop vs Boltzmann

Use the repo dev shell for scripts that import the NumPy reference stubs:

```bash
nix develop -c python v2-bytedmd/validate_implementations.py
nix develop -c python v2-bytedmd/total_cost_comparison.py
```

`encoder_pair_comparison.py` — compares one training step of `encoder-backprop-8-3-8` (MLP + SGD) against `encoder-3-parity` (RBM + CD-1). Both architectures solve the encoder bottleneck; the question is which pays less to move weights through the memory hierarchy.

### Results (single pattern, 5 random weight seeds)

```
Backprop 8-3-8  (48 weight values)
  forward  (reads W1 once, W2 once)         :     541
  backward (re-reads W2 once for delta_h)   :     685
  total                                      :   1,226
  cost/weight                                :    25.5
  second-pass penalty (bwd/fwd)             :    1.27x

Boltzmann CD-1  (12 weight values)
  positive phase (reads W once)             :     142
  negative phase (reads W twice more)       :     299
  total                                      :     441
  cost/weight                                :    36.8
  second-pass penalty (neg/pos)             :    2.11x
```

### Finding

Backprop has a **lower second-pass penalty** (1.27x vs 2.11x). This is counter to the naive "backprop refetches all activations" hypothesis and has a structural explanation:

- In the backward pass, **W1 is never re-read** — `dW1 = x · delta_h` only needs `x` and `delta_h`, both recently created (shallow). Only W2 is re-read once (for `delta_h` via `delta_out · W2`), and it has been displaced by just 11 forward activations (h=3, y=8).
- In CD-1, W is re-read **twice** in the negative phase (for v_neg and h_prob_neg), against a matrix of only 12 values. The ~8 intermediate values created during the positive phase (h_prob_pos=4, comparison results=4) represent ~67% relative displacement of W, vs ~23% for W2 in backprop (11 intermediates / 24 W2 values).

The "commute" that matters is not the absolute displacement but the **relative** one: a smaller weight matrix gets pushed relatively deeper by the same volume of intermediate activations.

Per-weight cost (25.5 vs 36.8) also favours backprop, though the two architectures differ in size so this is not a direct comparison.

### How to run

```bash
cd v2-bytedmd
python3 encoder_pair_comparison.py
```

No dependencies beyond the Python standard library. `bytedmd.py` is vendored from [cybertronai/ByteDMD](https://github.com/cybertronai/ByteDMD).

## Next pairs (open)

Per [issue #45](https://github.com/cybertronai/hinton-problems/issues/45), the recommended follow-up pairs are:

- `bars` vs `bars-rbm` (wake-sleep vs CD-1 on the same data)
- `shifter` vs `helmholtz-shifter` (Boltzmann vs Helmholtz on the same structure)
- `encoder-4-2-4` (Boltzmann) to close the size gap between the two encoders above
