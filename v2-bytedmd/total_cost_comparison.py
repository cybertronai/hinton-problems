"""
v2 ByteDMD canonical measurement contract: total cost to reference criterion.

Yaroslav's rule for ByteDMD/DALI comparisons is: compare total data movement
to reach the agreed reference accuracy/solve criterion, not isolated per-step
cost. Per-step ByteDMD is still measured, but only as the unit cost in:

    total_cost = steps_to_reference_criterion * bytedmd_per_step

This first canonical example compares backprop vs Boltzmann on the same
8-3-8 encoder problem. It addresses two gaps in encoder_pair_comparison.py
(raised in PR #50 review):

  1. Single-pattern measurement underestimates backprop's activation
     re-fetch cost. Fixed here by measuring the full batch (all 8 patterns
     at once), so all forward activations accumulate on the LRU stack before
     the backward pass re-reads W2.

  2. Per-step cost is not comparable across algorithms that need different
     numbers of steps. Fixed here by measuring total ByteDMD to reach the
     same reference criterion: 100% reconstruction accuracy and 8/8 distinct
     hidden codes.

     Algorithm pair is now encoder-backprop-8-3-8 vs encoder-8-3-8 (RBM),
     which solve the *same* problem (8 one-hot patterns through a 3-bit
     bottleneck), making the convergence comparison valid.

     Note: encoder-3-parity from the previous script solves a *different*
     problem (3-bit parity), so that per-step comparison was not comparable
     on convergence grounds either.

Architectures (both encode the same 8-pattern set):
  backprop-8-3-8   : W1 (8x3) + W2 (3x8) = 48 weight values
  encoder-8-3-8    : W (16x3)             = 48 weight values  ← same count
    16 visible = 8 input (V1) + 8 output (V2), 3 hidden

Usage:
    cd v2-bytedmd
    python3 total_cost_comparison.py
"""

import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))   # bytedmd
sys.path.insert(0, str(ROOT / "encoder-backprop-8-3-8"))
sys.path.insert(0, str(ROOT / "encoder-8-3-8"))

from bytedmd import bytedmd
import encoder_backprop_8_3_8 as bp_ref
import encoder_8_3_8 as rbm_ref

REFERENCE_CRITERION = "100% reconstruction accuracy and 8/8 distinct hidden codes"
N_BYTE_DMD_SEEDS = 5
N_CONVERGENCE_SEEDS = 10
RBM_CD_UPDATES_PER_EPOCH = 16

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(message=""):
    print(message, flush=True)


def _is_reference_solved(acc, n_codes):
    return acc >= 1.0 and n_codes == 8


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))

_SAMPLE_RNG = random.Random(0)

def _sample(probs):
    return [1.0 if _SAMPLE_RNG.random() < p else 0.0 for p in probs]

# ---------------------------------------------------------------------------
# Backprop 8-3-8 — full-batch kernels
# ---------------------------------------------------------------------------
# patterns: list of 8 lists of length 8 (one-hot)
# W1[i][j], W2[j][k] as before

def bp_fullbatch_forward(W1, b1, W2, b2, patterns):
    """Forward pass for all 8 patterns. Activations (8*3 h + 8*8 y = 88 values)
    accumulate on the LRU stack before backward begins."""
    hs, ys = [], []
    for x in patterns:
        h = [_sigmoid(sum(x[i] * W1[i][j] for i in range(8)) + b1[j])
             for j in range(3)]
        y = [_sigmoid(sum(h[j] * W2[j][k] for j in range(3)) + b2[k])
             for k in range(8)]
        hs.append(h)
        ys.append(y)
    return hs, ys


def bp_fullbatch_step(W1, b1, W2, b2, patterns):
    """Full-batch forward + backward for all 8 patterns.

    After the forward loop, all 88 activation values (h: 8x3, y: 8x8) are
    on the LRU stack. W2 (24 values) is now displaced by 88 activations when
    the backward pass re-reads it — much deeper than the single-pattern case.
    """
    # Forward — all activations land on stack
    hs, ys = [], []
    for x in patterns:
        h = [_sigmoid(sum(x[i] * W1[i][j] for i in range(8)) + b1[j])
             for j in range(3)]
        y = [_sigmoid(sum(h[j] * W2[j][k] for j in range(3)) + b2[k])
             for k in range(8)]
        hs.append(h)
        ys.append(y)
    # Backward — W2 re-read with 88 activations above it on the stack
    dW1 = [[0.0] * 3 for _ in range(8)]
    dW2 = [[0.0] * 8 for _ in range(3)]
    db1 = [0.0] * 3
    db2 = [0.0] * 8
    for idx, x in enumerate(patterns):
        h, y = hs[idx], ys[idx]
        delta_out = [y[k] - x[k] for k in range(8)]
        delta_h = [sum(delta_out[k] * W2[j][k] for k in range(8)) * h[j] * (1.0 - h[j])
                   for j in range(3)]
        for i in range(8):
            for j in range(3):
                dW1[i][j] += x[i] * delta_h[j]
        for j in range(3):
            for k in range(8):
                dW2[j][k] += h[j] * delta_out[k]
        for j in range(3):
            db1[j] += delta_h[j]
        for k in range(8):
            db2[k] += delta_out[k]
    return dW1, db1, dW2, db2


# ---------------------------------------------------------------------------
# Boltzmann encoder-8-3-8 — full-batch CD kernels
# ---------------------------------------------------------------------------
# W[i][j]: weight from visible i (0..15) to hidden j (0..2)
# One CD-k step on all 8 patterns.

def _rbm_h_given_v_16(W, b_h, v):
    """h_prob for one 16-bit visible pattern: sigmoid(v @ W + b_h)."""
    return [_sigmoid(sum(v[i] * W[i][j] for i in range(16)) + b_h[j])
            for j in range(3)]


def _rbm_v_given_h_16(W, b_v, h):
    """v_prob for one 3-bit hidden pattern: sigmoid(h @ W.T + b_v)."""
    return [_sigmoid(sum(h[j] * W[i][j] for j in range(3)) + b_v[i])
            for i in range(16)]


def rbm_fullbatch_step(W, b_v, b_h, patterns, k=1):
    """Full-batch CD-k on all 8 patterns.

    Positive phase reads W once per pattern (8 reads total).
    Negative phase reads W k times (v given h) + k times (h given v) per
    pattern after being displaced by positive-phase activations.
    Total W reads: (1 + 2k) * 8 patterns.
    """
    # Positive phase — W read once per pattern
    h_probs_pos = [_rbm_h_given_v_16(W, b_h, v) for v in patterns]
    h_samples   = [_sample(hp) for hp in h_probs_pos]

    # Negative CD chain — W pushed progressively deeper each step
    v_negs     = [v[:] for v in patterns]
    h_negs     = list(h_samples)
    h_probs_neg = list(h_probs_pos)
    for _ in range(k):
        v_prob_negs = [_rbm_v_given_h_16(W, b_v, h) for h in h_negs]
        v_negs      = [_sample(vp) for vp in v_prob_negs]
        h_probs_neg = [_rbm_h_given_v_16(W, b_h, v) for v in v_negs]
        h_negs      = [_sample(hp) for hp in h_probs_neg]

    # Gradients
    n = len(patterns)
    dW   = [[sum(patterns[b][i] * h_probs_pos[b][j] - v_negs[b][i] * h_probs_neg[b][j]
                 for b in range(n)) / n
             for j in range(3)] for i in range(16)]
    db_v = [sum(patterns[b][i] - v_negs[b][i] for b in range(n)) / n
            for i in range(16)]
    db_h = [sum(h_probs_pos[b][j] - h_probs_neg[b][j] for b in range(n)) / n
            for j in range(3)]
    return dW, db_v, db_h


def rbm_fullbatch_positive(W, b_v, b_h, patterns, k=1):
    """Positive phase only (consistent args with full step for decomposition)."""
    return [_rbm_h_given_v_16(W, b_h, v) for v in patterns]


# ---------------------------------------------------------------------------
# Weight initializers
# ---------------------------------------------------------------------------

def _init_bp(seed=0):
    rng = random.Random(seed)
    W1 = [[rng.uniform(-0.1, 0.1) for _ in range(3)] for _ in range(8)]
    b1 = [0.0] * 3
    W2 = [[rng.uniform(-0.1, 0.1) for _ in range(8)] for _ in range(3)]
    b2 = [0.0] * 8
    return W1, b1, W2, b2


def _init_rbm16(seed=0):
    rng = random.Random(seed)
    W   = [[rng.gauss(0, 0.1) for _ in range(3)] for _ in range(16)]
    b_v = [0.0] * 16
    b_h = [0.0] * 3
    return W, b_v, b_h


# ---------------------------------------------------------------------------
# Convergence step counts (via numpy stubs)
# ---------------------------------------------------------------------------

def count_bp_steps(n_seeds=10):
    """Return full-batch epochs to reference criterion, None if failed."""
    results = []
    for seed in range(n_seeds):
        _, hist = bp_ref.train(n_epochs=5000, seed=seed, verbose=False)
        solved = _is_reference_solved(
            hist['acc'][-1], hist['n_distinct_codes'][-1])
        results.append(len(hist['epoch']) if solved else None)
    return results


def count_rbm_steps(n_seeds=10):
    """Return CD updates to reference criterion for Boltzmann, None if failed.

    Each epoch in the NumPy reference does batch_repeats=16 CD updates on the
    8-pattern batch, so first matching epoch maps to epoch * 16 CD updates.
    """
    results = []
    for seed in range(n_seeds):
        _, hist = rbm_ref.train(n_epochs=4000, seed=seed, verbose=False)
        first = next(
            (i + 1 for i, (acc, n_codes) in enumerate(
                zip(hist['acc'], hist['n_distinct_codes']))
             if _is_reference_solved(acc, n_codes)),
            None,
        )
        results.append(first * RBM_CD_UPDATES_PER_EPOCH
                       if first is not None else None)
    return results


# ---------------------------------------------------------------------------
# ByteDMD measurements
# ---------------------------------------------------------------------------

def measure_bp_fullbatch(n_seeds=5):
    bp_patterns = [[1.0 if i == p else 0.0 for i in range(8)] for p in range(8)]
    costs = []
    for seed in range(n_seeds):
        global _SAMPLE_RNG
        _SAMPLE_RNG = random.Random(seed + 99)
        W1, b1, W2, b2 = _init_bp(seed)
        fwd  = bytedmd(bp_fullbatch_forward, (W1, b1, W2, b2, bp_patterns))
        full = bytedmd(bp_fullbatch_step,    (W1, b1, W2, b2, bp_patterns))
        costs.append({"forward": fwd, "backward": full - fwd, "total": full})
    return {k: sum(c[k] for c in costs) / len(costs) for k in costs[0]}


def measure_rbm_fullbatch(k=1, n_seeds=5):
    # 8 patterns: each pattern has V1[i]=V2[i]=1, rest 0, length 16
    rbm_patterns = []
    for p in range(8):
        v = [0.0] * 16
        v[p] = 1.0
        v[8 + p] = 1.0
        rbm_patterns.append(v)

    costs = []
    for seed in range(n_seeds):
        global _SAMPLE_RNG
        _SAMPLE_RNG = random.Random(seed + 99)
        W, b_v, b_h = _init_rbm16(seed)
        pos  = bytedmd(rbm_fullbatch_positive, (W, b_v, b_h, rbm_patterns, k))
        full = bytedmd(rbm_fullbatch_step,     (W, b_v, b_h, rbm_patterns, k))
        costs.append({"positive": pos, "negative": full - pos, "total": full})
    return {k_: sum(c[k_] for c in costs) / len(costs) for k_ in costs[0]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def median(lst):
    s = sorted(x for x in lst if x is not None)
    return s[len(s) // 2] if s else None


def main():
    _log("=" * 66)
    _log("ByteDMD total cost to reference criterion — canonical v2 example")
    _log("  encoder-backprop-8-3-8  vs  encoder-8-3-8 (RBM CD-k)")
    _log("  both encode 8 one-hot patterns through a 3-bit bottleneck")
    _log(f"  reference criterion: {REFERENCE_CRITERION}")
    _log("=" * 66)

    _log(f"\n[1/3] ByteDMD per full-batch step "
         f"(all 8 patterns, {N_BYTE_DMD_SEEDS} seeds)...")
    t0 = time.time()
    bp_cost  = measure_bp_fullbatch(n_seeds=N_BYTE_DMD_SEEDS)
    rbm1_cost = measure_rbm_fullbatch(k=1, n_seeds=N_BYTE_DMD_SEEDS)
    rbm5_cost = measure_rbm_fullbatch(k=5, n_seeds=N_BYTE_DMD_SEEDS)
    _log(f"      done in {time.time()-t0:.1f}s")

    _log(f"\n[2/3] Reference-criterion step counts "
         f"({N_CONVERGENCE_SEEDS} seeds each, running numpy stubs)...")
    t0 = time.time()
    bp_steps  = count_bp_steps(n_seeds=N_CONVERGENCE_SEEDS)
    rbm_steps = count_rbm_steps(n_seeds=N_CONVERGENCE_SEEDS)
    _log(f"      done in {time.time()-t0:.1f}s")

    bp_solved  = [s for s in bp_steps  if s is not None]
    rbm_solved = [s for s in rbm_steps if s is not None]

    bp_med   = median(bp_steps)
    rbm_med  = median(rbm_steps)

    # Total ByteDMD: backprop uses 1 full-batch step per epoch
    # Boltzmann: each "CD update" counted above is already one full-batch step
    bp_total_med  = bp_med  * bp_cost["total"]  if bp_med  else None
    # CD-5 is the actual training config; CD-1 shown for reference
    rbm_total_med_cd1 = rbm_med * rbm1_cost["total"] if rbm_med else None
    rbm_total_med_cd5 = rbm_med * rbm5_cost["total"] if rbm_med else None

    _log()
    _log("─" * 66)
    _log("Per full-batch step (diagnostic unit cost, not the headline):")
    _log()
    _log(f"  Backprop 8-3-8   (W1: 8x3, W2: 3x8 = 48 weights)")
    _log(f"    forward  (W1 once, W2 once per pattern)  : {bp_cost['forward']:>9,.0f}")
    _log(f"    backward (W2 re-read, 88 activations deep): {bp_cost['backward']:>9,.0f}")
    _log(f"    total                                      : {bp_cost['total']:>9,.0f}")
    _log(f"    second-pass penalty                        : {bp_cost['backward']/bp_cost['forward']:>9.2f}x")
    _log()
    _log(f"  Boltzmann CD-1   (W: 16x3 = 48 weights)")
    _log(f"    positive phase                             : {rbm1_cost['positive']:>9,.0f}")
    _log(f"    negative phase (W read 2x more)           : {rbm1_cost['negative']:>9,.0f}")
    _log(f"    total                                      : {rbm1_cost['total']:>9,.0f}")
    _log(f"    second-pass penalty                        : {rbm1_cost['negative']/rbm1_cost['positive']:>9.2f}x")
    _log()
    _log(f"  Boltzmann CD-5   (actual training config)")
    _log(f"    positive phase                             : {rbm5_cost['positive']:>9,.0f}")
    _log(f"    negative phase (W read 10x more)          : {rbm5_cost['negative']:>9,.0f}")
    _log(f"    total                                      : {rbm5_cost['total']:>9,.0f}")
    _log(f"    second-pass penalty                        : {rbm5_cost['negative']/rbm5_cost['positive']:>9.2f}x")

    _log()
    _log("─" * 66)
    _log(f"Steps to reference criterion ({N_CONVERGENCE_SEEDS} seeds):")
    _log()
    _log(f"  Backprop:  {len(bp_solved)}/{N_CONVERGENCE_SEEDS} solved,  "
         f"median {bp_med} epochs  (= {bp_med} full-batch steps)")
    _log(f"  Boltzmann: {len(rbm_solved)}/{N_CONVERGENCE_SEEDS} solved,  "
         f"median {rbm_med} CD updates  (each = 1 full-batch CD step)")

    _log()
    _log("─" * 66)
    _log("Total ByteDMD to reference criterion (median steps x per-step cost):")
    _log()
    if bp_total_med:
        _log(f"  Backprop              : {bp_total_med:>14,.0f}")
    if rbm_total_med_cd1:
        _log(f"  Boltzmann CD-1        : {rbm_total_med_cd1:>14,.0f}  "
             f"({rbm_total_med_cd1/bp_total_med:.1f}x backprop)")
    if rbm_total_med_cd5:
        _log(f"  Boltzmann CD-5        : {rbm_total_med_cd5:>14,.0f}  "
             f"({rbm_total_med_cd5/bp_total_med:.1f}x backprop)")

    _log()
    _log("─" * 66)
    _log("Full-batch second-pass penalty (updated from PR #50):")
    bp_ratio   = bp_cost['backward'] / bp_cost['forward']
    rbm1_ratio = rbm1_cost['negative'] / rbm1_cost['positive']
    _log(f"  Backprop  backward/forward    : {bp_ratio:.2f}x  "
         f"(was 1.27x single-pattern)")
    _log(f"  Boltzmann CD-1 neg/pos        : {rbm1_ratio:.2f}x  "
         f"(was 2.11x, single-pattern, encoder-3-parity)")
    _log()
    _log("Note: the 88 forward activations (8 patterns x 11 values each)")
    _log("now sit between W2 and the backward pass, increasing the penalty.")


if __name__ == "__main__":
    main()
