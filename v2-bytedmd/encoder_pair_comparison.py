"""
v2 ByteDMD instrumentation — encoder-3-parity vs encoder-backprop-8-3-8

Measures data-movement cost (ByteDMD) of one training step per algorithm
and compares second-pass penalty: how much more expensive does a weight
re-read become after the first pass fills the LRU stack with activations?

Backprop 8-3-8:
  - forward: reads W1 (8x3=24 values) and W2 (3x8=24 values)
  - backward: re-reads W2 only (for delta_h via error backprop); W1 is NOT
    re-read — dW1 = x.T @ delta_h needs only x and delta_h, both shallow.
  - Second-pass penalty = cost of backward / cost of forward (W2 deeper now)

Boltzmann CD-1 (encoder-3-parity):
  - positive phase: reads W (3x4=12 values)
  - negative phase: reads W twice more (for v_neg prob and h_prob_neg)
  - Second-pass penalty = cost of negative / cost of positive

Decomposition method: measure positive/forward phase independently with the
same argument signature as the full step, so stack depths are identical.
The negative/backward cost is then (full_cost - phase1_cost).

Architectures:
  backprop-8-3-8   : W1 (8x3) + W2 (3x8) = 48 weight values, 11 biases
  encoder-3-parity : W  (3x4) = 12 weight values, 7 biases (19 params total)

Usage:
    cd v2-bytedmd
    python3 encoder_pair_comparison.py
"""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bytedmd import bytedmd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


# ---------------------------------------------------------------------------
# Backprop 8-3-8
# ---------------------------------------------------------------------------
# W1[i][j]: weight from input i (0..7) to hidden j (0..2)
# W2[j][k]: weight from hidden j (0..2) to output k (0..7)

def bp_forward_phase(W1, b1, W2, b2, x, target):
    """Forward pass only. target is on the stack but not read — kept so that
    the stack layout matches bp_full_step exactly for a valid decomposition."""
    h = [_sigmoid(sum(x[i] * W1[i][j] for i in range(8)) + b1[j])
         for j in range(3)]
    y = [_sigmoid(sum(h[j] * W2[j][k] for j in range(3)) + b2[k])
         for k in range(8)]
    return h, y


def bp_full_step(W1, b1, W2, b2, x, target):
    """Full forward + backward for one pattern.

    Note on weight reads:
      - W1 is read once (forward, computing h). Not re-read in backward.
      - W2 is read twice: forward (computing y) then backward (computing
        delta_h via error backprop). W2 is deeper on the LRU stack the
        second time, displaced by h (3 values) and y (8 values).
      - dW1 = x . delta_h   -- only needs x and delta_h (both shallow)
      - dW2 = h . delta_out  -- only needs h and delta_out (both shallow)
    """
    # Forward
    h = [_sigmoid(sum(x[i] * W1[i][j] for i in range(8)) + b1[j])
         for j in range(3)]
    y = [_sigmoid(sum(h[j] * W2[j][k] for j in range(3)) + b2[k])
         for k in range(8)]
    # Backward
    delta_out = [y[k] - target[k] for k in range(8)]
    # W2 re-read here — it has been displaced by h (3) and y (8) = 11 new values
    delta_h = [sum(delta_out[k] * W2[j][k] for k in range(8)) * h[j] * (1.0 - h[j])
               for j in range(3)]
    dW2 = [[h[j] * delta_out[k] for k in range(8)] for j in range(3)]
    dW1 = [[x[i] * delta_h[j] for j in range(3)] for i in range(8)]
    db2 = list(delta_out)
    db1 = list(delta_h)
    return dW1, db1, dW2, db2


# ---------------------------------------------------------------------------
# Boltzmann / RBM — encoder-3-parity
# ---------------------------------------------------------------------------
# W[i][j]: weight from visible i (0..2) to hidden j (0..3)
# CD-1: positive → sample h → negative v → sample v_neg → negative h → gradient

_SAMPLE_RNG = random.Random(0)  # seeded per measurement

def _sample(probs):
    return [1.0 if _SAMPLE_RNG.random() < p else 0.0 for p in probs]


def cd_positive_phase(W, b_v, b_h, v):
    """Positive phase only. b_v is on stack but not read — present so the
    stack layout matches cd_full_step exactly."""
    return [_sigmoid(sum(v[i] * W[i][j] for i in range(3)) + b_h[j])
            for j in range(4)]


def cd_full_step(W, b_v, b_h, v):
    """Full CD-1 step for one visible pattern.

    Note on weight reads:
      - Positive phase: W read once (computing h_prob_pos).
      - Negative v:     W read again (computing v_prob_neg via W.T).
        By now W is displaced by h_prob_pos (4) and h_pos (4) = 8 new values.
      - Negative h:     W read a third time (computing h_prob_neg).
        W further displaced by v_prob_neg (3) and v_neg (3) = 6 more values.
    Total: W read 3x. No backward pass; all reads are within the Gibbs chain.
    """
    # Positive phase
    h_prob_pos = [_sigmoid(sum(v[i] * W[i][j] for i in range(3)) + b_h[j])
                  for j in range(4)]
    h_pos = _sample(h_prob_pos)
    # Negative phase — W displaced by h_prob_pos (4) + h_pos (4 untracked)
    v_prob_neg = [_sigmoid(sum(h_pos[j] * W[i][j] for j in range(4)) + b_v[i])
                  for i in range(3)]
    v_neg = _sample(v_prob_neg)
    # Negative h — W displaced further by v_prob_neg (3) + v_neg (3 untracked)
    h_prob_neg = [_sigmoid(sum(v_neg[i] * W[i][j] for i in range(3)) + b_h[j])
                  for j in range(4)]
    # Gradients
    dW   = [[v[i] * h_prob_pos[j] - v_neg[i] * h_prob_neg[j]
             for j in range(4)] for i in range(3)]
    db_v = [v[i] - v_neg[i] for i in range(3)]
    db_h = [h_prob_pos[j] - h_prob_neg[j] for j in range(4)]
    return dW, db_v, db_h


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


def _init_rbm(seed=0):
    rng = random.Random(seed)
    W    = [[rng.gauss(0, 0.1) for _ in range(4)] for _ in range(3)]
    b_v  = [0.0] * 3
    b_h  = [0.0] * 4
    return W, b_v, b_h


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def measure_backprop(seed=0):
    W1, b1, W2, b2 = _init_bp(seed)
    x      = [1.0] + [0.0] * 7   # one-hot pattern 0
    target = x[:]

    fwd_cost  = bytedmd(bp_forward_phase, (W1, b1, W2, b2, x, target))
    full_cost = bytedmd(bp_full_step,     (W1, b1, W2, b2, x, target))
    return {
        "forward":  fwd_cost,
        "backward": full_cost - fwd_cost,
        "total":    full_cost,
        "n_weights": 48,
    }


def measure_boltzmann(seed=0):
    global _SAMPLE_RNG
    _SAMPLE_RNG = random.Random(seed + 99)

    W, b_v, b_h = _init_rbm(seed)
    v = [0.0, 1.0, 1.0]   # even-parity pattern 011

    pos_cost  = bytedmd(cd_positive_phase, (W, b_v, b_h, v))
    full_cost = bytedmd(cd_full_step,      (W, b_v, b_h, v))
    return {
        "positive_phase": pos_cost,
        "negative_phase": full_cost - pos_cost,
        "total":          full_cost,
        "n_weights":      12,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main():
    N_SEEDS = 5
    bp_runs  = [measure_backprop(s)  for s in range(N_SEEDS)]
    rbm_runs = [measure_boltzmann(s) for s in range(N_SEEDS)]

    def avg(runs, key):
        return sum(r[key] for r in runs) / len(runs)

    bp  = {k: avg(bp_runs,  k) for k in bp_runs[0]}
    rbm = {k: avg(rbm_runs, k) for k in rbm_runs[0]}

    print("=" * 62)
    print("ByteDMD encoder pair — v2 instrumentation")
    print("  encoder-backprop-8-3-8  vs  encoder-3-parity (RBM CD-1)")
    print(f"  averaged over {N_SEEDS} random weight seeds, single pattern")
    print("=" * 62)

    bp_ratio  = bp["backward"]  / bp["forward"]
    rbm_ratio = rbm["negative_phase"] / rbm["positive_phase"]

    print()
    print(f"Backprop 8-3-8  ({int(bp['n_weights'])} weight values)")
    print(f"  forward  (reads W1 once, W2 once)         : {bp['forward']:>7.0f}")
    print(f"  backward (re-reads W2 once for delta_h)   : {bp['backward']:>7.0f}")
    print(f"  total                                      : {bp['total']:>7.0f}")
    print(f"  cost/weight                                : {bp['total']/bp['n_weights']:>7.1f}")
    print(f"  second-pass penalty (bwd/fwd)              : {bp_ratio:>7.2f}x")

    print()
    print(f"Boltzmann CD-1  ({int(rbm['n_weights'])} weight values)")
    print(f"  positive phase (reads W once)             : {rbm['positive_phase']:>7.0f}")
    print(f"  negative phase (reads W twice more)       : {rbm['negative_phase']:>7.0f}")
    print(f"  total                                      : {rbm['total']:>7.0f}")
    print(f"  cost/weight                                : {rbm['total']/rbm['n_weights']:>7.1f}")
    print(f"  second-pass penalty (neg/pos)              : {rbm_ratio:>7.2f}x")

    print()
    print("─" * 62)
    print("Finding:")
    if bp_ratio < rbm_ratio:
        print(f"  Backprop has LOWER second-pass penalty ({bp_ratio:.2f}x vs {rbm_ratio:.2f}x).")
        print(f"  W2 is re-read in backward (displaced by 11 forward activations: h=3, y=8)")
        print(f"  but only W2 is re-read — W1 is never accessed in the backward pass.")
        print(f"  Boltzmann CD reads its smaller W matrix three times (pos + two neg),")
        print(f"  and each re-read finds W relatively deeper (12 weights, ~8 intermediates).")
    else:
        print(f"  Boltzmann has LOWER second-pass penalty ({rbm_ratio:.2f}x vs {bp_ratio:.2f}x).")
    print()
    print(f"  Per-weight cost: backprop {bp['total']/bp['n_weights']:.1f}  vs  boltzmann {rbm['total']/rbm['n_weights']:.1f}")
    print(f"  (note: networks are different sizes — not a direct comparison)")


if __name__ == "__main__":
    main()
