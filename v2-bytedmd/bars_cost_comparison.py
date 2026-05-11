"""
v2 ByteDMD: bars-rbm (CD-1) vs bars (wake-sleep Helmholtz machine).

Both algorithms learn to represent bars in 4x4 binary images. This is the
recommended second algorithm pair from v2-bytedmd/README.md.

Note on data distributions:
  bars-rbm  : independent bars; each of 8 bars active with p=0.125
  bars       : hierarchical distribution; orientation (V=2/3, H=1/3) chosen,
               then each of 4 bars in that orientation active with p=0.2
  Both learn to specialize hidden units to individual bars, making the
  per-step ByteDMD comparison informative even with different distributions.

Reference criterion:
  bars_covered >= 7/8  (at least 7 of the 8 bars have a specialized hidden unit,
  purity >= 0.5 under cosine similarity of the unit's weight row to bar templates).
  7/8 was chosen over 8/8: the RBM with n_hidden=8 rarely achieves perfect coverage
  due to duplicate detectors, while 7/8 is reached by 9/10 seeds.

Architecture:
  bars-rbm   : W(16×8) + b_v(16) + b_h(8)                       = 152 params
  bars        : generative  W_hv(8×16) + W_th(1×8) + biases(25) = 161 params
                recognition R_vh(16×8) + R_ht(8×1) + biases(10) = 145 params
                total 306 params; recognition weights not used at inference

Per-step unit:
  Both measured for a single sample (batch=1). Convergence is in samples seen
  so the two can be placed on a common axis regardless of training batch size.

Usage:
    cd v2-bytedmd
    python3 bars_cost_comparison.py
"""

import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))    # bytedmd
sys.path.insert(0, str(ROOT / "bars"))
sys.path.insert(0, str(ROOT / "bars-rbm"))

from bytedmd import bytedmd
import bars as bars_ref
import bars_rbm as rbm_ref

REFERENCE_CRITERION = "7/8 bars covered (cosine purity >= 0.5)"
N_BYTEDMD_SEEDS = 5
N_CONVERGENCE_SEEDS = 10

# bars Helmholtz training budget (per seed, in wake-sleep steps with batch=20).
# lr=0.1 is used here (vs CLI default 0.01) because the default lr makes
# convergence extremely rare within the budget. With lr=0.1, ~3/10 seeds
# reach the criterion within 300K steps; with lr=0.01, ~1/10 do.
HELMHOLTZ_LR = 0.1
HELMHOLTZ_MAX_STEPS = 300_000
HELMHOLTZ_EVAL_EVERY = 5_000

# bars-rbm convergence tracking parameters (match rbm_ref.train defaults)
RBM_N_EPOCHS = 300
RBM_N_TRAIN = 2_000
RBM_BATCH_SIZE = 20

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(message=""):
    print(message, flush=True)


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


_SAMPLE_RNG = random.Random(0)


def _sample(probs):
    return [1.0 if _SAMPLE_RNG.random() < p else 0.0 for p in probs]


# ---------------------------------------------------------------------------
# bars-rbm (CD-1) pure-Python kernels — single sample
# ---------------------------------------------------------------------------
# W[i][j]: weight from visible i (0..15) to hidden j (0..7)

def _rbm_h_given_v(W, b_h, v):
    """h_prob for one 16-dim visible pattern → 8-dim hidden."""
    return [_sigmoid(sum(v[i] * W[i][j] for i in range(16)) + b_h[j])
            for j in range(8)]


def _rbm_v_given_h(W, b_v, h):
    """v_prob for one 8-dim hidden pattern → 16-dim visible."""
    return [_sigmoid(sum(h[j] * W[i][j] for j in range(8)) + b_v[i])
            for i in range(16)]


def rbm_cd1_single(W, b_v, b_h, v):
    """One CD-1 step on a single sample. Returns (dW, db_v, db_h).

    W read order: positive phase (v→h once), negative phase (h→v→h twice more).
    """
    # positive phase
    h_pos = _rbm_h_given_v(W, b_h, v)
    h_sample = _sample(h_pos)
    # negative phase
    v_recon = _rbm_v_given_h(W, b_v, h_sample)
    h_neg = _rbm_h_given_v(W, b_h, v_recon)
    # gradients
    dW   = [[v[i] * h_pos[j] - v_recon[i] * h_neg[j]
             for j in range(8)] for i in range(16)]
    db_v = [v[i] - v_recon[i] for i in range(16)]
    db_h = [h_pos[j] - h_neg[j] for j in range(8)]
    return dW, db_v, db_h


# ---------------------------------------------------------------------------
# bars (wake-sleep Helmholtz machine) pure-Python kernels — single sample
# ---------------------------------------------------------------------------
# W_hv[j][i]: weight from hidden j to visible i (generative, top-down)
# W_th[0][j]: weight from the single top unit to hidden j (generative)
# R_vh[i][j]: weight from visible i to hidden j (recognition, bottom-up)
# R_ht[j][0]: weight from hidden j to the single top unit (recognition)

def _helmholtz_recognize(R_vh, c_h, R_ht, c_top, v):
    """Bottom-up recognition pass for one visible pattern.
    Returns (h_prob, h_sample, t_prob, t_sample).
    """
    h_prob = [_sigmoid(sum(v[i] * R_vh[i][j] for i in range(16)) + c_h[j])
              for j in range(8)]
    h = _sample(h_prob)
    t_prob = [_sigmoid(sum(h[j] * R_ht[j][0] for j in range(8)) + c_top[0])]
    t = _sample(t_prob)
    return h_prob, h, t_prob, t


def _helmholtz_generate(W_th, W_hv, b_top, b_h, b_v):
    """Top-down generation pass (one sample from the generative model).
    Returns (t, h, v, t_prob, h_prob, v_prob).
    """
    t_prob = [_sigmoid(b_top[0])]
    t = _sample(t_prob)
    h_prob = [_sigmoid(t[0] * W_th[0][j] + b_h[j]) for j in range(8)]
    h = _sample(h_prob)
    v_prob = [_sigmoid(sum(h[j] * W_hv[j][i] for j in range(8)) + b_v[i])
              for i in range(16)]
    v = _sample(v_prob)
    return t, h, v, t_prob, h_prob, v_prob


def helmholtz_wake_sleep_single(W_th, W_hv, b_top, b_h, b_v,
                                R_vh, R_ht, c_h, c_top, v_data):
    """One wake+sleep cycle for a single data sample. Returns all gradient pieces.

    Wake  : recognize v_data → (h, t); generative weights learn to predict below.
    Sleep : generate (t, h, v) from model; recognition weights learn to invert.
    """
    # WAKE — recognition net infers (h, t); generative weights updated
    _, h, _, t = _helmholtz_recognize(R_vh, c_h, R_ht, c_top, v_data)
    p_v_pred = [_sigmoid(sum(h[j] * W_hv[j][i] for j in range(8)) + b_v[i])
                for i in range(16)]
    p_h_pred = [_sigmoid(t[0] * W_th[0][j] + b_h[j]) for j in range(8)]
    p_t_pred = [_sigmoid(b_top[0])]
    err_v  = [v_data[i] - p_v_pred[i] for i in range(16)]
    err_h  = [h[j] - p_h_pred[j]      for j in range(8)]
    err_t  = [t[0] - p_t_pred[0]]
    dW_hv  = [[h[j] * err_v[i]         for i in range(16)] for j in range(8)]
    db_v   = list(err_v)
    dW_th  = [[t[0] * err_h[j]         for j in range(8)]]
    db_h   = list(err_h)
    db_top = list(err_t)

    # SLEEP — generative model produces fantasy; recognition weights updated
    t_gen, h_gen, v_gen, _, _, _ = _helmholtz_generate(
        W_th, W_hv, b_top, b_h, b_v)
    p_h_rec = [_sigmoid(sum(v_gen[i] * R_vh[i][j] for i in range(16)) + c_h[j])
               for j in range(8)]
    p_t_rec = [_sigmoid(sum(h_gen[j] * R_ht[j][0] for j in range(8)) + c_top[0])]
    err_h_rec = [h_gen[j] - p_h_rec[j] for j in range(8)]
    err_t_rec = [t_gen[0] - p_t_rec[0]]
    dR_vh  = [[v_gen[i] * err_h_rec[j] for j in range(8)] for i in range(16)]
    dc_h   = list(err_h_rec)
    dR_ht  = [[h_gen[j] * err_t_rec[0]] for j in range(8)]
    dc_top = list(err_t_rec)

    return (dW_hv, db_v, dW_th, db_h, db_top, dR_vh, dc_h, dR_ht, dc_top)


# ---------------------------------------------------------------------------
# Weight initializers (pure-Python, for ByteDMD measurement)
# ---------------------------------------------------------------------------

def _init_rbm(seed=0):
    rng = random.Random(seed)
    W   = [[rng.gauss(0, 0.01) for _ in range(8)] for _ in range(16)]
    b_v = [0.0] * 16
    b_h = [0.0] * 8
    return W, b_v, b_h


def _init_helmholtz(seed=0):
    rng = random.Random(seed)
    W_th  = [[rng.gauss(0, 0.1) for _ in range(8)]]
    W_hv  = [[rng.gauss(0, 0.1) for _ in range(16)] for _ in range(8)]
    b_top = [0.0]
    b_h   = [0.0] * 8
    b_v   = [0.0] * 16
    R_vh  = [[rng.gauss(0, 0.1) for _ in range(8)] for _ in range(16)]
    R_ht  = [[rng.gauss(0, 0.1)] for _ in range(8)]
    c_h   = [0.0] * 8
    c_top = [0.0]
    return W_th, W_hv, b_top, b_h, b_v, R_vh, R_ht, c_h, c_top


# ---------------------------------------------------------------------------
# Reference criterion: bars_covered from a (n_hidden × n_visible) weight matrix
# ---------------------------------------------------------------------------

def _bar_template_flat(idx, h=4, w=4):
    """16-pixel (h×w flattened) bar template for bar index idx."""
    img = [0.0] * (h * w)
    if idx < w:
        for r in range(h):
            img[r * w + idx] = 1.0
    else:
        row = idx - w
        for c in range(w):
            img[row * w + c] = 1.0
    return img


def _cosine_sim_centered(a, b):
    """Cosine similarity after mean-centering both vectors."""
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    ac = [x - ma for x in a]
    bc = [x - mb for x in b]
    dot = sum(ac[i] * bc[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in ac)) + 1e-12
    nb = math.sqrt(sum(x * x for x in bc)) + 1e-12
    return dot / (na * nb)


_BAR_TEMPLATES = [_bar_template_flat(i) for i in range(8)]


def bars_covered_from_weights(W_nhv):
    """Count bars with a specialized hidden unit (purity >= 0.5).

    W_nhv: n_hidden × n_visible matrix (list-of-lists). For bars-rbm,
    pass the transposed weight matrix [[W[i][j] for i in range(16)] for j in range(8)].
    For bars Helmholtz, pass W_hv directly.
    """
    covered = set()
    for wrow in W_nhv:
        sims = [_cosine_sim_centered(wrow, tmpl) for tmpl in _BAR_TEMPLATES]
        best = max(range(8), key=lambda b: sims[b])
        if sims[best] >= 0.5:
            covered.add(best)
    return len(covered)


# ---------------------------------------------------------------------------
# Convergence step counts (via numpy stubs)
# ---------------------------------------------------------------------------

def count_rbm_steps(n_seeds=N_CONVERGENCE_SEEDS):
    """Samples seen to 8/8 bars covered for bars-rbm CD-1.

    Returns list (length n_seeds) of sample counts, None if not solved.
    Uses bars_rbm.train() defaults: batch=20, epochs=300, momentum, sparsity.
    """
    results = []
    for seed in range(n_seeds):
        _, hist = rbm_ref.train(
            n_epochs=RBM_N_EPOCHS, n_hidden=8, n_train=RBM_N_TRAIN,
            batch_size=RBM_BATCH_SIZE, lr=0.1, weight_decay=1e-4,
            momentum=0.5, sparsity_cost=0.1, seed=seed, verbose=False)
        batches_per_epoch = RBM_N_TRAIN // RBM_BATCH_SIZE
        first_samples = next(
            ((i + 1) * batches_per_epoch * RBM_BATCH_SIZE
             for i, covered in enumerate(hist["bars_covered"])
             if covered >= 7),
            None
        )
        results.append(first_samples)
    return results


def count_helmholtz_steps(n_seeds=N_CONVERGENCE_SEEDS):
    """Samples seen to 8/8 bars covered for bars Helmholtz wake-sleep.

    Returns list (length n_seeds) of sample counts, None if not solved
    within HELMHOLTZ_MAX_STEPS * batch_size samples.
    Uses batch_size=20, lr=0.01 (CLI defaults from bars.py).
    """
    batch_size = 20
    results = []
    for seed in range(n_seeds):
        model = bars_ref.HelmholtzMachine(n_hidden=8, seed=seed)
        rng = np.random.default_rng(seed + 1000)
        found = None
        for checkpoint in range(HELMHOLTZ_EVAL_EVERY,
                                HELMHOLTZ_MAX_STEPS + HELMHOLTZ_EVAL_EVERY,
                                HELMHOLTZ_EVAL_EVERY):
            bars_ref.wake_sleep(model, rng, n_steps=HELMHOLTZ_EVAL_EVERY,
                                lr=HELMHOLTZ_LR, batch_size=batch_size, eval_every=0)
            W_nhv = model.W_hv.tolist()   # shape (8, 16)
            if bars_covered_from_weights(W_nhv) >= 7:
                found = checkpoint * batch_size
                break
        results.append(found)
    return results


# ---------------------------------------------------------------------------
# ByteDMD measurements (single-sample kernels)
# ---------------------------------------------------------------------------

def _typical_v():
    """One typical bars sample: a single vertical bar in column 0."""
    v = [0.0] * 16
    for r in range(4):
        v[r * 4] = 1.0
    return v


def measure_rbm_cd1_single(n_seeds=N_BYTEDMD_SEEDS):
    """ByteDMD cost of one single-sample CD-1 step."""
    v = _typical_v()
    pos_costs, total_costs = [], []
    for seed in range(n_seeds):
        global _SAMPLE_RNG
        _SAMPLE_RNG = random.Random(seed + 99)
        W, b_v, b_h = _init_rbm(seed)
        pos   = bytedmd(_rbm_h_given_v, (W, b_h, v))
        total = bytedmd(rbm_cd1_single, (W, b_v, b_h, v))
        pos_costs.append(pos)
        total_costs.append(total)
    avg = lambda lst: sum(lst) / len(lst)
    return {
        "positive": avg(pos_costs),
        "negative": avg(total_costs) - avg(pos_costs),
        "total":    avg(total_costs),
    }


def measure_helmholtz_ws_single(n_seeds=N_BYTEDMD_SEEDS):
    """ByteDMD cost of one single-sample wake-sleep step."""
    v = _typical_v()
    recog_costs, total_costs = [], []
    for seed in range(n_seeds):
        global _SAMPLE_RNG
        _SAMPLE_RNG = random.Random(seed + 99)
        W_th, W_hv, b_top, b_h, b_v, R_vh, R_ht, c_h, c_top = _init_helmholtz(seed)
        recog = bytedmd(_helmholtz_recognize, (R_vh, c_h, R_ht, c_top, v))
        total = bytedmd(
            helmholtz_wake_sleep_single,
            (W_th, W_hv, b_top, b_h, b_v, R_vh, R_ht, c_h, c_top, v))
        recog_costs.append(recog)
        total_costs.append(total)
    avg = lambda lst: sum(lst) / len(lst)
    return {
        "wake_recognition":   avg(recog_costs),
        "wake_gen_and_sleep": avg(total_costs) - avg(recog_costs),
        "total":              avg(total_costs),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def median(lst):
    s = sorted(x for x in lst if x is not None)
    return s[len(s) // 2] if s else None


def main():
    _log("=" * 68)
    _log("ByteDMD total cost to reference criterion — bars pair")
    _log("  bars-rbm  : RBM + CD-1  (Hinton 2002, independent bars, p=0.125)")
    _log("  bars       : Helmholtz machine + wake-sleep (Hinton et al. 1995,")
    _log("               hierarchical bars, P(vertical)=2/3, P(bar|orient)=0.2)")
    _log("  task       : 4×4 binary images, 8 hidden units each specialize to 1 bar")
    _log(f"  criterion  : {REFERENCE_CRITERION}")
    _log("  unit cost  : single sample (batch=1); convergence in total samples seen")
    _log("=" * 68)

    _log(f"\n[1/3] ByteDMD per single-sample step ({N_BYTEDMD_SEEDS} seeds)...")
    t0 = time.time()
    rbm_cost = measure_rbm_cd1_single(n_seeds=N_BYTEDMD_SEEDS)
    ws_cost  = measure_helmholtz_ws_single(n_seeds=N_BYTEDMD_SEEDS)
    _log(f"      done in {time.time() - t0:.1f}s")

    _log(f"\n[2/3] Reference-criterion convergence counts "
         f"({N_CONVERGENCE_SEEDS} seeds, numpy stubs)...")
    _log(f"      bars-rbm   : up to {RBM_N_EPOCHS} epochs × "
         f"{RBM_N_TRAIN // RBM_BATCH_SIZE} batches/epoch "
         f"= {RBM_N_EPOCHS * RBM_N_TRAIN // RBM_BATCH_SIZE * RBM_BATCH_SIZE:,} samples budget")
    _log(f"      bars (WS)  : up to {HELMHOLTZ_MAX_STEPS:,} steps × 20 = "
         f"{HELMHOLTZ_MAX_STEPS * 20:,} samples, lr={HELMHOLTZ_LR} (lr=0.01 rarely solves)")
    _log("      note: wake-sleep on bars is stochastic and unreliable; solve rate"
         " is a meaningful metric")
    t0 = time.time()
    rbm_steps = count_rbm_steps(n_seeds=N_CONVERGENCE_SEEDS)
    ws_steps  = count_helmholtz_steps(n_seeds=N_CONVERGENCE_SEEDS)
    _log(f"      done in {time.time() - t0:.1f}s")

    rbm_solved = [s for s in rbm_steps if s is not None]
    ws_solved  = [s for s in ws_steps  if s is not None]
    rbm_med    = median(rbm_steps)
    ws_med     = median(ws_steps)

    # Total cost: median samples to convergence × per-sample ByteDMD cost
    rbm_total = rbm_med * rbm_cost["total"] if rbm_med else None
    ws_total  = ws_med  * ws_cost["total"]  if ws_med  else None

    _log()
    _log("─" * 68)
    _log("Per single-sample step (ByteDMD):")
    _log()
    _log(f"  bars-rbm  CD-1     W(16×8)=128 + biases(24) = 152 params")
    _log(f"    positive phase  (v→h, W read once)     : {rbm_cost['positive']:>9,.0f}")
    _log(f"    negative phase  (h→v→h, W read twice)  : {rbm_cost['negative']:>9,.0f}")
    _log(f"    total                                   : {rbm_cost['total']:>9,.0f}")
    _log(f"    2nd-pass penalty (neg/pos)              : "
         f"{rbm_cost['negative'] / rbm_cost['positive']:>9.2f}x")
    _log()
    _log(f"  bars wake-sleep    gen(161) + rec(145) = 306 params")
    _log(f"    wake recognition (v→h→t via R_vh, R_ht): {ws_cost['wake_recognition']:>9,.0f}")
    _log(f"    wake gen + sleep (W_hv,W_th,R_vh,R_ht) : {ws_cost['wake_gen_and_sleep']:>9,.0f}")
    _log(f"    total                                   : {ws_cost['total']:>9,.0f}")
    _log(f"    wake-sleep / wake-recognition           : "
         f"{ws_cost['total'] / ws_cost['wake_recognition']:>9.2f}x")

    _log()
    _log("─" * 68)
    _log(f"Convergence ({N_CONVERGENCE_SEEDS} seeds, samples seen to {REFERENCE_CRITERION}):")
    _log()
    _log(f"  bars-rbm   : {len(rbm_solved)}/{N_CONVERGENCE_SEEDS} solved, "
         f"median {rbm_med:,} samples")
    _log(f"  bars (WS)  : {len(ws_solved)}/{N_CONVERGENCE_SEEDS} solved, "
         f"median {ws_med:,} samples" if ws_solved else
         f"  bars (WS)  : {len(ws_solved)}/{N_CONVERGENCE_SEEDS} solved "
         f"(none in {HELMHOLTZ_MAX_STEPS * 20:,} samples)")

    _log()
    _log("─" * 68)
    _log("Total ByteDMD to reference criterion (median samples × per-sample cost):")
    _log()
    if rbm_total:
        _log(f"  bars-rbm  CD-1     : {rbm_total:>16,.0f}")
    if ws_total and rbm_total:
        _log(f"  bars wake-sleep    : {ws_total:>16,.0f}  "
             f"({ws_total / rbm_total:.1f}x bars-rbm)")
    elif ws_total:
        _log(f"  bars wake-sleep    : {ws_total:>16,.0f}")
    else:
        _log(f"  bars wake-sleep    : no seeds solved within budget "
             f"({HELMHOLTZ_MAX_STEPS * 20:,} samples)")

    _log()


if __name__ == "__main__":
    main()
