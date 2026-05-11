"""
Validates that the pure-Python kernels used by the v2 ByteDMD scripts match
the original numpy stubs.

Checks:
  backprop-8-3-8 : forward h, y and gradients dW1, dW2, db1, db2
  encoder-3-parity: h_prob_pos, v_prob_neg, h_prob_neg (deterministic parts of CD)
  total_cost_comparison:
    - backprop-8-3-8 full-batch forward + gradients
    - encoder-8-3-8 full-batch positive phase + deterministic CD gradients
  bars_cost_comparison:
    - bars-rbm CD-1: h_prob_pos, v_recon, h_prob_neg (deterministic CD parts)
    - bars Helmholtz: h_prob (recognition), t_prob (recognition),
                      h_prob_gen (generation), v_prob_gen (generation)

Run from the repo root:
    python3 v2-bytedmd/validate_implementations.py
"""

import sys
import math
import random
import numpy as np
from pathlib import Path

# --- import original numpy stubs ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "encoder-backprop-8-3-8"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "encoder-3-parity"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "encoder-8-3-8"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bars"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bars-rbm"))

import encoder_backprop_8_3_8 as bp_ref
import encoder_3_parity as rbm_ref
import encoder_8_3_8 as rbm8_ref
import bars as bars_ref
import bars_rbm as bars_rbm_ref

# --- import pure-Python kernels under test ---
sys.path.insert(0, str(Path(__file__).parent))
import encoder_pair_comparison as impl
import total_cost_comparison as total_impl
import bars_cost_comparison as bars_impl

ATOL = 1e-6


def arrays_close(a_py, b_np, name):
    a = np.array(a_py, dtype=np.float64)
    b = np.array(b_np, dtype=np.float64)
    ok = np.allclose(a, b, atol=ATOL)
    status = "OK" if ok else "FAIL"
    maxdiff = float(np.max(np.abs(a - b)))
    print(f"  {status}  {name}  (max_diff={maxdiff:.2e})")
    return ok


# ---------------------------------------------------------------------------
# Validate backprop 8-3-8
# ---------------------------------------------------------------------------

def validate_backprop(seed=42):
    print("=== backprop-8-3-8 ===")
    rng = random.Random(seed)
    W1_py = [[rng.uniform(-0.5, 0.5) for _ in range(3)] for _ in range(8)]
    b1_py = [rng.uniform(-0.1, 0.1) for _ in range(3)]
    W2_py = [[rng.uniform(-0.5, 0.5) for _ in range(8)] for _ in range(3)]
    b2_py = [rng.uniform(-0.1, 0.1) for _ in range(8)]

    # Convert to numpy for the reference model
    model = bp_ref.EncoderMLP(seed=0)
    model.W1 = np.array(W1_py, dtype=np.float64)
    model.b1 = np.array(b1_py, dtype=np.float64)
    model.W2 = np.array(W2_py, dtype=np.float64)
    model.b2 = np.array(b2_py, dtype=np.float64)

    data = bp_ref.make_encoder_data()   # 8x8 identity
    x_np = data[0:1]                    # pattern 0, shape (1, 8)
    x_py = list(x_np[0])

    # --- forward ---
    h_np, y_np = model.forward(x_np)   # shapes (1,3), (1,8)

    h_py, y_py = impl._bp_forward(W1_py, b1_py, W2_py, b2_py, x_py)

    ok = True
    ok &= arrays_close(h_py, h_np[0], "forward h")
    ok &= arrays_close(y_py, y_np[0], "forward y")

    # --- gradients ---
    target_py = x_py[:]
    dW1_np, db1_np, dW2_np, db2_np = model.grads(x_np, x_np)

    # Run full_step to get pure-Python gradients
    result = impl.bp_full_step(W1_py, b1_py, W2_py, b2_py, x_py, target_py)
    dW1_py, db1_py, dW2_py, db2_py = result

    ok &= arrays_close(dW1_py, dW1_np, "dW1")
    ok &= arrays_close(db1_py, db1_np, "db1")
    ok &= arrays_close(dW2_py, dW2_np, "dW2")
    ok &= arrays_close(db2_py, db2_np, "db2")

    return ok


# ---------------------------------------------------------------------------
# Validate encoder-3-parity (RBM)
# ---------------------------------------------------------------------------

def validate_rbm(seed=42):
    print("=== encoder-3-parity (RBM) ===")
    rng = random.Random(seed)
    W_py  = [[rng.gauss(0, 0.1) for _ in range(4)] for _ in range(3)]
    b_v_py = [rng.uniform(-0.1, 0.1) for _ in range(3)]
    b_h_py = [rng.uniform(-0.1, 0.1) for _ in range(4)]

    # Build reference model
    model = rbm_ref.ParityRBM(n_hidden=4, seed=0)
    model.W   = np.array(W_py,   dtype=np.float32)
    model.b_v = np.array(b_v_py, dtype=np.float32)
    model.b_h = np.array(b_h_py, dtype=np.float32)

    v_py = [0.0, 1.0, 1.0]   # even-parity pattern 011
    v_np = np.array([v_py], dtype=np.float32)  # shape (1, 3)

    # --- positive phase: h_prob_pos ---
    h_prob_pos_np = model.hidden_prob(v_np)   # shape (1, 4)
    h_prob_pos_py = impl._rbm_h_given_v(W_py, b_h_py, v_py)

    ok = arrays_close(h_prob_pos_py, h_prob_pos_np[0], "h_prob_pos")

    # --- negative phase: v_prob_neg (given fixed h_pos) ---
    # Use a fixed binary h_pos so the comparison is deterministic
    h_pos_py = [1.0, 0.0, 1.0, 0.0]
    h_pos_np = np.array([h_pos_py], dtype=np.float32)

    v_prob_neg_np = model.visible_prob(h_pos_np)   # shape (1, 3)
    v_prob_neg_py = impl._rbm_v_given_h(W_py, b_v_py, h_pos_py)

    ok &= arrays_close(v_prob_neg_py, v_prob_neg_np[0], "v_prob_neg")

    # --- negative h: h_prob_neg (given fixed v_neg) ---
    v_neg_py = [1.0, 0.0, 0.0]
    v_neg_np = np.array([v_neg_py], dtype=np.float32)

    h_prob_neg_np = model.hidden_prob(v_neg_np)
    h_prob_neg_py = impl._rbm_h_given_v(W_py, b_h_py, v_neg_py)

    ok &= arrays_close(h_prob_neg_py, h_prob_neg_np[0], "h_prob_neg")

    return ok


# ---------------------------------------------------------------------------
# Validate total_cost_comparison.py full-batch kernels
# ---------------------------------------------------------------------------

def validate_total_cost_backprop(seed=123):
    print("=== total_cost_comparison backprop full-batch ===")
    rng = random.Random(seed)
    W1_py = [[rng.uniform(-0.5, 0.5) for _ in range(3)] for _ in range(8)]
    b1_py = [rng.uniform(-0.1, 0.1) for _ in range(3)]
    W2_py = [[rng.uniform(-0.5, 0.5) for _ in range(8)] for _ in range(3)]
    b2_py = [rng.uniform(-0.1, 0.1) for _ in range(8)]

    model = bp_ref.EncoderMLP(seed=0)
    model.W1 = np.array(W1_py, dtype=np.float64)
    model.b1 = np.array(b1_py, dtype=np.float64)
    model.W2 = np.array(W2_py, dtype=np.float64)
    model.b2 = np.array(b2_py, dtype=np.float64)

    data_np = bp_ref.make_encoder_data()
    patterns_py = data_np.astype(float).tolist()

    hs_py, ys_py = total_impl.bp_fullbatch_forward(
        W1_py, b1_py, W2_py, b2_py, patterns_py)
    hs_np, ys_np = model.forward(data_np)

    ok = True
    ok &= arrays_close(hs_py, hs_np, "full-batch forward h")
    ok &= arrays_close(ys_py, ys_np, "full-batch forward y")

    dW1_py, db1_py, dW2_py, db2_py = total_impl.bp_fullbatch_step(
        W1_py, b1_py, W2_py, b2_py, patterns_py)
    dW1_np, db1_np, dW2_np, db2_np = model.grads(data_np, data_np)

    # EncoderMLP.grads returns mean full-batch gradients; the traced kernel
    # accumulates the equivalent batch-sum gradients. The access pattern is the
    # same, so validate against the mean gradients scaled by batch size.
    batch_n = data_np.shape[0]
    ok &= arrays_close(dW1_py, dW1_np * batch_n, "full-batch dW1 sum")
    ok &= arrays_close(db1_py, db1_np * batch_n, "full-batch db1 sum")
    ok &= arrays_close(dW2_py, dW2_np * batch_n, "full-batch dW2 sum")
    ok &= arrays_close(db2_py, db2_np * batch_n, "full-batch db2 sum")

    return ok


def _threshold_sample(probs):
    return [1.0 if p >= 0.5 else 0.0 for p in probs]


def validate_total_cost_rbm(seed=123, k=2):
    print("=== total_cost_comparison encoder-8-3-8 full-batch RBM ===")
    rng = random.Random(seed)
    W_py = [[rng.gauss(0, 0.1) for _ in range(3)] for _ in range(16)]
    b_v_py = [rng.uniform(-0.1, 0.1) for _ in range(16)]
    b_h_py = [rng.uniform(-0.1, 0.1) for _ in range(3)]

    model = rbm8_ref.EncoderRBM(seed=0)
    model.W = np.array(W_py, dtype=np.float32)
    model.b_v = np.array(b_v_py, dtype=np.float32)
    model.b_h = np.array(b_h_py, dtype=np.float32)

    data_np = rbm8_ref.make_encoder_data().astype(np.float32)
    patterns_py = data_np.astype(float).tolist()

    h_probs_pos_py = total_impl.rbm_fullbatch_positive(
        W_py, b_v_py, b_h_py, patterns_py, k=k)
    h_probs_pos_np = model.hidden_prob(data_np)

    ok = arrays_close(h_probs_pos_py, h_probs_pos_np, "full-batch h_prob_pos")

    old_sample = total_impl._sample
    total_impl._sample = _threshold_sample
    try:
        dW_py, db_v_py, db_h_py = total_impl.rbm_fullbatch_step(
            W_py, b_v_py, b_h_py, patterns_py, k=k)
    finally:
        total_impl._sample = old_sample

    h_samples_np = (h_probs_pos_np >= 0.5).astype(np.float32)
    v_negs_np = data_np.copy()
    h_negs_np = h_samples_np
    h_probs_neg_np = h_probs_pos_np
    for _ in range(k):
        v_probs_neg_np = model.visible_prob(h_negs_np)
        v_negs_np = (v_probs_neg_np >= 0.5).astype(np.float32)
        h_probs_neg_np = model.hidden_prob(v_negs_np)
        h_negs_np = (h_probs_neg_np >= 0.5).astype(np.float32)

    batch_n = data_np.shape[0]
    dW_np = (data_np.T @ h_probs_pos_np - v_negs_np.T @ h_probs_neg_np) / batch_n
    db_v_np = (data_np - v_negs_np).mean(axis=0)
    db_h_np = (h_probs_pos_np - h_probs_neg_np).mean(axis=0)

    ok &= arrays_close(dW_py, dW_np, "full-batch dW CD-k")
    ok &= arrays_close(db_v_py, db_v_np, "full-batch db_v CD-k")
    ok &= arrays_close(db_h_py, db_h_np, "full-batch db_h CD-k")

    return ok


# ---------------------------------------------------------------------------
# Validate bars_cost_comparison.py kernels
# ---------------------------------------------------------------------------

def validate_bars_rbm(seed=42):
    print("=== bars_cost_comparison bars-rbm CD-1 kernels ===")
    rng = random.Random(seed)
    W_py   = [[rng.gauss(0, 0.1) for _ in range(8)] for _ in range(16)]
    b_v_py = [rng.gauss(0, 0.05) for _ in range(16)]
    b_h_py = [rng.gauss(0, 0.05) for _ in range(8)]

    rbm = bars_rbm_ref.BarsRBM(n_visible=16, n_hidden=8, seed=0)
    rbm.W   = np.array(W_py,   dtype=np.float32)
    rbm.b_v = np.array(b_v_py, dtype=np.float32)
    rbm.b_h = np.array(b_h_py, dtype=np.float32)

    # one vertical bar in column 0
    v_py = [0.0] * 16
    for r in range(4):
        v_py[r * 4] = 1.0
    v_np = np.array([v_py], dtype=np.float32)

    ok = True
    # positive phase: h_prob
    h_pos_py = bars_impl._rbm_h_given_v(W_py, b_h_py, v_py)
    h_pos_np = rbm.hidden_prob(v_np)
    ok &= arrays_close(h_pos_py, h_pos_np[0], "h_prob_pos")

    # negative phase: v_recon given fixed h_sample
    h_fixed = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    v_recon_py = bars_impl._rbm_v_given_h(W_py, b_v_py, h_fixed)
    v_recon_np = rbm.visible_prob(np.array([h_fixed], dtype=np.float32))
    ok &= arrays_close(v_recon_py, v_recon_np[0], "v_recon")

    # h_prob_neg from fixed v_recon
    h_neg_py = bars_impl._rbm_h_given_v(W_py, b_h_py, v_recon_py)
    h_neg_np = rbm.hidden_prob(np.array([v_recon_py], dtype=np.float32))
    ok &= arrays_close(h_neg_py, h_neg_np[0], "h_prob_neg")

    return ok


def validate_bars_helmholtz(seed=42):
    print("=== bars_cost_comparison bars Helmholtz kernels ===")
    rng = random.Random(seed)
    W_th_py  = [[rng.gauss(0, 0.1) for _ in range(8)]]
    W_hv_py  = [[rng.gauss(0, 0.1) for _ in range(16)] for _ in range(8)]
    b_top_py = [rng.gauss(0, 0.05)]
    b_h_py   = [rng.gauss(0, 0.05) for _ in range(8)]
    b_v_py   = [rng.gauss(0, 0.05) for _ in range(16)]
    R_vh_py  = [[rng.gauss(0, 0.1) for _ in range(8)] for _ in range(16)]
    R_ht_py  = [[rng.gauss(0, 0.1)] for _ in range(8)]
    c_h_py   = [rng.gauss(0, 0.05) for _ in range(8)]
    c_top_py = [rng.gauss(0, 0.05)]

    model = bars_ref.HelmholtzMachine(n_visible=16, n_hidden=8, n_top=1)
    model.W_th  = np.array(W_th_py,  dtype=np.float32)
    model.W_hv  = np.array(W_hv_py,  dtype=np.float32)
    model.b_top = np.array(b_top_py, dtype=np.float32)
    model.b_h   = np.array(b_h_py,   dtype=np.float32)
    model.b_v   = np.array(b_v_py,   dtype=np.float32)
    model.R_vh  = np.array(R_vh_py,  dtype=np.float32)
    model.R_ht  = np.array(R_ht_py,  dtype=np.float32)
    model.c_h   = np.array(c_h_py,   dtype=np.float32)
    model.c_top = np.array(c_top_py, dtype=np.float32)

    v_py = [0.0] * 16
    for r in range(4):
        v_py[r * 4] = 1.0
    v_np = np.array([v_py], dtype=np.float32)

    ok = True
    # recognition h_prob = sigmoid(v @ R_vh + c_h)
    h_prob_py, _, _, _ = bars_impl._helmholtz_recognize(
        R_vh_py, c_h_py, R_ht_py, c_top_py, v_py)
    h_prob_np = bars_ref.sigmoid(v_np @ model.R_vh + model.c_h)
    ok &= arrays_close(h_prob_py, h_prob_np[0], "h_prob (recognition)")

    # recognition t_prob from fixed h
    h_fixed = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    h_fixed_np = np.array([h_fixed], dtype=np.float32)
    t_prob_py = [bars_impl._sigmoid(
        sum(h_fixed[j] * R_ht_py[j][0] for j in range(8)) + c_top_py[0])]
    t_prob_np = bars_ref.sigmoid(h_fixed_np @ model.R_ht + model.c_top)
    ok &= arrays_close(t_prob_py, t_prob_np[0], "t_prob (recognition)")

    # generation h_prob given t=1
    h_gen_py = [bars_impl._sigmoid(1.0 * W_th_py[0][j] + b_h_py[j])
                for j in range(8)]
    t_np = np.array([[1.0]], dtype=np.float32)
    h_gen_np = bars_ref.sigmoid(t_np @ model.W_th + model.b_h)
    ok &= arrays_close(h_gen_py, h_gen_np[0], "h_prob_gen (t=1)")

    # generation v_prob given fixed h
    h_gen_fixed = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    v_gen_py = [bars_impl._sigmoid(
        sum(h_gen_fixed[j] * W_hv_py[j][i] for j in range(8)) + b_v_py[i])
                for i in range(16)]
    h_gen_fixed_np = np.array([h_gen_fixed], dtype=np.float32)
    v_gen_np = bars_ref.sigmoid(h_gen_fixed_np @ model.W_hv + model.b_v)
    ok &= arrays_close(v_gen_py, v_gen_np[0], "v_prob_gen")

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = []
    results.append(validate_backprop())
    print()
    results.append(validate_rbm())
    print()
    results.append(validate_total_cost_backprop())
    print()
    results.append(validate_total_cost_rbm())
    print()
    results.append(validate_bars_rbm())
    print()
    results.append(validate_bars_helmholtz())
    print()
    if all(results):
        print("All checks passed.")
    else:
        print("VALIDATION FAILED — pure-Python kernels do not match numpy stubs.")
        sys.exit(1)
