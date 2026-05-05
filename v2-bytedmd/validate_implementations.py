"""
Validates that the pure-Python kernels in encoder_pair_comparison.py produce
identical outputs to the original numpy stubs.

Checks:
  backprop-8-3-8 : forward h, y and gradients dW1, dW2, db1, db2
  encoder-3-parity: h_prob_pos, v_prob_neg, h_prob_neg (deterministic parts of CD)

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

import encoder_backprop_8_3_8 as bp_ref
import encoder_3_parity as rbm_ref

# --- import pure-Python kernels under test ---
sys.path.insert(0, str(Path(__file__).parent))
import encoder_pair_comparison as impl

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
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = []
    results.append(validate_backprop())
    print()
    results.append(validate_rbm())
    print()
    if all(results):
        print("All checks passed.")
    else:
        print("VALIDATION FAILED — pure-Python kernels do not match numpy stubs.")
        sys.exit(1)
