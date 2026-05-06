"""
Deep Boltzmann Machine on MNIST -- the 2009 follow-up to the 2006 DBN.

Source:
  Salakhutdinov & Hinton (2009), "Deep Boltzmann Machines",
  AISTATS 2009. https://proceedings.mlr.press/v5/salakhutdinov09a.html

Where the 2006 DBN ([dbn-mnist/](../dbn-mnist/)) is a *hybrid* — directed
sigmoid belief net below, undirected RBM at the top — the 2009 DBM is
fully *undirected*: every layer connection is symmetric. The
representational difference is real:

  * In a DBN, p(h1 | v) does NOT depend on the layers above h1. The
    posterior is just the bottom RBM's q(h1 | v); h2 has no influence.
  * In a DBM, p(h1 | v) integrates top-down evidence from h2:
        p(h1 = 1 | v, h2) = sigmoid(W1.T @ v + W2 @ h2 + b_h1)
    The "explaining away" through h2 makes inference exact only at the
    fixed point of mean-field iteration.

This makes DBMs harder to train (PCD + mean-field positive phase) but
empirically generalizes a hair better — 0.95% vs the DBN's 1.25% on MNIST,
the original SOTA of the paper. This implementation reproduces the algorithm
without the discriminative fine-tuning step that closed the gap.

Architecture: 784 - 500 - 1000 (the paper's two-hidden-layer MNIST DBM).

Files:
    dbm_mnist.py             - this file (model + train + eval, CLI)
    visualize_dbm_mnist.py   - static viz: layer-1 filters, mean-field
                               trajectory, generative samples
    make_dbm_mnist_gif.py    - animated GIF of layer-1 filters during
                               joint PCD training
"""

from __future__ import annotations
import argparse
import gzip
import os
import time
import urllib.request

import numpy as np


# ----------------------------------------------------------------------
# MNIST (same loader as dbn-mnist)
# ----------------------------------------------------------------------

CACHE = os.path.expanduser("~/.cache/hinton-mnist")
URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def load_mnist() -> dict:
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for k, url in URLS.items():
        path = os.path.join(CACHE, os.path.basename(url))
        if not os.path.exists(path):
            print(f"  downloading {url}")
            urllib.request.urlretrieve(url, path)
        with gzip.open(path, "rb") as f:
            data = f.read()
        if "images" in k:
            out[k] = (np.frombuffer(data, np.uint8, offset=16)
                      .reshape(-1, 28, 28).astype(np.float32) / 255.0)
        else:
            out[k] = np.frombuffer(data, np.uint8, offset=8).astype(np.int64)
    return out


def balanced_subsample(images: np.ndarray, labels: np.ndarray,
                       n_per_class: int, rng: np.random.Generator) -> tuple:
    cls_indices = [np.where(labels == c)[0] for c in range(10)]
    n_per_class = min(n_per_class, min(len(c) for c in cls_indices))
    idx = []
    for c in range(10):
        idx.append(rng.choice(cls_indices[c], size=n_per_class, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return images[idx], labels[idx]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


# ----------------------------------------------------------------------
# Greedy doubled-input RBM pretraining
# ----------------------------------------------------------------------

class _RBM:
    """A plain CD-1 RBM, used only during DBM greedy pretraining.

    The "doubled-input" trick (Salakhutdinov & Hinton 2009 §4) is
    implemented by `bottom_doubling` / `top_doubling` flags: when set,
    the corresponding side's contribution is multiplied by 2 during
    pretraining, so the trained weights match what each layer would
    receive in the assembled DBM where the layer has TWO neighbours
    instead of one.
    """

    def __init__(self, n_visible: int, n_hidden: int, init_scale: float = 0.01,
                 bottom_doubling: bool = False, top_doubling: bool = False,
                 rng=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.bottom_doubling = bottom_doubling
        self.top_doubling = top_doubling
        rng = rng or np.random.default_rng()
        self.W = rng.normal(0, init_scale, size=(n_visible, n_hidden)).astype(np.float32)
        self.b_v = np.zeros(n_visible, dtype=np.float32)
        self.b_h = np.zeros(n_hidden, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db_v = np.zeros_like(self.b_v)
        self.db_h = np.zeros_like(self.b_h)

    def _h_input(self, v):
        x = v @ self.W + self.b_h
        return 2 * x if self.bottom_doubling else x

    def _v_input(self, h):
        x = h @ self.W.T + self.b_v
        return 2 * x if self.top_doubling else x

    def prob_h_given_v(self, v): return sigmoid(self._h_input(v))
    def prob_v_given_h(self, h): return sigmoid(self._v_input(h))

    def cd1_update(self, v0, lr, momentum, weight_decay, rng):
        ph0 = self.prob_h_given_v(v0)
        h0 = (rng.random(ph0.shape) < ph0).astype(np.float32)
        v1 = self.prob_v_given_h(h0)
        ph1 = self.prob_h_given_v(v1)
        n = v0.shape[0]
        dW = (v0.T @ ph0 - v1.T @ ph1) / n - weight_decay * self.W
        db_v = (v0 - v1).mean(axis=0)
        db_h = (ph0 - ph1).mean(axis=0)
        self.dW = momentum * self.dW + lr * dW
        self.db_v = momentum * self.db_v + lr * db_v
        self.db_h = momentum * self.db_h + lr * db_h
        self.W += self.dW
        self.b_v += self.db_v
        self.b_h += self.db_h
        return float(((v0 - v1) ** 2).mean())


def train_rbm(rbm, data, n_epochs, batch_size, lr, momentum, weight_decay,
              rng, label="L?", verbose=True):
    n = data.shape[0]
    losses = []
    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        loss = 0.0
        nb = 0
        for s in range(0, n, batch_size):
            mom = 0.5 if epoch < 5 else momentum
            loss += rbm.cd1_update(data[perm[s:s+batch_size]], lr, mom,
                                   weight_decay, rng)
            nb += 1
        loss /= max(nb, 1)
        losses.append(loss)
        if verbose:
            print(f"  [{label}] epoch {epoch+1}/{n_epochs}  recon_mse={loss:.4f}")
    return losses


# ----------------------------------------------------------------------
# DBM (joint, 2 hidden layers)
# ----------------------------------------------------------------------

class DBM:
    """Two-hidden-layer Deep Boltzmann Machine: V - H1 - H2.

    All layers are binary; all connections are undirected (symmetric).
    Inference is mean-field; learning is PCD with mean-field positive
    phase (Salakhutdinov & Hinton 2009 §3-§4).
    """

    def __init__(self, n_v=784, n_h1=500, n_h2=1000, rng=None):
        rng = rng or np.random.default_rng()
        self.n_v, self.n_h1, self.n_h2 = n_v, n_h1, n_h2
        # weights (real fields after halving from doubled pretraining)
        self.W1 = np.zeros((n_v, n_h1), dtype=np.float32)
        self.W2 = np.zeros((n_h1, n_h2), dtype=np.float32)
        self.b_v = np.zeros(n_v, dtype=np.float32)
        self.b_h1 = np.zeros(n_h1, dtype=np.float32)
        self.b_h2 = np.zeros(n_h2, dtype=np.float32)
        # momentum buffers
        self.dW1 = np.zeros_like(self.W1)
        self.dW2 = np.zeros_like(self.W2)
        self.db_v = np.zeros_like(self.b_v)
        self.db_h1 = np.zeros_like(self.b_h1)
        self.db_h2 = np.zeros_like(self.b_h2)

    # --- mean-field inference: returns posterior means q(h1|v), q(h2|v) ---
    def mean_field(self, v, n_iters=10, init_mu1=None, init_mu2=None):
        if init_mu2 is None:
            mu2 = sigmoid(np.zeros((v.shape[0], self.n_h2), dtype=np.float32))
        else:
            mu2 = init_mu2
        if init_mu1 is None:
            mu1 = sigmoid(v @ self.W1 + self.b_h1)
        else:
            mu1 = init_mu1
        for _ in range(n_iters):
            mu2 = sigmoid(mu1 @ self.W2 + self.b_h2)
            mu1 = sigmoid(v @ self.W1 + mu2 @ self.W2.T + self.b_h1)
        return mu1, mu2

    # --- one Gibbs step on the fantasy chain (even-odd alternating) ---
    def gibbs_step(self, v, h1, h2, rng):
        # Sample {v, h2} | h1 (these are conditionally independent given h1)
        pv = sigmoid(h1 @ self.W1.T + self.b_v)
        ph2 = sigmoid(h1 @ self.W2 + self.b_h2)
        v_new = (rng.random(pv.shape) < pv).astype(np.float32)
        h2_new = (rng.random(ph2.shape) < ph2).astype(np.float32)
        # Sample h1 | v, h2
        ph1 = sigmoid(v_new @ self.W1 + h2_new @ self.W2.T + self.b_h1)
        h1_new = (rng.random(ph1.shape) < ph1).astype(np.float32)
        return v_new, h1_new, h2_new

    def init_from_pretrained(self, rbm1: _RBM, rbm2: _RBM):
        """Halve the doubled-pretraining weights and stitch them together.

        rbm1 was trained as 'visible doubled' so its W is twice the value
        we want at the bottom of the DBM (which has two contributions to
        h1: one from v, one from h2). Same on the top side.
        """
        self.W1 = rbm1.W.copy()  # doubled-bottom; halve below.
        self.W2 = rbm2.W.copy()  # doubled-top; halve below.
        # average the two bias estimates for h1
        self.b_v = rbm1.b_v.copy()
        self.b_h1 = 0.5 * (rbm1.b_h + rbm2.b_v)
        self.b_h2 = rbm2.b_h.copy()
        # halve weights so that, in the assembled DBM where h1 receives
        # both v->h1 and h2->h1 traffic, the total drive matches what
        # each pretrained RBM saw with its doubled-input.
        self.W1 *= 0.5
        self.W2 *= 0.5

    # --- full joint update ---
    def joint_update(self, v_batch, fantasy, lr, momentum, weight_decay,
                     n_mf_iters, rng):
        # Positive phase: mean-field
        mu1, mu2 = self.mean_field(v_batch, n_iters=n_mf_iters)
        n = v_batch.shape[0]
        pos_W1 = v_batch.T @ mu1 / n
        pos_W2 = mu1.T @ mu2 / n
        pos_b_v = v_batch.mean(axis=0)
        pos_b_h1 = mu1.mean(axis=0)
        pos_b_h2 = mu2.mean(axis=0)

        # Negative phase: PCD — persistent fantasy chain advanced by one
        # Gibbs step.
        v_f, h1_f, h2_f = fantasy
        v_f, h1_f, h2_f = self.gibbs_step(v_f, h1_f, h2_f, rng)
        m = v_f.shape[0]
        neg_W1 = v_f.T @ h1_f / m
        neg_W2 = h1_f.T @ h2_f / m
        neg_b_v = v_f.mean(axis=0)
        neg_b_h1 = h1_f.mean(axis=0)
        neg_b_h2 = h2_f.mean(axis=0)

        # Gradients
        gW1 = pos_W1 - neg_W1 - weight_decay * self.W1
        gW2 = pos_W2 - neg_W2 - weight_decay * self.W2
        gb_v = pos_b_v - neg_b_v
        gb_h1 = pos_b_h1 - neg_b_h1
        gb_h2 = pos_b_h2 - neg_b_h2

        # Apply momentum + update
        self.dW1 = momentum * self.dW1 + lr * gW1
        self.dW2 = momentum * self.dW2 + lr * gW2
        self.db_v = momentum * self.db_v + lr * gb_v
        self.db_h1 = momentum * self.db_h1 + lr * gb_h1
        self.db_h2 = momentum * self.db_h2 + lr * gb_h2
        self.W1 += self.dW1
        self.W2 += self.dW2
        self.b_v += self.db_v
        self.b_h1 += self.db_h1
        self.b_h2 += self.db_h2

        recon = float(((v_batch - sigmoid(mu1 @ self.W1.T + self.b_v)) ** 2).mean())
        return (v_f, h1_f, h2_f), recon


# ----------------------------------------------------------------------
# Logistic regression on mean-field features
# ----------------------------------------------------------------------

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def train_logreg(X, y, n_classes, n_epochs, batch_size, lr, l2, rng,
                 X_val=None, y_val=None, verbose=True):
    n, d = X.shape
    W = np.zeros((d, n_classes), dtype=np.float32)
    b = np.zeros(n_classes, dtype=np.float32)
    Y = np.eye(n_classes, dtype=np.float32)[y]
    history = {"train_acc": [], "val_acc": []}
    for ep in range(n_epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch_size):
            sl = perm[s:s+batch_size]
            P = softmax(X[sl] @ W + b)
            W -= lr * (X[sl].T @ (P - Y[sl]) / X[sl].shape[0] + l2 * W)
            b -= lr * (P - Y[sl]).mean(axis=0)
        train_acc = float((np.argmax(softmax(X @ W + b), axis=1) == y).mean())
        history["train_acc"].append(train_acc)
        if X_val is not None:
            val_acc = float((np.argmax(softmax(X_val @ W + b), axis=1) == y_val).mean())
            history["val_acc"].append(val_acc)
            if verbose:
                print(f"  [logreg] epoch {ep+1}/{n_epochs}  "
                      f"train={train_acc*100:.2f}%  test={val_acc*100:.2f}%")
        elif verbose:
            print(f"  [logreg] epoch {ep+1}/{n_epochs}  train={train_acc*100:.2f}%")
    return W, b, history


# ----------------------------------------------------------------------
# End-to-end DBM training
# ----------------------------------------------------------------------

def train_dbm(layer_sizes=(784, 500, 1000),
              n_train_per_class=1000,
              n_pretrain_epochs=10,
              n_joint_epochs=5,
              batch_size=100,
              n_fantasies=100,
              n_mf_iters=5,
              pretrain_lr=0.05,
              joint_lr=0.001,
              momentum=0.9,
              joint_momentum=0.0,
              weight_decay=2e-4,
              n_classifier_epochs=30,
              classifier_lr=0.05,
              classifier_l2=1e-4,
              seed=0,
              snapshot_every=0,
              snapshot_callback=None,
              verbose=True) -> dict:
    rng = np.random.default_rng(seed)

    if verbose:
        print("Loading MNIST...")
    mnist = load_mnist()
    if verbose:
        print(f"Subsampling {n_train_per_class * 10} balanced training images...")
    X_train, y_train = balanced_subsample(
        mnist["train_images"].reshape(-1, 784),
        mnist["train_labels"], n_train_per_class, rng)
    X_test = mnist["test_images"].reshape(-1, 784)
    y_test = mnist["test_labels"]

    # === 1. Greedy pretraining with doubled inputs ===
    n_v, n_h1, n_h2 = layer_sizes
    if verbose:
        print(f"\nPretraining bottom RBM ({n_v} -> {n_h1}) with bottom-doubling...")
    rbm1 = _RBM(n_v, n_h1, bottom_doubling=True,
                rng=np.random.default_rng(seed * 1000 + 1))
    losses1 = train_rbm(rbm1, X_train, n_pretrain_epochs, batch_size,
                        pretrain_lr, momentum, weight_decay, rng,
                        label="L1*", verbose=verbose)

    # Layer-1 features (using NON-doubled forward to feed layer 2 — the
    # forward computes p(h1|v) as it would inside the joint DBM).
    rbm1_solo = _RBM(n_v, n_h1)
    rbm1_solo.W = rbm1.W.copy()
    rbm1_solo.b_h = rbm1.b_h.copy()
    F1 = rbm1_solo.prob_h_given_v(X_train)

    if verbose:
        print(f"\nPretraining top RBM ({n_h1} -> {n_h2}) with top-doubling...")
    rbm2 = _RBM(n_h1, n_h2, top_doubling=True,
                rng=np.random.default_rng(seed * 1000 + 2))
    losses2 = train_rbm(rbm2, F1, n_pretrain_epochs, batch_size,
                        pretrain_lr, momentum, weight_decay, rng,
                        label="L2*", verbose=verbose)

    # === 2. Stitch into DBM, halve weights ===
    dbm = DBM(n_v, n_h1, n_h2, rng=rng)
    dbm.init_from_pretrained(rbm1, rbm2)

    # === 3. Joint PCD training ===
    if verbose:
        print(f"\nJoint PCD training ({n_joint_epochs} epochs, "
              f"{n_fantasies} fantasy particles, {n_mf_iters} MF iters)...")
    # Initialize fantasy particles from data
    init_idx = rng.choice(len(X_train), size=n_fantasies, replace=False)
    v_f = X_train[init_idx].copy()
    mu1_init, mu2_init = dbm.mean_field(v_f, n_iters=n_mf_iters)
    h1_f = (rng.random(mu1_init.shape) < mu1_init).astype(np.float32)
    h2_f = (rng.random(mu2_init.shape) < mu2_init).astype(np.float32)
    fantasy = (v_f, h1_f, h2_f)

    joint_losses = []
    for ep in range(n_joint_epochs):
        perm = rng.permutation(len(X_train))
        ep_loss = 0.0
        nb = 0
        for s in range(0, len(X_train), batch_size):
            v_batch = X_train[perm[s:s+batch_size]]
            fantasy, recon = dbm.joint_update(
                v_batch, fantasy, joint_lr, joint_momentum, weight_decay,
                n_mf_iters, rng)
            ep_loss += recon
            nb += 1
        ep_loss /= max(nb, 1)
        joint_losses.append(ep_loss)
        if verbose:
            print(f"  [joint] epoch {ep+1}/{n_joint_epochs}  "
                  f"recon_mse={ep_loss:.4f}  "
                  f"|W1|={np.linalg.norm(dbm.W1):.2f}  "
                  f"|W2|={np.linalg.norm(dbm.W2):.2f}")
        if snapshot_every and snapshot_callback and (ep + 1) % snapshot_every == 0:
            snapshot_callback(ep, dbm, joint_losses)

    # === 4. Classifier on concatenated mean-field [h1, h2] features ===
    # Salakhutdinov & Hinton 2009 §6.1 read out from h2; using both
    # gives the classifier the full DBM representation, including the
    # explaining-away corrections that h2 imposes on h1 via mean-field.
    if verbose:
        print(f"\nMean-field inference for classifier features...")
    mu1_train, mu2_train = dbm.mean_field(X_train, n_iters=20)
    mu1_test, mu2_test = dbm.mean_field(X_test, n_iters=20)
    F_train = np.concatenate([mu1_train, mu2_train], axis=1)
    F_test = np.concatenate([mu1_test, mu2_test], axis=1)
    if verbose:
        print(f"Training logistic regression on {F_train.shape[1]}-d "
              f"[h1, h2] mean-field features...")
    W_cls, b_cls, cls_history = train_logreg(
        F_train, y_train, 10, n_classifier_epochs, batch_size,
        classifier_lr, classifier_l2, rng, F_test, y_test, verbose=verbose)

    final_train_acc = cls_history["train_acc"][-1]
    final_test_acc = cls_history["val_acc"][-1]
    if verbose:
        print(f"\nFinal: train {final_train_acc*100:.2f}%  test {final_test_acc*100:.2f}%  "
              f"(error {(1 - final_test_acc)*100:.2f}%)")

    return {
        "dbm": dbm,
        "rbm1_pretrained": rbm1,
        "rbm2_pretrained": rbm2,
        "pretrain_losses": [losses1, losses2],
        "joint_losses": joint_losses,
        "W_cls": W_cls, "b_cls": b_cls,
        "cls_history": cls_history,
        "final_train_acc": final_train_acc,
        "final_test_acc": final_test_acc,
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "layer_sizes": layer_sizes,
    }


# ----------------------------------------------------------------------
# Generative sampling
# ----------------------------------------------------------------------

def sample_dbm(result, n_samples=16, n_gibbs=200, seed=1, init_from_data=True):
    """Run alternating Gibbs on the trained DBM and emit p(v|h1) of the
    final state."""
    rng = np.random.default_rng(seed)
    dbm = result["dbm"]
    if init_from_data:
        idx = rng.choice(len(result["X_test"]), size=n_samples, replace=False)
        v = result["X_test"][idx].copy()
        mu1, mu2 = dbm.mean_field(v, n_iters=10)
        h1 = (rng.random(mu1.shape) < mu1).astype(np.float32)
        h2 = (rng.random(mu2.shape) < mu2).astype(np.float32)
    else:
        v = (rng.random((n_samples, dbm.n_v)) < 0.5).astype(np.float32)
        h1 = (rng.random((n_samples, dbm.n_h1)) < 0.5).astype(np.float32)
        h2 = (rng.random((n_samples, dbm.n_h2)) < 0.5).astype(np.float32)
    for _ in range(n_gibbs):
        v, h1, h2 = dbm.gibbs_step(v, h1, h2, rng)
    # final readout = p(v|h1)
    return sigmoid(h1 @ dbm.W1.T + dbm.b_v).reshape(-1, 28, 28)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train-per-class", type=int, default=1000)
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--joint-epochs", type=int, default=8)
    p.add_argument("--classifier-epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="Tiny budget for smoke testing.")
    args = p.parse_args()

    if args.quick:
        n_per_class = 300
        pre = 4
        joint = 3
        cls = 10
    else:
        n_per_class = args.n_train_per_class
        pre = args.pretrain_epochs
        joint = args.joint_epochs
        cls = args.classifier_epochs

    t0 = time.time()
    result = train_dbm(
        n_train_per_class=n_per_class,
        n_pretrain_epochs=pre,
        n_joint_epochs=joint,
        n_classifier_epochs=cls,
        seed=args.seed,
    )
    print(f"\nTotal wallclock: {time.time() - t0:.1f}s")
    return result


if __name__ == "__main__":
    main()
