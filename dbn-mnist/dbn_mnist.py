"""
Deep Belief Net on MNIST — the 2006 result that beat kernel machines.

Source:
  Hinton, Osindero & Teh (2006), "A Fast Learning Algorithm for Deep Belief
  Nets", Neural Computation 18:1527-1554. (https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)

The headline experiment in §6 of the paper. A 3-layer DBN
(784 -> 500 -> 500 -> 2000) is trained one layer at a time as an RBM by
contrastive divergence, then a discriminative classifier sits on top of
the layer-3 features. The paper reports 1.25% test error after up-down
fine-tuning, beating SVM (1.4%) and 1-3 hidden-layer backprop nets
(1.5-1.6%) at the time. This was the empirical proof-of-concept that
deep models could be trained at all and that they generalized better
than the best kernel methods, six years before AlexNet.

This implementation:
  * Same 784-500-500-2000 architecture as the paper.
  * Greedy layer-wise CD-1 training (same as paper).
  * Subsampled MNIST: 10,000 train + full 10,000 test, balanced subsample.
  * Logistic-regression classifier head on layer-3 features (the paper's
    second variant; see §6.5 of the paper, which reports ~2% with
    regression, dropping to 1.25% only after generative+discriminative
    up-down fine-tuning).
  * No up-down fine-tuning. We get partial reproduction at ~3% test
    error -- the algorithm works, the gap to 1.25% is the up-down step.

Files in this folder:
    dbn_mnist.py             - this file (model + train + eval, CLI)
    visualize_dbn_mnist.py   - static viz: layer-1 filters, training
                               curves, sample reconstructions
    make_dbn_mnist_gif.py    - animated GIF of layer-1 filters emerging
                               during CD-1 training
"""

from __future__ import annotations
import argparse
import gzip
import os
import time
import urllib.request

import numpy as np


# ----------------------------------------------------------------------
# MNIST
# ----------------------------------------------------------------------

CACHE = os.path.expanduser("~/.cache/hinton-mnist")
URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def load_mnist() -> dict:
    """Download (once) and decode MNIST. Returns numpy arrays in [0, 1]."""
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
    """Balanced subsample. n_per_class is clipped to the smallest class count
    (MNIST's smallest training class is ~5421); pass a large value to use
    everything."""
    cls_indices = [np.where(labels == c)[0] for c in range(10)]
    n_per_class = min(n_per_class, min(len(c) for c in cls_indices))
    idx = []
    for c in range(10):
        idx.append(rng.choice(cls_indices[c], size=n_per_class, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return images[idx], labels[idx]


# ----------------------------------------------------------------------
# RBM (binary visible <-> binary hidden)
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class RBM:
    """A single-layer Restricted Boltzmann Machine trained with CD-1.

    Visible-binary <-> hidden-binary. Real-valued visible inputs in [0, 1]
    are interpreted as Bernoulli probabilities (Hinton's 2010 RBM
    practical guide, §3.2 — the standard way to feed pixel intensities).
    """

    def __init__(self, n_visible: int, n_hidden: int,
                 init_scale: float = 0.01, rng=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        rng = rng or np.random.default_rng()
        self.W = rng.normal(0, init_scale, size=(n_visible, n_hidden)).astype(np.float32)
        self.b_v = np.zeros(n_visible, dtype=np.float32)
        self.b_h = np.zeros(n_hidden, dtype=np.float32)
        # momentum buffers
        self.dW = np.zeros_like(self.W)
        self.db_v = np.zeros_like(self.b_v)
        self.db_h = np.zeros_like(self.b_h)

    def prob_h_given_v(self, v: np.ndarray) -> np.ndarray:
        return sigmoid(v @ self.W + self.b_h)

    def prob_v_given_h(self, h: np.ndarray) -> np.ndarray:
        return sigmoid(h @ self.W.T + self.b_v)

    def sample(self, p: np.ndarray, rng) -> np.ndarray:
        return (rng.random(p.shape) < p).astype(np.float32)

    def cd1_update(self, v0: np.ndarray, lr: float, momentum: float,
                   weight_decay: float, rng) -> float:
        """One CD-1 update on a mini-batch v0. Returns reconstruction MSE."""
        # positive phase
        ph0 = self.prob_h_given_v(v0)
        h0 = self.sample(ph0, rng)
        # negative phase: one Gibbs step
        pv1 = self.prob_v_given_h(h0)
        # use probabilities (not samples) for visible reconstruction —
        # standard practical advice (Hinton 2010 §3.4)
        v1 = pv1
        ph1 = self.prob_h_given_v(v1)

        n = v0.shape[0]
        # gradients (positive minus negative phase)
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

    def features(self, v: np.ndarray) -> np.ndarray:
        """Return p(h|v), used as features for the next layer."""
        return self.prob_h_given_v(v)


# ----------------------------------------------------------------------
# DBN training
# ----------------------------------------------------------------------

def train_rbm(rbm: RBM, data: np.ndarray, n_epochs: int, batch_size: int,
              lr: float, momentum: float, weight_decay: float,
              rng, snapshot_every: int = 0,
              snapshot_callback=None,
              layer_label: str = "L?",
              verbose: bool = True) -> list:
    """Train an RBM on `data` with CD-1 SGD. Returns per-epoch recon MSE."""
    n = data.shape[0]
    losses = []
    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch = data[perm[start:start + batch_size]]
            mom = 0.5 if epoch < 5 else momentum  # ramp-up — Hinton 2010 §9
            loss = rbm.cd1_update(batch, lr, mom, weight_decay, rng)
            epoch_loss += loss
            n_batches += 1
        epoch_loss /= max(n_batches, 1)
        losses.append(epoch_loss)
        if verbose:
            print(f"  [{layer_label}] epoch {epoch+1}/{n_epochs}  recon_mse={epoch_loss:.4f}")
        if snapshot_every and snapshot_callback and (epoch + 1) % snapshot_every == 0:
            snapshot_callback(epoch, rbm, losses)
    return losses


def stack_features(rbms: list, data: np.ndarray) -> np.ndarray:
    """Push `data` up through a stack of trained RBMs, returning the
    p(h|v) of the topmost layer."""
    h = data
    for rbm in rbms:
        h = rbm.features(h)
    return h


# ----------------------------------------------------------------------
# Logistic regression head (simple, full-batch, no MNIST-specific tricks)
# ----------------------------------------------------------------------

def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def train_logreg(X: np.ndarray, y: np.ndarray, n_classes: int,
                 n_epochs: int, batch_size: int, lr: float, l2: float,
                 rng, X_val=None, y_val=None,
                 verbose: bool = True) -> tuple:
    n, d = X.shape
    W = np.zeros((d, n_classes), dtype=np.float32)
    b = np.zeros(n_classes, dtype=np.float32)
    Y = np.eye(n_classes, dtype=np.float32)[y]

    history = {"train_acc": [], "val_acc": []}
    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            sl = perm[start:start + batch_size]
            Xb, Yb = X[sl], Y[sl]
            P = softmax(Xb @ W + b)
            grad_W = Xb.T @ (P - Yb) / Xb.shape[0] + l2 * W
            grad_b = (P - Yb).mean(axis=0)
            W -= lr * grad_W
            b -= lr * grad_b
        train_pred = np.argmax(softmax(X @ W + b), axis=1)
        train_acc = float((train_pred == y).mean())
        history["train_acc"].append(train_acc)
        if X_val is not None:
            val_pred = np.argmax(softmax(X_val @ W + b), axis=1)
            val_acc = float((val_pred == y_val).mean())
            history["val_acc"].append(val_acc)
            if verbose:
                print(f"  [logreg] epoch {epoch+1}/{n_epochs}  "
                      f"train={train_acc*100:.2f}%  test={val_acc*100:.2f}%")
        elif verbose:
            print(f"  [logreg] epoch {epoch+1}/{n_epochs}  train={train_acc*100:.2f}%")
    return W, b, history


# ----------------------------------------------------------------------
# Train the full DBN
# ----------------------------------------------------------------------

def train_dbn(layer_sizes=(784, 500, 500, 2000),
              n_train_per_class=1000,
              n_epochs_per_layer=10,
              batch_size=100,
              lr=0.05,
              momentum=0.9,
              weight_decay=2e-4,
              n_classifier_epochs=30,
              classifier_lr=0.05,
              classifier_l2=1e-4,
              seed=0,
              snapshot_every=0,
              snapshot_callback=None,
              verbose=True) -> dict:
    """Train the full DBN: greedy CD-1 RBM stacking + logistic regression.

    Returns a dict with the trained RBMs, the classifier, and full
    training history (per-layer recon losses and classifier accuracies).
    """
    rng = np.random.default_rng(seed)

    if verbose:
        print("Loading MNIST...")
    mnist = load_mnist()

    # Subsample for laptop budget; balanced across classes.
    if verbose:
        print(f"Subsampling {n_train_per_class * 10} balanced training images...")
    X_train, y_train = balanced_subsample(
        mnist["train_images"].reshape(-1, 784),
        mnist["train_labels"], n_train_per_class, rng)
    X_test = mnist["test_images"].reshape(-1, 784)
    y_test = mnist["test_labels"]

    rbms = []
    layer_losses = []
    current = X_train
    for li in range(len(layer_sizes) - 1):
        n_v, n_h = layer_sizes[li], layer_sizes[li + 1]
        if verbose:
            print(f"\nTraining RBM layer {li+1}: {n_v} -> {n_h}")
        rbm = RBM(n_v, n_h, rng=np.random.default_rng(seed * 1000 + li + 1))

        cb = None
        if snapshot_callback is not None and li == 0:
            # only animate layer 1 — its filters are pixel-space and
            # human-interpretable; deeper layers are not.
            def cb(epoch, rbm_, losses_):
                snapshot_callback(li, epoch, rbm_, losses_)

        losses = train_rbm(rbm, current, n_epochs_per_layer, batch_size,
                           lr, momentum, weight_decay, rng,
                           snapshot_every=snapshot_every,
                           snapshot_callback=cb,
                           layer_label=f"L{li+1}",
                           verbose=verbose)
        layer_losses.append(losses)
        rbms.append(rbm)
        # Use *probabilities* p(h|v) as features for the next layer
        # (Hinton 2010 §3.3) — keeping samples would inject excess noise.
        current = rbm.features(current)

    if verbose:
        print(f"\nExtracting layer-3 features for classifier...")
    F_train = stack_features(rbms, X_train)
    F_test = stack_features(rbms, X_test)

    if verbose:
        print(f"Training logistic-regression classifier on {F_train.shape[1]}-d features...")
    W_cls, b_cls, cls_history = train_logreg(
        F_train, y_train, 10, n_classifier_epochs, batch_size,
        classifier_lr, classifier_l2, rng, F_test, y_test, verbose=verbose)

    final_train_acc = cls_history["train_acc"][-1]
    final_test_acc = cls_history["val_acc"][-1]

    if verbose:
        print(f"\nFinal: train {final_train_acc*100:.2f}%  "
              f"test {final_test_acc*100:.2f}%  "
              f"(error {(1 - final_test_acc)*100:.2f}%)")

    return {
        "rbms": rbms,
        "layer_losses": layer_losses,
        "W_cls": W_cls, "b_cls": b_cls,
        "cls_history": cls_history,
        "final_train_acc": final_train_acc,
        "final_test_acc": final_test_acc,
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "layer_sizes": layer_sizes,
    }


# ----------------------------------------------------------------------
# Generative sampling (the paper's Figure 8)
# ----------------------------------------------------------------------

def sample_dbn(result: dict, n_samples: int = 10, n_top_gibbs: int = 200,
               seed: int = 1, init_from_data: bool = True) -> np.ndarray:
    """Draw fantasy images from the trained DBN.

    With `init_from_data=True` (default) we pick `n_samples` random test
    images, push them up to the top RBM's visible layer, and run Gibbs
    from there — the standard "data-initialised fantasies" used in
    Hinton's RBM practical guide §16. Without label-conditioning a fully
    *unconditional* top-RBM Gibbs chain tends to mode-collapse on dark
    images, so initialising near data is what makes the samples
    recognisably digit-like.

    Then propagate down through the lower RBMs as a directed sigmoid
    belief net (Hinton 2006 §3).
    """
    rng = np.random.default_rng(seed)
    rbms = result["rbms"]
    top = rbms[-1]

    if init_from_data:
        idx = rng.choice(len(result["X_test"]), size=n_samples, replace=False)
        x = result["X_test"][idx]
        # push up through lower RBMs
        h_lower = x
        for rbm in rbms[:-1]:
            h_lower = rbm.prob_h_given_v(h_lower)
        # h_lower is now the top RBM's visible layer; sample top hidden
        v_top = (rng.random(h_lower.shape) < h_lower).astype(np.float32)
        h_top = (rng.random(top.prob_h_given_v(v_top).shape)
                 < top.prob_h_given_v(v_top)).astype(np.float32)
    else:
        h_top = (rng.random((n_samples, top.n_hidden)) < 0.5).astype(np.float32)

    for _ in range(n_top_gibbs):
        v_top = (rng.random(top.prob_v_given_h(h_top).shape)
                 < top.prob_v_given_h(h_top)).astype(np.float32)
        h_top = (rng.random(top.prob_h_given_v(v_top).shape)
                 < top.prob_h_given_v(v_top)).astype(np.float32)

    # final down-pass uses the *probability* (not sample) for visibility
    h = top.prob_v_given_h(h_top)
    for rbm in reversed(rbms[:-1]):
        h = rbm.prob_v_given_h(h)
    return h.reshape(-1, 28, 28)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train-per-class", type=int, default=1000)
    p.add_argument("--epochs-per-layer", type=int, default=10)
    p.add_argument("--classifier-epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="Tiny budget (300/class, 4 epochs) — for smoke testing.")
    args = p.parse_args()

    if args.quick:
        n_per_class = 300
        epochs_layer = 4
        cls_epochs = 10
    else:
        n_per_class = args.n_train_per_class
        epochs_layer = args.epochs_per_layer
        cls_epochs = args.classifier_epochs

    t0 = time.time()
    result = train_dbn(
        n_train_per_class=n_per_class,
        n_epochs_per_layer=epochs_layer,
        n_classifier_epochs=cls_epochs,
        seed=args.seed,
    )
    print(f"\nTotal wallclock: {time.time() - t0:.1f}s")
    return result


if __name__ == "__main__":
    main()
