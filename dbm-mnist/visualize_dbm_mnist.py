"""
Static visualizations for the trained 784-500-1000 DBM.

Outputs (in `viz/`):
  layer1_filters.png       - 12x12 gallery of layer-1 receptive fields.
  training_curves.png      - per-layer pretraining recon MSE +
                             joint-PCD recon MSE + classifier accuracy.
  mean_field_iterations.png- per-iteration evolution of mu1 on test
                             digits (the DBM's defining inference step).
  reconstructions.png      - test digits and their up-down recons.
  generated_samples.png    - DBM Gibbs samples after data-init burn-in.
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from dbm_mnist import train_dbm, sample_dbm, sigmoid


def _normalize(W: np.ndarray) -> np.ndarray:
    out = np.empty((W.shape[1], W.shape[0]))
    for i in range(W.shape[1]):
        w = W[:, i]
        m = np.abs(w).max() + 1e-9
        out[i] = (w / m + 1) / 2
    return out


def plot_layer1_filters(dbm, out_path: str, ncols=12, nrows=12):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 0.7, nrows * 0.7),
                             dpi=120)
    norm = _normalize(dbm.W1)
    for i, ax in enumerate(axes.flat):
        if i < norm.shape[0]:
            ax.imshow(norm[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Layer-1 DBM filters (rows of W1 reshaped to 28×28)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_training_curves(result, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=120)

    ax = axes[0]
    losses1, losses2 = result["pretrain_losses"]
    ax.plot(range(1, len(losses1) + 1), losses1, "o-", color="#1f77b4",
            label=f"L1*: 784→{result['layer_sizes'][1]} (bottom-doubled)",
            markersize=3)
    ax.plot(range(1, len(losses2) + 1), losses2, "o-", color="#2ca02c",
            label=f"L2*: {result['layer_sizes'][1]}→{result['layer_sizes'][2]} (top-doubled)",
            markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("CD-1 reconstruction MSE")
    ax.set_yscale("log")
    ax.set_title("Pretraining (greedy doubled-RBM)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    j = result["joint_losses"]
    if j:
        ax.plot(range(1, len(j) + 1), j, "o-", color="#9467bd", markersize=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel("recon MSE (mean-field → p(v|h1))")
        ax.set_title("Joint PCD")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "(joint training disabled)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    ax = axes[2]
    h = result["cls_history"]
    ax.plot(range(1, len(h["train_acc"]) + 1),
            np.array(h["train_acc"]) * 100,
            color="#1f77b4", label="train", linewidth=1.5)
    ax.plot(range(1, len(h["val_acc"]) + 1),
            np.array(h["val_acc"]) * 100,
            color="#d62728", label="test", linewidth=1.5)
    final = h["val_acc"][-1] * 100
    ax.axhline(final, color="#d62728", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"Logreg on [h1, h2]  (final test: {final:.2f}%)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_mean_field_iterations(result, out_path: str, n: int = 6,
                               iters_to_show=(0, 1, 2, 5, 10, 20),
                               seed: int = 3):
    """Watch mean-field activations relax across iterations on a few
    test digits."""
    rng = np.random.default_rng(seed)
    dbm = result["dbm"]
    idx = rng.choice(len(result["X_test"]), size=n, replace=False)
    v = result["X_test"][idx]

    snapshots = []
    mu1 = sigmoid(v @ dbm.W1 + dbm.b_h1)
    mu2 = sigmoid(np.zeros((v.shape[0], dbm.n_h2), dtype=np.float32))
    snapshots.append(("init", mu1.copy()))
    for k in range(1, max(iters_to_show) + 1):
        mu2 = sigmoid(mu1 @ dbm.W2 + dbm.b_h2)
        mu1 = sigmoid(v @ dbm.W1 + mu2 @ dbm.W2.T + dbm.b_h1)
        if k in iters_to_show:
            snapshots.append((f"iter {k}", mu1.copy()))

    n_iters = len(snapshots)
    fig, axes = plt.subplots(n + 1, n_iters, figsize=(n_iters * 1.0, (n + 1) * 1.0),
                             dpi=140)

    # top row: input digits
    for j in range(n_iters):
        if j < n:
            axes[0, j].imshow(v[j].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        else:
            axes[0, j].axis("off")
        axes[0, j].set_xticks([])
        axes[0, j].set_yticks([])
    axes[0, 0].set_ylabel("input", fontsize=9)

    # remaining rows: each test digit's mu1 across iterations
    # but mu1 is 500-d — display as a 22x23 reshape (≈ 500 = 22.36²)
    side = int(np.ceil(np.sqrt(dbm.n_h1)))
    pad = side * side - dbm.n_h1
    for r in range(n):
        for j in range(n_iters):
            label, mu1_snap = snapshots[j]
            grid = np.concatenate([mu1_snap[r], np.zeros(pad, dtype=np.float32)])
            axes[r + 1, j].imshow(grid.reshape(side, side),
                                  cmap="viridis", vmin=0, vmax=1)
            axes[r + 1, j].set_xticks([])
            axes[r + 1, j].set_yticks([])
            if r == n - 1:
                axes[r + 1, j].set_xlabel(label, fontsize=8)
        axes[r + 1, 0].set_ylabel(f"digit {result['y_test'][idx[r]]}", fontsize=8)

    fig.suptitle("Mean-field iterations on h1 (DBM's defining inference step)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_reconstructions(result, out_path: str, n: int = 8, seed: int = 7):
    rng = np.random.default_rng(seed)
    dbm = result["dbm"]
    idx = rng.choice(len(result["X_test"]), size=n, replace=False)
    v = result["X_test"][idx]
    mu1, _ = dbm.mean_field(v, n_iters=20)
    recon = sigmoid(mu1 @ dbm.W1.T + dbm.b_v).reshape(-1, 28, 28)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.2, 2.6), dpi=140)
    for i in range(n):
        axes[0, i].imshow(v[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(recon[i], cmap="gray", vmin=0, vmax=1)
        for ax in (axes[0, i], axes[1, i]):
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("input", fontsize=9)
    axes[1, 0].set_ylabel("recon\n(MF→p(v|h1))", fontsize=9)
    fig.suptitle("Mean-field reconstruction through the DBM",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_generated_samples(result, out_path: str, n: int = 16,
                           n_gibbs: int = 50, seed: int = 11):
    samples = sample_dbm(result, n_samples=n,
                         n_gibbs=n_gibbs, seed=seed,
                         init_from_data=True)
    nrows = 2
    ncols = (n + nrows - 1) // nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.0, nrows * 1.05),
                             dpi=140)
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(samples[i], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"DBM samples ({n_gibbs} Gibbs steps from data-init)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train-per-class", type=int, default=1000)
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--joint-epochs", type=int, default=5)
    p.add_argument("--classifier-epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Training DBM (seed={args.seed})...")
    result = train_dbm(
        n_train_per_class=args.n_train_per_class,
        n_pretrain_epochs=args.pretrain_epochs,
        n_joint_epochs=args.joint_epochs,
        n_classifier_epochs=args.classifier_epochs,
        seed=args.seed,
    )

    plot_layer1_filters(result["dbm"],
                        os.path.join(args.outdir, "layer1_filters.png"))
    plot_training_curves(result,
                         os.path.join(args.outdir, "training_curves.png"))
    plot_mean_field_iterations(result,
                               os.path.join(args.outdir, "mean_field_iterations.png"))
    plot_reconstructions(result,
                         os.path.join(args.outdir, "reconstructions.png"))
    plot_generated_samples(result,
                           os.path.join(args.outdir, "generated_samples.png"))


if __name__ == "__main__":
    main()
