"""
Static visualizations for the trained 784-500-500-2000 DBN on MNIST.

Outputs (in `viz/`):
  layer1_filters.png       - 12x12 gallery of layer-1 receptive fields
                             (rows of W reshaped to 28x28).
  training_curves.png      - per-layer CD-1 reconstruction MSE +
                             classifier accuracy.
  reconstructions.png      - sample test digits and their up-down
                             reconstructions through the 3-RBM stack.
  generated_samples.png    - digits drawn from the trained DBN by
                             alternating Gibbs at the top RBM and
                             propagating down.
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from dbn_mnist import (train_dbn, sample_dbn, stack_features, sigmoid)


def _normalize_for_display(W: np.ndarray) -> np.ndarray:
    """Per-filter zero-centred scale to [0, 1] for imshow."""
    n = W.shape[1]
    out = np.empty((n, W.shape[0]))
    for i in range(n):
        w = W[:, i]
        m = np.abs(w).max() + 1e-9
        out[i] = (w / m + 1) / 2  # → [0,1]
    return out


def plot_layer1_filters(rbm, out_path: str, ncols: int = 12, nrows: int = 12):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 0.7, nrows * 0.7),
                             dpi=120)
    norm = _normalize_for_display(rbm.W)
    for i, ax in enumerate(axes.flat):
        if i < norm.shape[0]:
            ax.imshow(norm[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Layer-1 RBM filters (rows of W reshaped to 28×28)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_training_curves(result: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)

    ax = axes[0]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    for i, losses in enumerate(result["layer_losses"]):
        ax.plot(range(1, len(losses) + 1), losses, color=colors[i],
                marker="o", markersize=3, linewidth=1.2,
                label=f"L{i+1}: {result['layer_sizes'][i]}→{result['layer_sizes'][i+1]}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("CD-1 reconstruction MSE")
    ax.set_title("Per-layer RBM pretraining")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")

    ax = axes[1]
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
    ax.set_title(f"Logistic regression on layer-3 features  (final test: {final:.2f}%)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_reconstructions(result: dict, out_path: str, n: int = 8,
                         seed: int = 7):
    """Push test digits up the RBM stack, then back down — visualizes
    what the layer-3 representation has compressed away."""
    rng = np.random.default_rng(seed)
    rbms = result["rbms"]

    idx = rng.choice(len(result["X_test"]), size=n, replace=False)
    v = result["X_test"][idx]

    # up
    h = v
    for rbm in rbms:
        h = rbm.prob_h_given_v(h)
    # down (deterministic prob_v_given_h, no sampling)
    for rbm in reversed(rbms):
        h = rbm.prob_v_given_h(h)
    recon = h.reshape(-1, 28, 28)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.2, 2.6), dpi=140)
    for i in range(n):
        axes[0, i].imshow(v[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(recon[i], cmap="gray", vmin=0, vmax=1)
        for ax in (axes[0, i], axes[1, i]):
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("input", fontsize=9)
    axes[1, 0].set_ylabel("up→down\nrecon", fontsize=9)
    fig.suptitle("Up-down reconstruction through the 3-RBM stack",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_generated_samples(result: dict, out_path: str, n: int = 16,
                           n_top_gibbs: int = 10, seed: int = 11):
    samples = sample_dbn(result, n_samples=n,
                         n_top_gibbs=n_top_gibbs, seed=seed,
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
    fig.suptitle(f"Generative samples ({n_top_gibbs} top-RBM Gibbs steps, "
                 "then deterministic down-pass)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train-per-class", type=int, default=1000)
    p.add_argument("--epochs-per-layer", type=int, default=10)
    p.add_argument("--classifier-epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Training DBN (seed={args.seed})...")
    result = train_dbn(
        n_train_per_class=args.n_train_per_class,
        n_epochs_per_layer=args.epochs_per_layer,
        n_classifier_epochs=args.classifier_epochs,
        seed=args.seed,
    )

    plot_layer1_filters(result["rbms"][0],
                        os.path.join(args.outdir, "layer1_filters.png"))
    plot_training_curves(result,
                         os.path.join(args.outdir, "training_curves.png"))
    plot_reconstructions(result,
                         os.path.join(args.outdir, "reconstructions.png"))
    plot_generated_samples(result,
                           os.path.join(args.outdir, "generated_samples.png"))


if __name__ == "__main__":
    main()
