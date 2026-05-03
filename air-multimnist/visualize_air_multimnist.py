"""Static visualizations for AIR Multi-MNIST.

Outputs (in `viz/`):
  example_scenes.png         - 8 example training scenes with ground-truth count
  attention_steps.png        - per-step attention boxes overlaid on val scenes
  reconstructions.png        - input | recon | per-step canvases | residual
  count_distribution.png     - prediction confusion (true count -> predicted count)
  training_curves.png        - recon loss / KL terms / count accuracy across epochs
  per_step_patches.png       - decoded 16x16 patches for each step + their renders
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from air_multimnist import AIR, train, env_info


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def zwhere_to_box(z_where: np.ndarray, canvas_size: int):
    """Convert z_where=(log_s, tx, ty) (normalized in [-1, 1]) into a pixel box.

    The patch covers the canvas region [tx-s, tx+s] x [ty-s, ty+s] in
    normalized coords. Convert back to pixels: x_norm in [-1, 1] maps to
    pixel j in [0, W-1] via j = (x_norm + 1) * (W - 1) / 2.
    """
    log_s, tx, ty = z_where
    s = float(np.exp(np.clip(log_s, -3.0, 1.5)))
    W = canvas_size
    def to_pix(u):
        return (u + 1.0) * (W - 1) / 2.0
    x0 = to_pix(tx - s); x1 = to_pix(tx + s)
    y0 = to_pix(ty - s); y1 = to_pix(ty + s)
    return x0, y0, x1 - x0, y1 - y0  # (left, top, width, height)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_example_scenes(scenes: np.ndarray, counts: np.ndarray,
                        out_path: str, n: int = 8):
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.4), dpi=110)
    for i in range(n):
        axes[i].imshow(scenes[i], cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(f"n={counts[i]}", fontsize=10)
        axes[i].set_xticks([]); axes[i].set_yticks([])
    fig.suptitle("Multi-MNIST scenes (32x32, 0-2 digits each at random scale & position)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_attention_steps(model: AIR, scenes: np.ndarray, counts: np.ndarray,
                         out_path: str, n: int = 6, threshold: float = 0.5):
    """Show original scene with per-step attention boxes; box color = z_pres."""
    out = model.parse_scene(scenes[:n], threshold=threshold)
    cum_pres = out["cum_pres"]   # (n, T)
    pred_count = out["count"]
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.8), dpi=110)
    if n == 1:
        axes = [axes]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i in range(n):
        ax = axes[i]
        ax.imshow(scenes[i], cmap="gray", vmin=0, vmax=1)
        for t in range(model.max_steps):
            z_where = out["per_step"][t]["z_where"][i]
            cp = float(cum_pres[i, t])
            x, y, w, h = zwhere_to_box(z_where, model.canvas_size)
            # alpha = cum_pres so off slots fade
            alpha = max(0.15, min(1.0, cp))
            edge = colors[t]
            rect = Rectangle((x, y), w, h, fill=False, edgecolor=edge,
                             linewidth=2.2 if cp > threshold else 0.9,
                             linestyle="-" if cp > threshold else "--",
                             alpha=alpha)
            ax.add_patch(rect)
        ax.set_title(f"true={int(counts[i])}  pred={int(pred_count[i])}\n"
                     f"cum_pres={[f'{c:.2f}' for c in cum_pres[i]]}",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    # legend (color = step)
    handles = [Rectangle((0, 0), 1, 1, fill=False, edgecolor=c, linewidth=2)
               for c in colors[:model.max_steps]]
    labels = [f"step {t}" for t in range(model.max_steps)]
    fig.legend(handles, labels, loc="lower center", ncol=model.max_steps,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("AIR per-step attention (solid = active, dashed = z_pres < 0.5)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_reconstructions(model: AIR, scenes: np.ndarray, counts: np.ndarray,
                         out_path: str, n: int = 6):
    out = model.parse_scene(scenes[:n])
    recon = out["recon"]
    fig, axes = plt.subplots(n, 2 + model.max_steps + 1,
                             figsize=(2.0 * (2 + model.max_steps + 1), 2.0 * n),
                             dpi=110)
    for i in range(n):
        # input
        axes[i, 0].imshow(scenes[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_ylabel(f"true={int(counts[i])}", fontsize=10)
        if i == 0: axes[i, 0].set_title("input", fontsize=10)
        # recon
        axes[i, 1].imshow(recon[i], cmap="gray", vmin=0, vmax=1)
        if i == 0: axes[i, 1].set_title(f"recon (pred={int(out['count'][i])})", fontsize=10)
        # per-step canvases (effective contribution, scaled by cum_pres)
        for t in range(model.max_steps):
            ct = out["per_step"][t]["canvas_t"][i]
            cp = float(out["cum_pres"][i, t])
            shown = ct * cp
            ax = axes[i, 2 + t]
            ax.imshow(shown, cmap="gray", vmin=0, vmax=1)
            if i == 0:
                ax.set_title(f"step {t}", fontsize=10)
            ax.set_xlabel(f"cum_pres={cp:.2f}", fontsize=8)
        # residual
        ax = axes[i, 2 + model.max_steps]
        residual = scenes[i] - recon[i]
        ax.imshow(residual, cmap="seismic", vmin=-1, vmax=1)
        if i == 0: ax.set_title("residual", fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("AIR reconstructions: per-step contributions sum to the prediction",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_count_distribution(model: AIR, scenes: np.ndarray, counts: np.ndarray,
                            out_path: str):
    out = model.parse_scene(scenes)
    pred = out["count"]
    max_c = max(int(counts.max()), int(pred.max())) + 1
    confusion = np.zeros((max_c, max_c), dtype=np.int64)
    for t, p in zip(counts, pred):
        confusion[int(t), int(p)] += 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=110)
    ax = axes[0]
    im = ax.imshow(confusion, cmap="Blues", origin="lower")
    for i in range(max_c):
        for j in range(max_c):
            txt = ax.text(j, i, str(confusion[i, j]), ha="center", va="center",
                          color="white" if confusion[i, j] > confusion.max() / 2 else "black",
                          fontsize=10)
    ax.set_xlabel("predicted count")
    ax.set_ylabel("true count")
    ax.set_xticks(range(max_c)); ax.set_yticks(range(max_c))
    ax.set_title("Count confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # histogram of per-class accuracy
    ax2 = axes[1]
    accs = []
    labels_x = []
    for c in range(max_c):
        mask = counts == c
        if mask.sum() == 0:
            continue
        accs.append(float(np.mean(pred[mask] == c)))
        labels_x.append(f"n={c}\n({mask.sum()})")
    bars = ax2.bar(range(len(accs)), accs, color="#2ca02c")
    ax2.axhline(1.0 / 3, color="gray", linestyle=":", label=f"chance (1/3 = {1/3:.2f})")
    ax2.axhline(0.5, color="black", linestyle="--", label="target 0.5")
    ax2.set_xticks(range(len(accs)))
    ax2.set_xticklabels(labels_x)
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.set_title(f"Per-class count accuracy (overall = {float(np.mean(pred==counts)):.3f})")
    ax2.legend(loc="lower right", fontsize=9)
    for i, a in enumerate(accs):
        ax2.text(i, a + 0.02, f"{a:.2f}", ha="center", fontsize=9)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_training_curves(history: dict, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=110)
    ax = axes[0]
    ax.plot(history["epoch"], history["recon"], label="recon", color="#1f77b4")
    ax.set_xlabel("epoch"); ax.set_ylabel("recon loss (per-image MSE x 1024)")
    ax.set_title("Reconstruction loss")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(history["epoch"], history["kl_what"], label="KL(z_what)", color="#ff7f0e")
    ax.plot(history["epoch"], history["kl_pres"], label="KL(z_pres)", color="#d62728")
    ax.set_xlabel("epoch"); ax.set_ylabel("KL")
    ax.set_title("KL terms")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(history["epoch"], history["val_count_acc"], color="#2ca02c", marker="o")
    ax.axhline(1.0 / 3, color="gray", linestyle=":", label=f"chance ({1/3:.3f})")
    ax.axhline(0.5, color="black", linestyle="--", label="target 0.5")
    if "best_count_acc" in history:
        ax.axhline(history["best_count_acc"], color="#2ca02c", linestyle=":",
                   alpha=0.6, label=f"best ({history['best_count_acc']:.3f})")
    ax.set_xlabel("epoch"); ax.set_ylabel("val count accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Count accuracy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_per_step_patches(model: AIR, scenes: np.ndarray, out_path: str,
                          n: int = 4):
    """Show each step's decoded patch (16x16) and where it lands on the canvas."""
    out = model.parse_scene(scenes[:n])
    fig, axes = plt.subplots(n, 1 + 2 * model.max_steps,
                             figsize=(1.8 * (1 + 2 * model.max_steps), 2.0 * n),
                             dpi=110)
    for i in range(n):
        axes[i, 0].imshow(scenes[i], cmap="gray", vmin=0, vmax=1)
        if i == 0: axes[i, 0].set_title("input", fontsize=9)
        for t in range(model.max_steps):
            patch = out["per_step"][t]["patch"][i]    # (16, 16)
            canvas_t = out["per_step"][t]["canvas_t"][i]  # (32, 32)
            cp = float(out["cum_pres"][i, t])
            ax_p = axes[i, 1 + 2 * t]
            ax_c = axes[i, 1 + 2 * t + 1]
            ax_p.imshow(patch, cmap="gray", vmin=0, vmax=1)
            ax_c.imshow(canvas_t * cp, cmap="gray", vmin=0, vmax=1)
            if i == 0:
                ax_p.set_title(f"patch[{t}]", fontsize=9)
                ax_c.set_title(f"render[{t}]", fontsize=9)
            ax_c.set_xlabel(f"cp={cp:.2f}", fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("AIR per-step decoded patches (16x16) and their spatial-transformer renders",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-epochs", type=int, default=8)
    ap.add_argument("--out-dir", type=str, default=str(Path(__file__).parent / "viz"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"# environment: {env_info()}")
    print(f"# training AIR (n_train={args.n_train}, n_epochs={args.n_epochs})...")
    model, history, td, vd = train(n_train=args.n_train, n_val=args.n_val,
                                   n_epochs=args.n_epochs, seed=args.seed,
                                   verbose=True)

    print(f"\n# rendering visualizations to {out_dir}/")
    plot_example_scenes(td["scenes"], td["counts"],
                        str(out_dir / "example_scenes.png"))
    plot_attention_steps(model, vd["scenes"], vd["counts"],
                         str(out_dir / "attention_steps.png"))
    plot_reconstructions(model, vd["scenes"], vd["counts"],
                         str(out_dir / "reconstructions.png"))
    plot_count_distribution(model, vd["scenes"], vd["counts"],
                            str(out_dir / "count_distribution.png"))
    plot_training_curves(history, str(out_dir / "training_curves.png"))
    plot_per_step_patches(model, vd["scenes"],
                          str(out_dir / "per_step_patches.png"))
    print(f"\nDone. best_count_acc={history['best_count_acc']:.3f} "
          f"at epoch {history['best_epoch']}")


if __name__ == "__main__":
    main()
