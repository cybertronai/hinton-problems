"""Render an animated GIF of AIR learning to attend, infer, and count.

Each frame shows:
  - Top row: 4 fixed validation scenes with per-step attention boxes
  - Middle row: AIR reconstructions
  - Bottom row: per-step decoded patches (one column per step)
  - Footer: training curves so far (recon loss + count accuracy)

Usage:
    python3 make_air_multimnist_gif.py
    python3 make_air_multimnist_gif.py --n-epochs 8 --snapshot-every 25 --fps 8
"""
from __future__ import annotations
import argparse
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from air_multimnist import (
    AIR, train, generate_scenes, load_mnist, env_info,
)
from visualize_air_multimnist import zwhere_to_box


# ----------------------------------------------------------------------
# Frame rendering
# ----------------------------------------------------------------------

def _fixed_demo(n: int = 4, seed: int = 7, canvas: int = 32):
    images, labels = load_mnist("train")
    rng = np.random.default_rng(seed)
    scenes, counts, _ = generate_scenes(n, images, labels, canvas, rng=rng)
    return scenes, counts


STEP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def render_frame(model: AIR, history: dict, step: int,
                 scenes: np.ndarray, counts: np.ndarray) -> Image.Image:
    n = scenes.shape[0]
    out = model.parse_scene(scenes)
    recon = out["recon"]

    T = model.max_steps
    fig = plt.figure(figsize=(11, 7), dpi=110)
    gs = fig.add_gridspec(3 + 1, max(n, T + 1),
                          height_ratios=[1.4, 1.4, 1.4, 1.6],
                          hspace=0.35, wspace=0.2)

    # Row 0: input scenes with per-step attention boxes
    for i in range(n):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(scenes[i], cmap="gray", vmin=0, vmax=1)
        for t in range(T):
            zw = out["per_step"][t]["z_where"][i]
            cp = float(out["cum_pres"][i, t])
            x, y, w, h = zwhere_to_box(zw, model.canvas_size)
            alpha = max(0.15, min(1.0, cp))
            rect = Rectangle((x, y), w, h, fill=False,
                             edgecolor=STEP_COLORS[t],
                             linewidth=2.0 if cp > 0.5 else 0.8,
                             linestyle="-" if cp > 0.5 else "--",
                             alpha=alpha)
            ax.add_patch(rect)
        ax.set_title(f"true={int(counts[i])} pred={int(out['count'][i])}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # Row 1: reconstructions
    for i in range(n):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(recon[i], cmap="gray", vmin=0, vmax=1)
        if i == 0:
            ax.set_ylabel("recon", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # Row 2: per-step decoded patches (using example 0 only)
    for t in range(T):
        ax = fig.add_subplot(gs[2, t])
        patch = out["per_step"][t]["patch"][0]
        cp = float(out["cum_pres"][0, t])
        ax.imshow(patch, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"step {t} patch (cp={cp:.2f})",
                     fontsize=9, color=STEP_COLORS[t])
        ax.set_xticks([]); ax.set_yticks([])
    # last column: an extra blank slot for layout (or could put summary)
    for t in range(T, max(n, T + 1)):
        ax = fig.add_subplot(gs[2, t])
        ax.axis("off")

    # Row 3: training curves
    ax_loss = fig.add_subplot(gs[3, :max(n, T + 1) // 2])
    ax_acc = fig.add_subplot(gs[3, max(n, T + 1) // 2:])

    if history["step"]:
        ax_loss.plot(history["step"], history["recon"],
                     color="#1f77b4", label="recon")
        ax_loss.set_xlabel("step")
        ax_loss.set_ylabel("recon loss")
        ax_loss.set_title("Reconstruction loss")
        ax_loss.grid(alpha=0.3)
        ax_loss.legend(fontsize=8)

        ax_acc.plot(history["step"], history["val_count_acc"],
                    color="#2ca02c", marker="o", markersize=3,
                    label="val count acc")
        ax_acc.axhline(1 / 3, color="gray", linestyle=":",
                       label=f"chance ({1/3:.2f})")
        ax_acc.axhline(0.5, color="black", linestyle="--",
                       label="target 0.5")
        ax_acc.set_ylim(0, 1.0)
        ax_acc.set_xlabel("step")
        ax_acc.set_ylabel("count accuracy")
        ax_acc.set_title("Validation count accuracy")
        ax_acc.grid(alpha=0.3)
        ax_acc.legend(fontsize=8, loc="lower right")
    else:
        ax_loss.axis("off")
        ax_acc.axis("off")

    fig.suptitle(f"AIR Multi-MNIST -- step {step}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128)
    return img


# ----------------------------------------------------------------------
# Training with snapshots
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--n-epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--snapshot-every", type=int, default=25)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--out-path", type=str,
                    default=str(Path(__file__).parent / "air_multimnist.gif"))
    args = ap.parse_args()

    print(f"# environment: {env_info()}")
    demo_scenes, demo_counts = _fixed_demo(n=4, seed=args.seed + 99)

    frames = []
    snapshot_history = {"step": [], "recon": [], "val_count_acc": []}

    def snapshot_callback(step, model, history, _train_demo, val_scenes, val_counts):
        snapshot_history["step"].append(step)
        # Use most recent epoch's recon (since we don't get per-step values from
        # the training loop, just use the last epoch summary if we have one).
        # Recon recorded per-epoch in history; we'll align with current step.
        if history["step"]:
            snapshot_history["recon"].append(history["recon"][-1])
            snapshot_history["val_count_acc"].append(history["val_count_acc"][-1])
        else:
            # before first epoch end - run a quick val check
            from air_multimnist import _count_accuracy
            snapshot_history["recon"].append(0.0)
            snapshot_history["val_count_acc"].append(
                _count_accuracy(model, val_scenes, val_counts))
        frames.append(render_frame(model, snapshot_history, step,
                                   demo_scenes, demo_counts))
        if len(frames) % 10 == 0:
            print(f"  snapshot {len(frames)} at step {step}", flush=True)

    print(f"# training with snapshot_every={args.snapshot_every}...")
    model, history, _, _ = train(
        n_train=args.n_train, n_val=args.n_val,
        n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr,
        seed=args.seed, verbose=True,
        snapshot_callback=snapshot_callback, snapshot_every=args.snapshot_every,
    )
    # Final frame uses best-restored model + full per-step history
    final_history = dict(step=history["step"], recon=history["recon"],
                         val_count_acc=history["val_count_acc"])
    frames.append(render_frame(model, final_history, history["step"][-1],
                               demo_scenes, demo_counts))
    # hold final frame for emphasis
    frames.extend([frames[-1]] * 6)

    print(f"# saving {len(frames)} frames -> {args.out_path}")
    duration_ms = int(1000 / args.fps)
    frames[0].save(args.out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(args.out_path) / 1024
    print(f"# done. gif size: {size_kb:.1f} KB")
    print(f"# best epoch: {history['best_epoch']}, "
          f"best count_acc: {history['best_count_acc']:.3f}")


if __name__ == "__main__":
    main()
