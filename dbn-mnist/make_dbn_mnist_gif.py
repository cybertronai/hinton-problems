"""
Animated GIF of layer-1 RBM filters emerging during CD-1 training.

Layer 1's filters live in pixel space (28x28) and are
human-interpretable. Deeper layers operate on previous-layer features
and don't have a meaningful pixel-space rendering, so we only animate
layer 1.

Usage:
    python3 make_dbn_mnist_gif.py
    python3 make_dbn_mnist_gif.py --n-train-per-class 1000 --epochs-per-layer 10 \
                                  --snapshot-every 1 --fps 6
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from dbn_mnist import train_dbn
from visualize_dbn_mnist import _normalize_for_display


def render_frame(rbm, epoch: int, layer_loss: float,
                 ncols: int = 12, nrows: int = 12) -> Image.Image:
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 0.55, nrows * 0.55),
                             dpi=100)
    norm = _normalize_for_display(rbm.W)
    for i, ax in enumerate(axes.flat):
        if i < norm.shape[0]:
            ax.imshow(norm[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Layer-1 filters — epoch {epoch + 1}  "
                 f"(recon MSE = {layer_loss:.4f})",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train-per-class", type=int, default=1000)
    p.add_argument("--epochs-per-layer", type=int, default=10)
    p.add_argument("--snapshot-every", type=int, default=1)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="dbn_mnist.gif")
    p.add_argument("--hold-final", type=int, default=8)
    args = p.parse_args()

    frames = []

    def cb(layer_idx, epoch, rbm, losses):
        if layer_idx == 0:
            frame = render_frame(rbm, epoch, losses[-1])
            frames.append(frame)
            print(f"  frame {len(frames):3d}  L1 epoch {epoch + 1}  "
                  f"recon={losses[-1]:.4f}")

    print(f"Training DBN with snapshots every {args.snapshot_every} epoch(s)...")
    result = train_dbn(
        n_train_per_class=args.n_train_per_class,
        n_epochs_per_layer=args.epochs_per_layer,
        n_classifier_epochs=10,  # we animate L1 only; classifier need not run long
        seed=args.seed,
        snapshot_every=args.snapshot_every,
        snapshot_callback=cb,
        verbose=True,
    )
    print(f"Final test acc: {result['final_test_acc']*100:.2f}%")

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 60)
    out_path = args.out
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nWrote {out_path}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
