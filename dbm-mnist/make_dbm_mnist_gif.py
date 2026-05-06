"""
Animated GIF of layer-1 DBM filters across the full training pipeline:
  pretraining (CD-1 with bottom-doubling) -> joint PCD with mean-field
  positive phase.

Frames are tagged with the phase ("pretrain L1" vs "joint PCD") so the
visual transition between greedy pretraining and joint training is
visible.
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from dbm_mnist import (load_mnist, balanced_subsample, _RBM, train_rbm, DBM)
from visualize_dbm_mnist import _normalize


def render_frame(W1: np.ndarray, label: str, ncols=12, nrows=12) -> Image.Image:
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 0.55, nrows * 0.55),
                             dpi=100)
    norm = _normalize(W1)
    for i, ax in enumerate(axes.flat):
        if i < norm.shape[0]:
            ax.imshow(norm[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(label, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train-per-class", type=int, default=1000)
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--joint-epochs", type=int, default=5)
    p.add_argument("--snapshot-every", type=int, default=1)
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="dbm_mnist.gif")
    p.add_argument("--hold-final", type=int, default=8)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Loading MNIST...")
    mnist = load_mnist()
    X_train, y_train = balanced_subsample(
        mnist["train_images"].reshape(-1, 784),
        mnist["train_labels"], args.n_train_per_class, rng)

    n_v, n_h1, n_h2 = 784, 500, 1000

    # Custom pretraining loop with snapshot per epoch on layer 1
    rbm1 = _RBM(n_v, n_h1, bottom_doubling=True,
                rng=np.random.default_rng(args.seed * 1000 + 1))
    frames = []
    print("\nPretraining bottom RBM (with snapshots)...")
    losses1 = []
    for ep in range(args.pretrain_epochs):
        perm = rng.permutation(len(X_train))
        for s in range(0, len(X_train), 100):
            mom = 0.5 if ep < 5 else 0.9
            rbm1.cd1_update(X_train[perm[s:s+100]], 0.05, mom, 2e-4, rng)
        if (ep + 1) % args.snapshot_every == 0:
            label = f"pretrain L1*  epoch {ep+1}/{args.pretrain_epochs}"
            frames.append(render_frame(rbm1.W, label))
            print(f"  frame {len(frames):3d}  {label}")

    # Pretrain layer 2 (no snapshots — these aren't pixel-space)
    rbm1_solo = _RBM(n_v, n_h1)
    rbm1_solo.W = rbm1.W.copy(); rbm1_solo.b_h = rbm1.b_h.copy()
    F1 = rbm1_solo.prob_h_given_v(X_train)
    rbm2 = _RBM(n_h1, n_h2, top_doubling=True,
                rng=np.random.default_rng(args.seed * 1000 + 2))
    print("\nPretraining top RBM (no snapshots; not pixel-space)...")
    train_rbm(rbm2, F1, args.pretrain_epochs, 100, 0.05, 0.9, 2e-4, rng,
              label="L2*", verbose=False)

    # Stitch into DBM
    dbm = DBM(n_v, n_h1, n_h2, rng=rng)
    dbm.init_from_pretrained(rbm1, rbm2)
    label = "after halve+stitch (start of joint phase)"
    frames.append(render_frame(dbm.W1, label))
    print(f"  frame {len(frames):3d}  {label}")

    # Joint PCD with snapshots
    print("\nJoint PCD (with snapshots)...")
    init_idx = rng.choice(len(X_train), size=100, replace=False)
    v_f = X_train[init_idx].copy()
    mu1_init, mu2_init = dbm.mean_field(v_f, n_iters=5)
    h1_f = (rng.random(mu1_init.shape) < mu1_init).astype(np.float32)
    h2_f = (rng.random(mu2_init.shape) < mu2_init).astype(np.float32)
    fantasy = (v_f, h1_f, h2_f)

    for ep in range(args.joint_epochs):
        perm = rng.permutation(len(X_train))
        for s in range(0, len(X_train), 100):
            v_batch = X_train[perm[s:s+100]]
            fantasy, _ = dbm.joint_update(v_batch, fantasy, 0.001, 0.0, 2e-4, 5, rng)
        if (ep + 1) % args.snapshot_every == 0:
            label = f"joint PCD  epoch {ep+1}/{args.joint_epochs}"
            frames.append(render_frame(dbm.W1, label))
            print(f"  frame {len(frames):3d}  {label}")

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
