"""Build ``air_3d_primitives.gif``: an animation of the AIR-3D inference
network learning to invert the renderer.

Each frame shows, for a held-out test scene:
- left:   input image (ground-truth render)
- middle: re-render of the inference network's *current* prediction
- right:  absolute pixel difference

The training loop is tweaked here to take a snapshot every few epochs so we
can render a clean training-progress animation.

Usage:
    python3 make_air_3d_primitives_gif.py --seed 0 --n-epochs 40 \
        --snapshot-every 2 --fps 6 --out air_3d_primitives.gif
"""

from __future__ import annotations

import argparse
import io
import os
from copy import deepcopy

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from air_3d_primitives import (
    AIR3DEncoder,
    PRIMITIVE_TYPES,
    Primitive,
    backward,
    build_air_model_3d,
    compute_loss,
    generate_dataset,
    render_3d_scene,
)


def _snapshot_train(model: AIR3DEncoder, dataset: dict, n_epochs: int,
                    batch_size: int, lr: float, weight_decay: float,
                    snapshot_every: int, seed: int):
    """Train and yield (epoch, copy_of_model) snapshots for the GIF."""
    rng = np.random.default_rng(seed + 7)
    images = dataset["images"]
    presence = dataset["presence"].astype(np.float32)
    types = dataset["types"].astype(np.int64)
    positions = dataset["positions"].astype(np.float32)
    rotations = dataset["rotations"].astype(np.float32)
    n = images.shape[0]

    params = model.params()
    is_weight = [p.ndim >= 2 for p in params]
    m_state = [np.zeros_like(p) for p in params]
    v_state = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0

    snapshots = [(-1, _clone(model))]  # epoch -1 = pre-training
    for epoch in range(n_epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            cache = model.forward(images[idx])
            _, grad_out, _ = compute_loss(
                cache["out"], presence[idx], types[idx],
                positions[idx], rotations[idx],
            )
            grads = backward(model, cache, grad_out)
            step += 1
            for p, g, m, v, decay in zip(params, grads, m_state, v_state, is_weight):
                m[...] = beta1 * m + (1.0 - beta1) * g
                v[...] = beta2 * v + (1.0 - beta2) * (g * g)
                m_hat = m / (1.0 - beta1 ** step)
                v_hat = v / (1.0 - beta2 ** step)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)
                if decay and weight_decay > 0:
                    p -= lr * weight_decay * p
        if epoch % snapshot_every == 0 or epoch == n_epochs - 1:
            snapshots.append((epoch, _clone(model)))
    return snapshots


def _clone(model: AIR3DEncoder) -> AIR3DEncoder:
    new = AIR3DEncoder(image_size=model.image_size, max_slots=model.max_slots,
                       hidden=model.hidden, input_pool=model.input_pool, seed=0)
    new.W1 = model.W1.copy(); new.b1 = model.b1.copy()
    new.W2 = model.W2.copy(); new.b2 = model.b2.copy()
    new.W3 = model.W3.copy(); new.b3 = model.b3.copy()
    return new


def _make_frame(image: np.ndarray, gt_prims, pred_prims, image_size: int,
                epoch_label: str, dpi: int = 80) -> np.ndarray:
    pred_render = render_3d_scene(pred_prims, image_size=image_size)
    diff = np.abs(image - pred_render)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), dpi=dpi)
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    gt_str = "+".join(p.type[:3] for p in gt_prims) or "-"
    axes[0].set_title(f"input  (gt: {gt_str})", fontsize=9)
    axes[1].imshow(pred_render, cmap="gray", vmin=0, vmax=1)
    pred_str = "+".join(p.type[:3] for p in pred_prims) or "(none)"
    axes[1].set_title(f"pred re-render  ({pred_str})", fontsize=9)
    axes[2].imshow(diff, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title(f"|input - pred|  mae={diff.mean():.3f}", fontsize=9)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(epoch_label, fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--max-primitives", type=int, default=3)
    ap.add_argument("--n-epochs", type=int, default=40)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--input-pool", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--snapshot-every", type=int, default=2)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--out", type=str, default="air_3d_primitives.gif")
    ap.add_argument("--n-scenes", type=int, default=2,
                    help="number of held-out test scenes to animate side-by-side per "
                         "frame group (cycle through them)")
    args = ap.parse_args()

    print("[gif] sampling training set + held-out demo scenes...")
    images, presence, types, positions, rotations = generate_dataset(
        args.n_train, max_primitives=args.max_primitives,
        image_size=args.image_size, seed=args.seed,
    )
    train_dataset = {
        "images": images, "presence": presence, "types": types,
        "positions": positions, "rotations": rotations,
    }
    demo_imgs, demo_pres, demo_types, demo_pos, demo_rot = generate_dataset(
        args.n_scenes, max_primitives=args.max_primitives,
        image_size=args.image_size, seed=args.seed + 999,
    )
    demo_gt_prims = []
    for k in range(args.n_scenes):
        prims = []
        for slot in range(args.max_primitives):
            if demo_pres[k, slot] > 0.5:
                prims.append(Primitive(
                    type=PRIMITIVE_TYPES[int(demo_types[k, slot])],
                    position=demo_pos[k, slot].astype(np.float64),
                    euler=demo_rot[k, slot].astype(np.float64),
                ))
        demo_gt_prims.append(prims)

    model = build_air_model_3d(image_size=args.image_size,
                               max_slots=args.max_primitives,
                               hidden=args.hidden, input_pool=args.input_pool,
                               seed=args.seed)

    print(f"[gif] training {args.n_epochs} epochs, snapshot every {args.snapshot_every}...")
    snapshots = _snapshot_train(
        model, train_dataset, n_epochs=args.n_epochs,
        batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay,
        snapshot_every=args.snapshot_every, seed=args.seed,
    )
    print(f"[gif] {len(snapshots)} snapshots collected")

    frames = []
    # Cycle scene index k = step % n_scenes so the GIF doesn't get monotonous
    for snap_idx, (epoch, m) in enumerate(snapshots):
        k = snap_idx % args.n_scenes
        pred = m.decode(demo_imgs[k])
        epoch_label = "init (random weights)" if epoch < 0 else f"epoch {epoch}"
        frames.append(_make_frame(
            demo_imgs[k], demo_gt_prims[k], pred,
            image_size=args.image_size, epoch_label=epoch_label,
        ))

    # Hold the last frame for a beat
    for _ in range(max(1, args.fps)):
        frames.append(frames[-1])

    print(f"[gif] writing {len(frames)} frames to {args.out} at {args.fps} fps...")
    imageio.mimsave(args.out, frames, fps=args.fps)
    size_kb = os.path.getsize(args.out) / 1024.0
    print(f"[gif] done. {args.out} = {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
