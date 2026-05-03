"""Static visualizations for the AIR 3D-primitives experiment.

Produces, in ``viz/``:
- ``primitive_gallery.png`` -- canonical render of each primitive type at a
  few rotations, so the renderer's behaviour is auditable at a glance.
- ``scene_examples.png`` -- random sampled scenes (1, 2, 3 primitives).
- ``training_curves.png`` -- train/val loss + per-component breakdown.
- ``predictions.png`` -- side-by-side ground-truth image, predicted-primitive
  re-render, and an absolute difference map.
- ``error_distributions.png`` -- histograms of per-axis position and rotation
  errors on the held-out test set.

This script trains a fresh model from scratch (it does not rely on cached
``weights.npz``) so the figures track the current source.

Usage:
    python3 visualize_air_3d_primitives.py --seed 0 --image-size 64 \
        --n-epochs 80 --n-train 3000 --outdir viz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from air_3d_primitives import (
    PRIMITIVE_TYPES,
    Primitive,
    generate_dataset,
    render_3d_scene,
    run,
)


def primitive_gallery(outdir: Path):
    """Per-type renders at a handful of rotations."""
    rotations = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, np.pi / 4, 0.0]),
        np.array([np.pi / 6, np.pi / 4, np.pi / 6]),
        np.array([np.pi / 3, np.pi / 3, np.pi / 3]),
    ]
    fig, axes = plt.subplots(len(PRIMITIVE_TYPES), len(rotations),
                             figsize=(2.0 * len(rotations), 2.0 * len(PRIMITIVE_TYPES)))
    for i, t in enumerate(PRIMITIVE_TYPES):
        for j, eul in enumerate(rotations):
            prim = Primitive(type=t, position=np.zeros(3), euler=eul)
            img = render_3d_scene([prim], image_size=64)
            ax = axes[i, j]
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(t)
            if i == 0:
                ax.set_title(f"euler={[round(float(v), 2) for v in eul]}",
                             fontsize=8)
    fig.suptitle("Lambertian renderer: per-type rotation gallery")
    fig.tight_layout()
    fig.savefig(outdir / "primitive_gallery.png", dpi=110)
    plt.close(fig)


def scene_examples(outdir: Path, seed: int, n: int = 12):
    rng = np.random.default_rng(seed + 1)
    images, presence, types, _, _ = generate_dataset(
        n, max_primitives=3, image_size=64, seed=seed + 1,
    )
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5))
    for k, ax in enumerate(axes.ravel()):
        ax.imshow(images[k], cmap="gray", vmin=0.0, vmax=1.0)
        n_obj = int(presence[k].sum())
        type_str = "+".join(PRIMITIVE_TYPES[t][:3] for t in types[k] if t >= 0)
        ax.set_title(f"n={n_obj}  [{type_str}]", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Sampled scenes (variable count, types, positions, rotations)")
    fig.tight_layout()
    fig.savefig(outdir / "scene_examples.png", dpi=110)
    plt.close(fig)


def training_curves(outdir: Path, history: dict):
    epochs = np.array(history["epoch"])
    train = np.array(history["train_loss"])
    val = np.array(history["val_loss"])
    components = history["components"]
    pres = np.array([c["presence"] for c in components])
    typ = np.array([c["type"] for c in components])
    pos = np.array([c["position"] for c in components])
    rot = np.array([c["rotation"] for c in components])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(epochs, train, label="train")
    axes[0, 0].plot(epochs, val, label="val")
    axes[0, 0].axvline(history.get("best_epoch", -1), color="red",
                        linestyle="--", alpha=0.5, label="best val")
    axes[0, 0].set_title("total loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, pres, label="presence (BCE)")
    axes[0, 1].plot(epochs, typ, label="type (CE)")
    axes[0, 1].axhline(np.log(3.0), color="gray", linestyle=":",
                        label="type chance")
    axes[0, 1].set_title("classification components (val)")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, pos)
    axes[1, 0].set_title("position MSE (val)")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("MSE")

    axes[1, 1].plot(epochs, rot)
    axes[1, 1].set_title("rotation MSE (val, sphere-masked)")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("MSE")

    fig.suptitle("AIR-3D inference network: training curves")
    fig.tight_layout()
    fig.savefig(outdir / "training_curves.png", dpi=110)
    plt.close(fig)


def predictions(model, outdir: Path, seed: int, n_examples: int = 6):
    """Pick scenes from a fresh test split, decode each, re-render the
    predicted primitives, and diff against the ground truth."""
    rng = np.random.default_rng(seed + 31)
    images, presence, types, positions, rotations = generate_dataset(
        n_examples, max_primitives=3, image_size=model.image_size, seed=seed + 31,
    )
    pred_lists = model.decode(images)
    if isinstance(pred_lists, list) and pred_lists and isinstance(pred_lists[0], Primitive):
        pred_lists = [pred_lists]

    fig, axes = plt.subplots(n_examples, 4, figsize=(11, 2.5 * n_examples))
    if n_examples == 1:
        axes = axes[None, :]
    for k in range(n_examples):
        # Ground truth re-render (sanity: should equal image up to noise)
        gt_prims = []
        for slot in range(3):
            if presence[k, slot] > 0.5:
                gt_prims.append(Primitive(
                    type=PRIMITIVE_TYPES[int(types[k, slot])],
                    position=positions[k, slot].astype(np.float64),
                    euler=rotations[k, slot].astype(np.float64),
                ))
        gt_render = render_3d_scene(gt_prims, image_size=model.image_size)
        pred_render = render_3d_scene(pred_lists[k], image_size=model.image_size)
        diff = np.abs(images[k] - pred_render)

        gt_str = "+".join(p.type[:3] for p in gt_prims)
        pred_str = "+".join(p.type[:3] for p in pred_lists[k]) or "(none)"

        axes[k, 0].imshow(images[k], cmap="gray", vmin=0, vmax=1)
        axes[k, 0].set_title(f"input (gt: {gt_str})", fontsize=9)
        axes[k, 1].imshow(gt_render, cmap="gray", vmin=0, vmax=1)
        axes[k, 1].set_title("gt re-render", fontsize=9)
        axes[k, 2].imshow(pred_render, cmap="gray", vmin=0, vmax=1)
        axes[k, 2].set_title(f"pred re-render ({pred_str})", fontsize=9)
        axes[k, 3].imshow(diff, cmap="hot", vmin=0, vmax=1)
        axes[k, 3].set_title(f"|input - pred| (mae {diff.mean():.3f})", fontsize=9)
        for c in range(4):
            axes[k, c].set_xticks([]); axes[k, c].set_yticks([])

    fig.suptitle("Per-scene primitive recovery + reconstruction error",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(outdir / "predictions.png", dpi=110)
    plt.close(fig)


def error_distributions(model, outdir: Path, seed: int, n: int = 500):
    """Histograms of per-axis position and rotation error on a fresh sample."""
    images, presence, types, positions, rotations = generate_dataset(
        n, max_primitives=3, image_size=model.image_size, seed=seed + 71,
    )
    out = model.forward(images)["out"]
    pres_logit = out[..., 0]
    type_logits = out[..., 1:4]
    pos_p = out[..., 4:7]
    rot_p = out[..., 7:10]
    type_pred = np.argmax(type_logits, axis=-1)
    mask = presence.astype(bool)

    pos_err = np.abs(pos_p - positions)[mask]
    rot_diff = np.abs(rot_p - rotations)
    rot_diff = np.minimum(rot_diff, np.pi - rot_diff)
    rot_err = rot_diff[mask]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].hist([pos_err[:, 0], pos_err[:, 1], pos_err[:, 2]],
                 bins=20, label=["x", "y", "z"], alpha=0.7)
    axes[0].axvline(np.mean(pos_err), color="black", linestyle="--",
                    alpha=0.6, label=f"mean {pos_err.mean():.3f}")
    axes[0].set_title("position |error| per axis")
    axes[0].set_xlabel("|prediction - target|  (positions in [-1, 1])")
    axes[0].legend()

    axes[1].hist([rot_err[:, 0], rot_err[:, 1], rot_err[:, 2]],
                 bins=20, label=["alpha", "beta", "gamma"], alpha=0.7)
    axes[1].axvline(np.mean(rot_err), color="black", linestyle="--",
                    alpha=0.6, label=f"mean {rot_err.mean():.3f} rad")
    axes[1].set_title("rotation |error| per Euler axis (mod pi)")
    axes[1].set_xlabel("rad")
    axes[1].legend()

    # Confusion matrix of type predictions
    conf = np.zeros((3, 3), dtype=int)
    for t_true, t_pred in zip(types[mask], type_pred[mask]):
        conf[int(t_true), int(t_pred)] += 1
    axes[2].imshow(conf, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axes[2].text(j, i, str(conf[i, j]), ha="center", va="center",
                         color="black" if conf[i, j] < conf.max() / 2 else "white")
    axes[2].set_xticks(range(3)); axes[2].set_yticks(range(3))
    axes[2].set_xticklabels(PRIMITIVE_TYPES); axes[2].set_yticklabels(PRIMITIVE_TYPES)
    axes[2].set_xlabel("predicted")
    axes[2].set_ylabel("true")
    axes[2].set_title("type confusion (present slots)")

    fig.suptitle("Held-out error distributions")
    fig.tight_layout()
    fig.savefig(outdir / "error_distributions.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--max-primitives", type=int, default=3)
    ap.add_argument("--n-epochs", type=int, default=80)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--input-pool", type=int, default=2)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--outdir", type=str, default="viz")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[viz] rendering primitive gallery...")
    primitive_gallery(outdir)
    print("[viz] rendering scene examples...")
    scene_examples(outdir, seed=args.seed)

    print("[viz] training the AIR-3D encoder for figures...")
    result, model = run(
        seed=args.seed,
        image_size=args.image_size,
        max_primitives=args.max_primitives,
        n_epochs=args.n_epochs,
        n_train=args.n_train,
        n_test=args.n_test,
        hidden=args.hidden,
        input_pool=args.input_pool,
        weight_decay=args.weight_decay,
        save_weights=None,
        save_results=None,
        verbose=False,
    )
    history = result["history"]
    print(f"[viz] best val epoch={history.get('best_epoch')} "
          f"val_loss={history.get('best_val'):.4f}")
    print(f"[viz] count_acc={result['metrics']['count_acc']:.3f}  "
          f"type_acc={result['metrics']['type_acc']:.3f}")

    print("[viz] writing training curves...")
    training_curves(outdir, history)
    print("[viz] writing prediction panel...")
    predictions(model, outdir, seed=args.seed)
    print("[viz] writing error distributions...")
    error_distributions(model, outdir, seed=args.seed)
    print(f"[viz] all figures written to {outdir}/")


if __name__ == "__main__":
    main()
