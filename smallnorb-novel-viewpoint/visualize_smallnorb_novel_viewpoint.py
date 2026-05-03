"""
Static visualizations for the smallNORB novel-viewpoint experiment.

Trains MatrixCapsNet + CNN baseline (small config, ~30s wall) and writes:
  viz/example_views.png         5 categories x 6 azimuths
  viz/elevation_strip.png       5 categories x 4 elevations
  viz/training_curves.png       loss + val_acc per model per epoch
  viz/extrapolation_bars.png    familiar vs held-out per model
  viz/azimuth_accuracy.png      per-azimuth accuracy (train range vs held-out range)
  viz/pose_matrices.png         class capsule pose matrices for example inputs
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from smallnorb_novel_viewpoint import (
    SHAPE_CLASSES, render_view, make_dataset, parse_az_range, split_azimuths,
    CapsNet, CapsNetParams, CNNBaseline,
    train_capsnet, train_cnn,
    evaluate_acc_capsnet, evaluate_acc_cnn,
    caps_logits,
)


def plot_example_views(out_path: str):
    """Render each of the 5 categories at 6 evenly-spaced azimuths."""
    azs = np.linspace(0, 300, 6)
    fig, axes = plt.subplots(len(SHAPE_CLASSES), len(azs),
                             figsize=(1.4 * len(azs), 1.4 * len(SHAPE_CLASSES)))
    for i, cls in enumerate(SHAPE_CLASSES):
        for j, az in enumerate(azs):
            img = render_view(cls, float(az), 30.0, size=32)
            ax = axes[i, j]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"az={az:.0f}°", fontsize=9)
            if j == 0:
                ax.set_ylabel(cls, fontsize=10)
    fig.suptitle("Synthesized smallNORB-like dataset: 5 categories x 6 azimuths",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_elevation_strip(out_path: str):
    """Each category at 4 elevations to show the elevation degree of freedom."""
    els = [10, 25, 40, 55]
    fig, axes = plt.subplots(len(SHAPE_CLASSES), len(els),
                             figsize=(1.4 * len(els), 1.4 * len(SHAPE_CLASSES)))
    for i, cls in enumerate(SHAPE_CLASSES):
        for j, el in enumerate(els):
            img = render_view(cls, 60.0, float(el), size=32)
            ax = axes[i, j]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"el={el}°", fontsize=9)
            if j == 0:
                ax.set_ylabel(cls, fontsize=10)
    fig.suptitle("Elevation sweep at azimuth 60°", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_training_curves(caps_hist: dict, cnn_hist: dict, out_path: str):
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.2))
    a.plot(caps_hist['epoch'], caps_hist['train_loss'], 'b-o', label='MatrixCaps')
    a.plot(cnn_hist['epoch'], cnn_hist['train_loss'], 'r-s', label='CNN')
    a.set_xlabel('epoch'); a.set_ylabel('train loss'); a.set_title('Training loss')
    a.legend(); a.grid(alpha=0.3)

    b.plot(caps_hist['epoch'], caps_hist['val_acc'], 'b-o', label='MatrixCaps')
    b.plot(cnn_hist['epoch'], cnn_hist['val_acc'], 'r-s', label='CNN')
    b.axhline(0.2, color='gray', linestyle='--', alpha=0.5, label='chance')
    b.set_xlabel('epoch'); b.set_ylabel('val accuracy (familiar)')
    b.set_title('Val accuracy on familiar viewpoints')
    b.legend(); b.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_extrapolation_bars(caps_fam: float, caps_novel: float,
                            cnn_fam: float, cnn_novel: float, out_path: str):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(2)
    w = 0.35
    fam = [caps_fam, cnn_fam]
    novel = [caps_novel, cnn_novel]
    ax.bar(x - w/2, fam, w, label='familiar viewpoint', color='#4878d0')
    ax.bar(x + w/2, novel, w, label='held-out viewpoint', color='#ee854a')
    ax.set_xticks(x); ax.set_xticklabels(['MatrixCaps', 'CNN'])
    ax.set_ylim(0, 1.05); ax.set_ylabel('accuracy')
    ax.axhline(0.2, color='gray', linestyle='--', alpha=0.5, label='chance')
    for i, (f, n) in enumerate(zip(fam, novel)):
        ax.text(i - w/2, f + 0.01, f'{f:.2f}', ha='center', fontsize=9)
        ax.text(i + w/2, n + 0.01, f'{n:.2f}', ha='center', fontsize=9)
        ax.text(i, -0.06, f'drop: {f - n:.2f}', ha='center', fontsize=8.5,
                color='#444')
    ax.set_title('Familiar vs held-out viewpoint accuracy')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_azimuth_accuracy(caps: CapsNet, cnn: CNNBaseline,
                          train_azs, test_azs, elevations, seed: int,
                          out_path: str):
    """Per-azimuth accuracy on a fine grid 0..360 to show extrapolation."""
    azs = np.linspace(0, 350, 36)
    caps_acc = []
    cnn_acc = []
    for az in azs:
        X, y, _, _ = make_dataset([float(az)], elevations, n_per_combo=4,
                                  seed=seed + 700 + int(az))
        caps_acc.append(evaluate_acc_capsnet(caps, X, y))
        cnn_acc.append(evaluate_acc_cnn(cnn, X, y))
    caps_acc = np.array(caps_acc)
    cnn_acc = np.array(cnn_acc)

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(azs, caps_acc, 'b-o', label='MatrixCaps', markersize=4)
    ax.plot(azs, cnn_acc, 'r-s', label='CNN', markersize=4)
    # Shade train and test azimuth ranges.
    train_lo, train_hi = min(train_azs), max(train_azs)
    test_lo, test_hi = min(test_azs), max(test_azs)
    ax.axvspan(train_lo, train_hi, color='green', alpha=0.10,
               label=f'train [{train_lo:.0f}-{train_hi:.0f}]')
    ax.axvspan(test_lo, test_hi, color='orange', alpha=0.15,
               label=f'held-out [{test_lo:.0f}-{test_hi:.0f}]')
    ax.axhline(0.2, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('azimuth (degrees)'); ax.set_ylabel('accuracy')
    ax.set_title('Per-azimuth accuracy: extrapolation outside training range')
    ax.set_xlim(0, 360); ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_pose_matrices(caps: CapsNet, seed: int, out_path: str):
    """For one example per category, show the 5 class-capsule pose matrices.

    Each row is a category; each column is one of the 5 class capsules.
    The diagonal-most-active column should match the row's class.
    """
    fig, axes = plt.subplots(len(SHAPE_CLASSES), len(SHAPE_CLASSES) + 1,
                             figsize=(1.5 * (len(SHAPE_CLASSES) + 1),
                                      1.5 * len(SHAPE_CLASSES)))
    for i, cls in enumerate(SHAPE_CLASSES):
        img = render_view(cls, 60.0, 30.0, size=32)
        x = img[None, :, :].astype(np.float64)
        class_pose, class_act, _ = caps.forward(x)
        # Column 0: input image
        ax = axes[i, 0]
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title('input', fontsize=9)
        ax.set_ylabel(cls, fontsize=10)
        # Remaining: 5 pose matrices
        for j in range(len(SHAPE_CLASSES)):
            ax = axes[i, j + 1]
            mat = class_pose[0, j]  # 4x4
            ax.imshow(mat, cmap='RdBu_r',
                      vmin=-np.abs(class_pose).max(),
                      vmax=np.abs(class_pose).max())
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f'{SHAPE_CLASSES[j]}\na={class_act[0, j]:.2f}',
                             fontsize=8)
            else:
                ax.set_title(f'a={class_act[0, j]:.2f}', fontsize=8)
            # Highlight the predicted class with a green frame.
            logits = caps_logits(class_pose, class_act)
            pred = int(logits[0].argmax())
            if j == pred:
                for spine in ax.spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(2.5)
    fig.suptitle('Class capsule 4x4 pose matrices and activations\n'
                 '(green frame = predicted class; rows = ground truth)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--train-azimuths", type=str, default="0:150")
    ap.add_argument("--test-azimuths", type=str, default="200:330")
    ap.add_argument("--n-train-views", type=int, default=6)
    ap.add_argument("--n-test-views", type=int, default=6)
    ap.add_argument("--n-elev", type=int, default=3)
    ap.add_argument("--n-per-combo", type=int, default=5)
    ap.add_argument("--out-dir", type=str, default="viz")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    az_train_range = parse_az_range(args.train_azimuths)
    az_test_range = parse_az_range(args.test_azimuths)
    train_azs, test_azs = split_azimuths(az_train_range, az_test_range,
                                         args.n_train_views, args.n_test_views)
    elevations = list(np.linspace(10, 50, args.n_elev))

    print("=> Static figure 1: example views")
    plot_example_views(os.path.join(args.out_dir, "example_views.png"))
    print("=> Static figure 2: elevation strip")
    plot_elevation_strip(os.path.join(args.out_dir, "elevation_strip.png"))

    print("=> Generating data...")
    X_tr, y_tr, _, _ = make_dataset(train_azs, elevations,
                                    n_per_combo=args.n_per_combo, seed=args.seed)
    X_fam, y_fam, _, _ = make_dataset(train_azs, elevations,
                                      n_per_combo=2, seed=args.seed + 100)
    X_no, y_no, _, _ = make_dataset(test_azs, elevations,
                                    n_per_combo=2, seed=args.seed + 200)

    print("=> Training MatrixCapsNet ...")
    p = CapsNetParams()
    caps = CapsNet(p, np.random.default_rng(args.seed + 1))
    t0 = time.time()
    caps_hist = train_capsnet(caps, X_tr, y_tr, X_fam, y_fam,
                              n_epochs=args.n_epochs, batch_size=args.batch_size,
                              lr=args.lr, rng=np.random.default_rng(args.seed + 2))
    caps_train_t = time.time() - t0

    print("=> Training CNN baseline ...")
    cnn = CNNBaseline(rng=np.random.default_rng(args.seed + 3))
    t1 = time.time()
    cnn_hist = train_cnn(cnn, X_tr, y_tr, X_fam, y_fam,
                         n_epochs=args.n_epochs, batch_size=args.batch_size,
                         lr=args.lr, rng=np.random.default_rng(args.seed + 4))
    cnn_train_t = time.time() - t1

    caps_fam = evaluate_acc_capsnet(caps, X_fam, y_fam)
    caps_novel = evaluate_acc_capsnet(caps, X_no, y_no)
    cnn_fam = evaluate_acc_cnn(cnn, X_fam, y_fam)
    cnn_novel = evaluate_acc_cnn(cnn, X_no, y_no)

    print(f"=> Caps fam={caps_fam:.3f} novel={caps_novel:.3f}  "
          f"CNN fam={cnn_fam:.3f} novel={cnn_novel:.3f}")

    print("=> Static figure 3: training curves")
    plot_training_curves(caps_hist, cnn_hist,
                         os.path.join(args.out_dir, "training_curves.png"))
    print("=> Static figure 4: extrapolation bars")
    plot_extrapolation_bars(caps_fam, caps_novel, cnn_fam, cnn_novel,
                            os.path.join(args.out_dir, "extrapolation_bars.png"))
    print("=> Static figure 5: per-azimuth accuracy")
    plot_azimuth_accuracy(caps, cnn, train_azs, test_azs, elevations,
                          args.seed, os.path.join(args.out_dir, "azimuth_accuracy.png"))
    print("=> Static figure 6: pose matrices")
    plot_pose_matrices(caps, args.seed,
                       os.path.join(args.out_dir, "pose_matrices.png"))

    print(f"\nWrote 6 figures to {args.out_dir}/")
    print(f"Total wallclock: caps {caps_train_t:.1f}s  cnn {cnn_train_t:.1f}s")


if __name__ == "__main__":
    main()
