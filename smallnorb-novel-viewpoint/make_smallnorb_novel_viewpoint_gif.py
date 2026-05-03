"""
Animated GIF for the smallNORB novel-viewpoint experiment.

Three panels per frame:
  1. The current input shape rotating through azimuths (one shape sweeps 0..360).
     Top half of the sweep is "in training range"; bottom half is "held out."
  2. Per-class capsule activations as the shape rotates.
  3. Per-azimuth accuracy curve growing in over training, with the train and
     held-out ranges shaded.

We train both models for a few epochs, take snapshots of capsule activations
across azimuths and the per-azimuth accuracy at each snapshot, then render
all frames into a single GIF.
"""

from __future__ import annotations

import argparse
import io
import os
import time

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from smallnorb_novel_viewpoint import (
    SHAPE_CLASSES, render_view, make_dataset, parse_az_range, split_azimuths,
    CapsNet, CapsNetParams, CNNBaseline,
    spread_loss, softmax_xent_loss, Adam,
    iterate_minibatches,
    caps_logits, caps_logits_grad,
    evaluate_acc_capsnet, evaluate_acc_cnn,
)


def _train_step_caps(model: CapsNet, opt: Adam, xb, yb) -> float:
    params = model.params()
    class_pose, class_act, cache = model.forward(xb)
    logits = caps_logits(class_pose, class_act)
    loss, d_logits = softmax_xent_loss(logits, yb)
    d_pose, d_act = caps_logits_grad(class_pose, class_act, d_logits)
    grads = model.backward(d_pose, d_act, cache)
    new_params = opt.step(params, grads)
    model.set_params(new_params)
    return float(loss)


def _train_step_cnn(model: CNNBaseline, opt: Adam, xb, yb) -> float:
    params = model.params()
    logits, cache = model.forward(xb)
    loss, d = softmax_xent_loss(logits, yb)
    grads = model.backward(d, cache)
    new_params = opt.step(params, grads)
    model.set_params(new_params)
    return float(loss)


def render_frame(caps: CapsNet, cnn: CNNBaseline,
                 az_grid, az_caps_acc, az_cnn_acc,
                 train_lo, train_hi, test_lo, test_hi,
                 sweep_az: float, sweep_cls_idx: int,
                 step: int, total_steps: int,
                 last_caps_loss: float) -> np.ndarray:
    """Render one frame as an RGB array."""
    fig = plt.figure(figsize=(10, 4.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.05])

    # Panel 1: current shape at sweep_az
    ax_img = fig.add_subplot(gs[0, 0])
    cls = SHAPE_CLASSES[sweep_cls_idx]
    img = render_view(cls, sweep_az, 30.0, size=32)
    ax_img.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    in_train = train_lo <= sweep_az <= train_hi
    in_test = test_lo <= sweep_az <= test_hi
    region_lbl = ("training" if in_train
                  else ("held-out" if in_test else "interpolation"))
    region_clr = ("green" if in_train
                  else ("orange" if in_test else "gray"))
    ax_img.set_title(f'"{cls}" @ az={sweep_az:.0f}°  ({region_lbl})',
                     fontsize=10, color=region_clr)

    # Panel 2: capsule activations vs class for this input
    ax_act = fig.add_subplot(gs[0, 1])
    x = img[None, :, :].astype(np.float64)
    class_pose, class_act, _ = caps.forward(x)
    logits = caps_logits(class_pose, class_act)
    pred = int(logits[0].argmax())
    colors = ['#888'] * len(SHAPE_CLASSES)
    colors[pred] = '#2a9'
    if pred == sweep_cls_idx:
        colors[pred] = '#2a9'
    else:
        colors[pred] = '#c33'
    bars = ax_act.bar(SHAPE_CLASSES, class_act[0], color=colors)
    # mark ground-truth with a hatch
    bars[sweep_cls_idx].set_edgecolor('black')
    bars[sweep_cls_idx].set_linewidth(2.0)
    ax_act.set_ylim(0, 1.05); ax_act.set_ylabel('class capsule activation')
    ax_act.set_title(f'predicted = "{SHAPE_CLASSES[pred]}"  '
                     f'(true = "{cls}")', fontsize=9)
    plt.setp(ax_act.get_xticklabels(), rotation=20, fontsize=8)

    # Panel 3: per-azimuth accuracy curves for caps and cnn
    ax_curve = fig.add_subplot(gs[0, 2])
    ax_curve.plot(az_grid, az_caps_acc, 'b-o', label='MatrixCaps', markersize=3)
    ax_curve.plot(az_grid, az_cnn_acc, 'r-s', label='CNN', markersize=3)
    ax_curve.axvspan(train_lo, train_hi, color='green', alpha=0.10)
    ax_curve.axvspan(test_lo, test_hi, color='orange', alpha=0.15)
    ax_curve.axvline(sweep_az, color='black', linewidth=1.2, linestyle='--')
    ax_curve.set_xlim(0, 360); ax_curve.set_ylim(0, 1.05)
    ax_curve.axhline(0.2, color='gray', linestyle=':', alpha=0.6)
    ax_curve.set_xlabel('azimuth (deg)'); ax_curve.set_ylabel('accuracy')
    ax_curve.set_title(f'per-azimuth accuracy  (step {step}/{total_steps})',
                       fontsize=9)
    ax_curve.legend(loc='lower left', fontsize=8)
    ax_curve.grid(alpha=0.3)

    fig.suptitle(
        f"Matrix capsules with EM routing -- viewpoint extrapolation "
        f"(caps loss = {last_caps_loss:.3f})", fontsize=10)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=85, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)


def _per_az_eval(model, model_kind: str, az_grid, elevations, seed):
    """Evaluate model at each azimuth in az_grid; returns array of accuracies."""
    out = []
    for az in az_grid:
        X, y, _, _ = make_dataset([float(az)], elevations, n_per_combo=2,
                                  seed=seed + int(az) + 999)
        if model_kind == 'caps':
            out.append(evaluate_acc_capsnet(model, X, y))
        else:
            out.append(evaluate_acc_cnn(model, X, y))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-epochs", type=int, default=8)
    ap.add_argument("--snapshots", type=int, default=12,
                    help="number of training-progress snapshots")
    ap.add_argument("--sweep-frames", type=int, default=18,
                    help="number of azimuth sweep frames per snapshot")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--train-azimuths", type=str, default="0:150")
    ap.add_argument("--test-azimuths", type=str, default="200:330")
    ap.add_argument("--n-train-views", type=int, default=6)
    ap.add_argument("--n-test-views", type=int, default=6)
    ap.add_argument("--n-elev", type=int, default=3)
    ap.add_argument("--n-per-combo", type=int, default=5)
    ap.add_argument("--out-path", type=str, default="smallnorb_novel_viewpoint.gif")
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    az_train_range = parse_az_range(args.train_azimuths)
    az_test_range = parse_az_range(args.test_azimuths)
    train_azs, test_azs = split_azimuths(az_train_range, az_test_range,
                                         args.n_train_views, args.n_test_views)
    elevations = list(np.linspace(10, 50, args.n_elev))
    train_lo, train_hi = min(train_azs), max(train_azs)
    test_lo, test_hi = min(test_azs), max(test_azs)

    print("=> Generating data...")
    X_tr, y_tr, _, _ = make_dataset(train_azs, elevations,
                                    n_per_combo=args.n_per_combo, seed=args.seed)
    X_fam, y_fam, _, _ = make_dataset(train_azs, elevations,
                                      n_per_combo=2, seed=args.seed + 100)

    p = CapsNetParams()
    caps = CapsNet(p, np.random.default_rng(args.seed + 1))
    cnn = CNNBaseline(rng=np.random.default_rng(args.seed + 3))
    caps_opt = Adam(caps.params(), lr=args.lr)
    cnn_opt = Adam(cnn.params(), lr=args.lr)

    # Total minibatch steps over training:
    n_batches_per_epoch = max(1, len(X_tr) // args.batch_size)
    total_batches = args.n_epochs * n_batches_per_epoch
    snapshot_every = max(1, total_batches // args.snapshots)

    az_grid = np.linspace(0, 350, 24)
    sweep_azs = np.linspace(0, 350, args.sweep_frames)

    frames = []
    rng_train = np.random.default_rng(args.seed + 5)
    last_caps_loss = float('nan')
    step = 0
    snapshot_idx = 0

    # Initial snapshot before training starts (frame at step 0)
    az_caps_acc = _per_az_eval(caps, 'caps', az_grid, elevations, args.seed)
    az_cnn_acc = _per_az_eval(cnn, 'cnn', az_grid, elevations, args.seed)
    sweep_cls = snapshot_idx % len(SHAPE_CLASSES)
    for sa in sweep_azs:
        frames.append(render_frame(caps, cnn,
                                   az_grid, az_caps_acc, az_cnn_acc,
                                   train_lo, train_hi, test_lo, test_hi,
                                   float(sa), sweep_cls, step, total_batches,
                                   last_caps_loss))

    print(f"=> Training {args.n_epochs} epochs, {total_batches} steps total, "
          f"snapshot every {snapshot_every}")
    for ep in range(args.n_epochs):
        for xb, yb in iterate_minibatches(X_tr, y_tr, args.batch_size, rng_train):
            last_caps_loss = _train_step_caps(caps, caps_opt, xb, yb)
            _train_step_cnn(cnn, cnn_opt, xb, yb)
            step += 1
            if step % snapshot_every == 0 or step == total_batches:
                snapshot_idx += 1
                az_caps_acc = _per_az_eval(caps, 'caps', az_grid, elevations, args.seed)
                az_cnn_acc = _per_az_eval(cnn, 'cnn', az_grid, elevations, args.seed)
                sweep_cls = snapshot_idx % len(SHAPE_CLASSES)
                for sa in sweep_azs:
                    frames.append(render_frame(
                        caps, cnn, az_grid, az_caps_acc, az_cnn_acc,
                        train_lo, train_hi, test_lo, test_hi,
                        float(sa), sweep_cls, step, total_batches,
                        last_caps_loss))
                print(f"  step {step}/{total_batches} caps_acc(fam) "
                      f"= {evaluate_acc_capsnet(caps, X_fam, y_fam):.3f}")

    print(f"=> Writing {len(frames)} frames to {args.out_path}")
    imageio.mimsave(args.out_path, frames, fps=args.fps, loop=0)
    size_mb = os.path.getsize(args.out_path) / 1e6
    print(f"=> {args.out_path}  {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
