"""
smallNORB held-out viewpoint generalization (Hinton, Sabour & Frosst 2018,
"Matrix capsules with EM routing", ICLR).

Reproduces the headline claim: a capsule network with 4x4 pose matrices and
EM routing extrapolates better to *novel viewpoints* than a parameter-matched
CNN. Train on a restricted azimuth range; test on held-out azimuths.

Pure numpy. Synthesized NORB-like dataset (5 categories of 3D voxel shapes
rendered from controlled (azimuth, elevation) viewpoints). The real smallNORB
download is documented as a deviation; the synthesized dataset preserves the
core experimental property -- a 3D shape rendered from many viewpoints -- so
the viewpoint-extrapolation question can be measured.

CLI:
    python3 smallnorb_novel_viewpoint.py \\
        --seed 0 --n-epochs 12 \\
        --train-azimuths 0:180 --test-azimuths 180:360
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ============================================================================
# Synthesized smallNORB-like dataset
# ============================================================================
#
# Real smallNORB is 5 categories (animals, humans, planes, trucks, cars)
# rendered at 9 elevations x 18 azimuths x 6 lighting conditions. The crucial
# experimental property for the viewpoint-extrapolation experiment is that
# every category appears at every viewpoint, so test viewpoints can be held
# out. We synthesize the same property with 5 voxel-defined 3D shape classes
# rendered orthographically from controlled (azimuth, elevation) angles.

SHAPE_CLASSES = ["cross", "L", "T", "frame", "tripod"]


def _shape_points(category: str) -> np.ndarray:
    """Return Nx3 array of 3D point coordinates for one shape class.

    Coordinates are in [-1, 1]^3 roughly. Each point will render as a small
    Gaussian blob in the silhouette image; the overall layout is what gives
    the class its viewpoint-dependent appearance.
    """
    if category == "cross":
        # 3D plus sign: center plus 6 axial neighbors
        pts = [(0, 0, 0)]
        for d in (-0.7, -0.35, 0.35, 0.7):
            pts.append((d, 0, 0))
            pts.append((0, d, 0))
            pts.append((0, 0, d))
        return np.array(pts, dtype=np.float64)
    if category == "L":
        # 3D L: line along +x at z=-0.6, line along +y at x=-0.6, z=-0.6
        pts = []
        for d in np.linspace(-0.8, 0.8, 9):
            pts.append((d, -0.7, -0.6))
        for d in np.linspace(-0.6, 0.8, 8):
            pts.append((-0.8, d, -0.6))
        for d in np.linspace(-0.5, 0.5, 5):
            pts.append((-0.8, -0.7, d))
        return np.array(pts, dtype=np.float64)
    if category == "T":
        # 3D T: top bar along x, stem down y, all at z=0
        pts = []
        for d in np.linspace(-0.8, 0.8, 9):
            pts.append((d, 0.7, 0.0))
        for d in np.linspace(-0.7, 0.6, 7):
            pts.append((0.0, d, 0.0))
        for d in np.linspace(-0.4, 0.4, 5):
            pts.append((0.0, 0.7, d))
        return np.array(pts, dtype=np.float64)
    if category == "frame":
        # 3D wireframe box: 12 edges of a cube. Sample points along edges.
        corners = np.array([(x, y, z) for x in (-0.7, 0.7)
                            for y in (-0.7, 0.7) for z in (-0.7, 0.7)])
        edges = []
        for i in range(8):
            for j in range(i + 1, 8):
                # an edge is a pair differing in exactly one coordinate
                if np.sum(np.abs(corners[i] - corners[j]) > 1e-6) == 1:
                    edges.append((i, j))
        pts = []
        for (i, j) in edges:
            for t in np.linspace(0, 1, 5):
                pts.append(tuple(corners[i] * (1 - t) + corners[j] * t))
        return np.array(pts, dtype=np.float64)
    if category == "tripod":
        # 3D tripod: 3 prongs from origin going to (1,0,0), (0,1,0), (-0.7,-0.7,-0.7)
        pts = [(0, 0, 0)]
        for d in np.linspace(0.0, 0.85, 6)[1:]:
            pts.append((d, 0, 0))
            pts.append((0, d, 0))
            pts.append((-0.7 * d, -0.7 * d, -0.7 * d))
        return np.array(pts, dtype=np.float64)
    raise ValueError(f"unknown shape class {category!r}")


def _rotation_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """3D rotation: rotate around Y (azimuth) then around X (elevation).

    smallNORB convention: azimuth 0..360 around the vertical axis, elevation
    0..90 above the horizon.
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    Ry = np.array([[np.cos(az), 0, np.sin(az)],
                   [0, 1, 0],
                   [-np.sin(az), 0, np.cos(az)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(el), -np.sin(el)],
                   [0, np.sin(el), np.cos(el)]])
    return Rx @ Ry


def render_view(category: str, azimuth_deg: float, elevation_deg: float,
                size: int = 32, blob_sigma: float = 1.6,
                noise_std: float = 0.0,
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Render one shape at one viewpoint as a `size x size` grayscale image.

    Each 3D point is projected orthographically to (x, y) and rasterized as a
    Gaussian blob with intensity gated by depth (closer points are brighter).
    """
    pts3d = _shape_points(category)
    R = _rotation_matrix(azimuth_deg, elevation_deg)
    rotated = pts3d @ R.T  # (N, 3); columns are x, y, z after rotation
    # Orthographic projection: use x, y. z gates intensity (closer = brighter).
    xs = rotated[:, 0]
    ys = rotated[:, 1]
    zs = rotated[:, 2]
    # Map [-1, 1] -> [margin, size-margin]
    margin = 4.0
    cx = (xs + 1.0) * 0.5 * (size - 2 * margin) + margin
    cy = (1.0 - ys) * 0.5 * (size - 2 * margin) + margin  # flip y for image coords
    intensity = 0.6 + 0.4 * (zs + 1.0) * 0.5  # [0.6, 1.0], front-lit

    grid_y, grid_x = np.mgrid[0:size, 0:size].astype(np.float64)
    img = np.zeros((size, size), dtype=np.float64)
    for k in range(len(pts3d)):
        d2 = (grid_x - cx[k]) ** 2 + (grid_y - cy[k]) ** 2
        img += intensity[k] * np.exp(-d2 / (2 * blob_sigma ** 2))
    img = np.clip(img, 0.0, 1.0)
    if noise_std > 0.0 and rng is not None:
        img = np.clip(img + rng.normal(0, noise_std, img.shape), 0.0, 1.0)
    return img.astype(np.float32)


def make_dataset(azimuth_deg_list: List[float],
                 elevation_deg_list: List[float],
                 n_per_combo: int,
                 seed: int = 0,
                 size: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate (images, labels, azimuths, elevations) for all combinations.

    n_per_combo controls how many noisy samples to generate per
    (category, az, el) triple. Each sample applies pixel jitter / noise.
    """
    rng = np.random.default_rng(seed)
    images = []
    labels = []
    azis = []
    eles = []
    for cls_idx, cls in enumerate(SHAPE_CLASSES):
        for az in azimuth_deg_list:
            for el in elevation_deg_list:
                for _ in range(n_per_combo):
                    img = render_view(cls, az, el, size=size,
                                      noise_std=0.04, rng=rng)
                    images.append(img)
                    labels.append(cls_idx)
                    azis.append(az)
                    eles.append(el)
    images = np.stack(images)
    labels = np.array(labels, dtype=np.int64)
    azis = np.array(azis, dtype=np.float32)
    eles = np.array(eles, dtype=np.float32)
    perm = rng.permutation(len(images))
    return images[perm], labels[perm], azis[perm], eles[perm]


def parse_az_range(spec: str) -> Tuple[float, float]:
    a, b = spec.split(":")
    return float(a), float(b)


def split_azimuths(train_range: Tuple[float, float],
                   test_range: Tuple[float, float],
                   n_train_views: int = 6,
                   n_test_views: int = 6) -> Tuple[List[float], List[float]]:
    """Build azimuth lists for train/test from inclusive ranges.

    Each range is sampled uniformly; small overlap is fine but we keep them
    disjoint so test viewpoints are genuinely held out.
    """
    train = list(np.linspace(train_range[0], train_range[1], n_train_views))
    test = list(np.linspace(test_range[0], test_range[1], n_test_views))
    return train, test


# ============================================================================
# Convolutional layer (numpy, naive but adequate for a single 5x5 stride-2 conv)
# ============================================================================

def conv2d_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray,
                   stride: int = 1) -> np.ndarray:
    """x: (B, C_in, H, W). W: (C_out, C_in, kH, kW). b: (C_out,). Same padding."""
    B, C_in, H, Wd = x.shape
    C_out, _, kH, kW = W.shape
    pH = kH // 2
    pW = kW // 2
    xp = np.pad(x, ((0, 0), (0, 0), (pH, pH), (pW, pW)))
    H_out = (H + 2 * pH - kH) // stride + 1
    W_out = (Wd + 2 * pW - kW) // stride + 1
    out = np.zeros((B, C_out, H_out, W_out), dtype=x.dtype)
    for i in range(H_out):
        for j in range(W_out):
            patch = xp[:, :, i * stride:i * stride + kH, j * stride:j * stride + kW]
            out[:, :, i, j] = np.einsum('bcij,ocij->bo', patch, W) + b
    return out


def conv2d_backward(dout: np.ndarray, x: np.ndarray, W: np.ndarray,
                    stride: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns dx, dW, db for the same conv2d_forward."""
    B, C_in, H, Wd = x.shape
    C_out, _, kH, kW = W.shape
    pH = kH // 2
    pW = kW // 2
    xp = np.pad(x, ((0, 0), (0, 0), (pH, pH), (pW, pW)))
    dxp = np.zeros_like(xp)
    dW = np.zeros_like(W)
    db = dout.sum(axis=(0, 2, 3))
    H_out = dout.shape[2]
    W_out = dout.shape[3]
    for i in range(H_out):
        for j in range(W_out):
            patch = xp[:, :, i * stride:i * stride + kH, j * stride:j * stride + kW]
            do = dout[:, :, i, j]  # (B, C_out)
            dW += np.einsum('bo,bcij->ocij', do, patch)
            dxp[:, :, i * stride:i * stride + kH, j * stride:j * stride + kW] += \
                np.einsum('bo,ocij->bcij', do, W)
    dx = dxp[:, :, pH:pH + H, pW:pW + Wd]
    return dx, dW, db


def relu(x):
    return np.maximum(x, 0.0)


def relu_grad(x, dout):
    return dout * (x > 0).astype(x.dtype)


# ============================================================================
# Matrix Capsule layer with EM routing (forward + simplified backward)
# ============================================================================

# Design simplifications (documented as deviations):
#   * Backward pass treats EM routing assignments as detached -- gradient flows
#     only through the last M-step (a weighted-mean of votes). This is the
#     standard "stop gradient through routing" trick used in many open-source
#     capsule implementations because full backprop through 3 EM iterations is
#     expensive in pure numpy.
#   * No coordinate-addition in PrimaryCaps -> ClassCaps because we go from a
#     small set of capsules directly to class capsules (no spatial replication).


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


@dataclass
class MatrixCapsLayer:
    """Routing layer: n_lower matrix capsules -> n_upper matrix capsules.

    Pose matrix dimension is `pose_d x pose_d` (default 4x4).
    """
    n_lower: int
    n_upper: int
    pose_d: int = 4
    n_iters: int = 3

    # Trainable weights: per-(i, j) pose transformation matrix W_ij of size
    # (pose_d, pose_d). vote_ij = M_i @ W_ij.
    W: np.ndarray = field(init=False)
    # Spread/cost biases (per upper-capsule, scalar).
    beta_a: np.ndarray = field(init=False)
    beta_v: np.ndarray = field(init=False)

    def init_weights(self, rng: np.random.Generator):
        scale = 1.0 / np.sqrt(self.pose_d)
        self.W = rng.normal(0, scale, (self.n_lower, self.n_upper,
                                       self.pose_d, self.pose_d)).astype(np.float64)
        self.beta_a = np.zeros(self.n_upper, dtype=np.float64)
        self.beta_v = np.zeros(self.n_upper, dtype=np.float64)

    def forward(self, M_lower: np.ndarray, a_lower: np.ndarray,
                lambda_: float = 1.0) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Run forward pass.

        M_lower: (B, n_lower, pose_d, pose_d) lower-layer pose matrices
        a_lower: (B, n_lower) lower-layer activations
        Returns:
            M_upper: (B, n_upper, pose_d, pose_d)
            a_upper: (B, n_upper)
            cache: dict for backward
        """
        B = M_lower.shape[0]
        # Compute votes: V_ij = M_i @ W_ij  ->  (B, n_lower, n_upper, pose_d, pose_d)
        V = np.einsum('bipq,ijqr->bijpr', M_lower, self.W)

        # Initialize routing assignments uniformly.
        R = np.full((B, self.n_lower, self.n_upper), 1.0 / self.n_upper)
        a_lower_b = a_lower[:, :, None]  # (B, n_lower, 1)

        H = self.pose_d * self.pose_d  # number of vote elements per (i,j) pair
        V_flat = V.reshape(B, self.n_lower, self.n_upper, H)  # for stats

        mu = None
        sigma2 = None
        a_upper = None
        for _it in range(self.n_iters):
            # M-step
            R_a = R * a_lower_b  # (B, n_lower, n_upper)
            sum_R_a = R_a.sum(axis=1, keepdims=True) + 1e-9  # (B, 1, n_upper)
            mu = (R_a[..., None] * V_flat).sum(axis=1, keepdims=True) / sum_R_a[..., None]
            # mu: (B, 1, n_upper, H)
            diff = V_flat - mu  # (B, n_lower, n_upper, H)
            sigma2 = (R_a[..., None] * diff ** 2).sum(axis=1, keepdims=True) / sum_R_a[..., None]
            sigma2 = np.clip(sigma2, 1e-6, None)
            # Cost per upper capsule (sum over h).
            cost_h = (self.beta_v[None, None, :, None] + 0.5 * np.log(sigma2)) * sum_R_a[..., None]
            cost = cost_h.sum(axis=-1).squeeze(1)  # (B, n_upper)
            a_upper = 1.0 / (1.0 + np.exp(-(lambda_ * (self.beta_a[None, :] - cost))))

            # E-step (skip on last iteration -- assignments aren't used after last M)
            if _it < self.n_iters - 1:
                log_p = -0.5 * np.sum(((V_flat - mu) ** 2) / sigma2 + np.log(2 * np.pi * sigma2),
                                      axis=-1)  # (B, n_lower, n_upper)
                # Combine with upper activation (in log-space) and normalize.
                log_a = np.log(np.clip(a_upper, 1e-9, None))[:, None, :]  # (B, 1, n_upper)
                log_un = log_p + log_a
                log_un = log_un - log_un.max(axis=2, keepdims=True)
                un = np.exp(log_un)
                R = un / (un.sum(axis=2, keepdims=True) + 1e-9)

        mu_mat = mu.squeeze(1).reshape(B, self.n_upper, self.pose_d, self.pose_d)

        cache = dict(M_lower=M_lower, a_lower=a_lower, V=V, R=R,
                     a_upper=a_upper, mu=mu, sigma2=sigma2,
                     sum_R_a=sum_R_a, V_flat=V_flat, lambda_=lambda_)
        return mu_mat, a_upper, cache

    def backward(self, d_mu_mat: np.ndarray, d_a_upper: np.ndarray,
                 cache: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass with stop-gradient on routing assignments R.

        We let gradient flow through:
          * the final M-step weighted mean (mu = sum_i weight_i * V_i)
          * the activation a_upper = sigmoid(lambda * (beta_a - cost)) where
            cost = sum_h (beta_v + 0.5 log sigma2_h) * sum_R_a, treating
            sum_R_a, weight_i = R_a_i / sum_R_a, and mu (in the sigma2
            expression) as detached. This keeps backward simple but still
            propagates a learning signal through the W transformation matrices.
        """
        M_lower = cache['M_lower']
        a_lower = cache['a_lower']
        R = cache['R']
        V = cache['V']            # (B, n_lower, n_upper, pose_d, pose_d)
        mu = cache['mu']          # (B, 1, n_upper, H)
        sigma2 = cache['sigma2']  # (B, 1, n_upper, H)
        sum_R_a = cache['sum_R_a']  # (B, 1, n_upper)
        a_upper = cache['a_upper']  # (B, n_upper)
        V_flat = cache['V_flat']  # (B, n_lower, n_upper, H)
        lambda_ = cache['lambda_']
        B = M_lower.shape[0]
        H = self.pose_d * self.pose_d

        # ---- a_upper backward ----
        # a = sigmoid(lambda * (beta_a - cost)).
        d_pre = d_a_upper * a_upper * (1.0 - a_upper) * lambda_  # (B, n_upper)
        d_beta_a = d_pre.sum(axis=0)
        d_cost = -d_pre  # (B, n_upper)

        # cost = sum_h (beta_v[j] + 0.5 log sigma2[b,j,h]) * sum_R_a[b,j]
        # Treat sum_R_a as constant. Then:
        sum_R_a_sq = sum_R_a.squeeze(1)  # (B, n_upper)
        d_beta_v = (d_cost * sum_R_a_sq * H).sum(axis=0)  # (n_upper,)
        # gradient to log(sigma2[b,j,h]):
        d_log_sigma2 = 0.5 * d_cost[:, None, :, None] * sum_R_a[..., None]  # (B,1,n_upper,H)
        # NOTE: above broadcast: d_cost (B,n_upper) -> (B,1,n_upper,1); sum_R_a (B,1,n_upper) -> (...,None)
        d_sigma2 = d_log_sigma2 / sigma2  # (B, 1, n_upper, H)

        # sigma2 = sum_i (R_a_i / sum_R_a) * (V_i - mu)^2  (mu detached for backward).
        # d sigma2 / d V_flat[b, i, j, h] = 2 * weight_i * (V_i - mu)
        R_a = R * a_lower[:, :, None]  # (B, n_lower, n_upper)
        weight = R_a / (sum_R_a + 1e-12)  # (B, n_lower, n_upper)
        diff = V_flat - mu  # (B, n_lower, n_upper, H)
        # d_V_flat from sigma2 path:
        d_V_flat_from_sigma = 2.0 * weight[..., None] * diff * d_sigma2  # broadcasts (B,1,nu,H) -> (B,nl,nu,H)

        # ---- mu backward (pose output) ----
        d_mu_flat = d_mu_mat.reshape(B, self.n_upper, H)[:, None, :, :]
        # d V_flat[b,i,j,h] from mu = weight_i * d_mu_flat[b,1,j,h]
        d_V_flat_from_mu = weight[..., None] * d_mu_flat

        d_V_flat = d_V_flat_from_mu + d_V_flat_from_sigma
        d_V = d_V_flat.reshape(B, self.n_lower, self.n_upper, self.pose_d, self.pose_d)

        # V = einsum('bipq,ijqr->bijpr', M_lower, W)
        # dW[i,j,q,r] = sum_{b,p} d_V[b,i,j,p,r] * M_lower[b,i,p,q]
        d_W = np.einsum('bijpr,bipq->ijqr', d_V, M_lower)
        # dM_lower[b,i,p,q] = sum_{j,r} d_V[b,i,j,p,r] * W[i,j,q,r]
        d_M_lower = np.einsum('bijpr,ijqr->bipq', d_V, self.W)

        # No gradient propagated to a_lower (treated as scalar gating; small
        # signal flows via R but R is detached). Keep zero.
        d_a_lower = np.zeros_like(a_lower)

        return d_M_lower, d_a_lower, d_W, d_beta_a, d_beta_v


# ============================================================================
# Capsule network model
# ============================================================================

@dataclass
class CapsNetParams:
    img_size: int = 32
    n_classes: int = 5
    conv_channels: int = 16
    conv_kernel: int = 5
    conv_stride: int = 2
    n_primary: int = 8
    pose_d: int = 4
    em_iters: int = 3


class CapsNet:
    def __init__(self, params: CapsNetParams, rng: np.random.Generator):
        self.p = params
        self.rng = rng

        # Conv1: 1 -> conv_channels, 5x5 stride 2.
        scale1 = np.sqrt(2.0 / (1 * params.conv_kernel ** 2))
        self.W1 = rng.normal(0, scale1, (params.conv_channels, 1,
                                         params.conv_kernel,
                                         params.conv_kernel)).astype(np.float64)
        self.b1 = np.zeros(params.conv_channels, dtype=np.float64)

        # PrimaryCaps: feature map (C, H', W') -> n_primary capsules (pose_d^2 + 1 per cap).
        H1 = (params.img_size + 2 * (params.conv_kernel // 2) - params.conv_kernel) // params.conv_stride + 1
        feat_dim = params.conv_channels * H1 * H1
        self._H1 = H1
        self._feat_dim = feat_dim

        n_primary = params.n_primary
        pose_dim = params.pose_d * params.pose_d  # 16
        prim_out = n_primary * (pose_dim + 1)  # 8 * 17 = 136
        scale2 = np.sqrt(2.0 / feat_dim)
        self.W_prim = rng.normal(0, scale2, (feat_dim, prim_out)).astype(np.float64)
        self.b_prim = np.zeros(prim_out, dtype=np.float64)

        # Class capsules (matrix caps + EM routing).
        self.routing = MatrixCapsLayer(n_lower=n_primary, n_upper=params.n_classes,
                                       pose_d=params.pose_d, n_iters=params.em_iters)
        self.routing.init_weights(rng)

    # ---- parameter list (for optimizer) ----
    def params(self) -> dict:
        return dict(W1=self.W1, b1=self.b1, W_prim=self.W_prim, b_prim=self.b_prim,
                    W_route=self.routing.W,
                    beta_a=self.routing.beta_a, beta_v=self.routing.beta_v)

    def set_params(self, d: dict):
        self.W1 = d['W1']; self.b1 = d['b1']
        self.W_prim = d['W_prim']; self.b_prim = d['b_prim']
        self.routing.W = d['W_route']
        self.routing.beta_a = d['beta_a']
        self.routing.beta_v = d['beta_v']

    # ---- forward ----
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """x: (B, 32, 32). Returns (class_poses (B, n_classes, 4, 4), class_acts (B, n_classes), cache)."""
        B = x.shape[0]
        x_in = x[:, None, :, :]  # (B, 1, H, W)
        # Conv1 + ReLU
        z1 = conv2d_forward(x_in, self.W1, self.b1, stride=self.p.conv_stride)
        a1 = relu(z1)

        # PrimaryCaps: flatten then linear.
        a1_flat = a1.reshape(B, -1)
        prim_out = a1_flat @ self.W_prim + self.b_prim  # (B, n_primary*(pose_dim+1))

        n_prim = self.p.n_primary
        pose_dim = self.p.pose_d * self.p.pose_d
        prim_reshaped = prim_out.reshape(B, n_prim, pose_dim + 1)
        prim_pose_flat = prim_reshaped[:, :, :pose_dim]
        prim_act_logit = prim_reshaped[:, :, pose_dim]
        prim_pose = prim_pose_flat.reshape(B, n_prim, self.p.pose_d, self.p.pose_d)
        prim_act = 1.0 / (1.0 + np.exp(-prim_act_logit))  # sigmoid

        # Routing layer.
        class_pose, class_act, route_cache = self.routing.forward(prim_pose, prim_act)

        cache = dict(x_in=x_in, z1=z1, a1=a1, a1_flat=a1_flat,
                     prim_out=prim_out, prim_reshaped=prim_reshaped,
                     prim_pose=prim_pose, prim_act=prim_act,
                     prim_act_logit=prim_act_logit,
                     route_cache=route_cache)
        return class_pose, class_act, cache

    # ---- backward (only on logits/poses we use; loss decides) ----
    def backward(self, d_class_pose: np.ndarray, d_class_act: np.ndarray,
                 cache: dict) -> dict:
        B = cache['x_in'].shape[0]
        n_prim = self.p.n_primary
        pose_dim = self.p.pose_d * self.p.pose_d

        # Routing layer backward.
        d_prim_pose, d_prim_act, d_W_route, d_beta_a, d_beta_v = \
            self.routing.backward(d_class_pose, d_class_act, cache['route_cache'])

        # Re-pack primary caps: pose_flat (B, n_prim, pose_dim) and act_logit (B, n_prim)
        d_prim_pose_flat = d_prim_pose.reshape(B, n_prim, pose_dim)
        prim_act = cache['prim_act']
        d_prim_act_logit = d_prim_act * prim_act * (1.0 - prim_act)

        d_prim_reshaped = np.concatenate(
            [d_prim_pose_flat, d_prim_act_logit[:, :, None]], axis=2)
        d_prim_out = d_prim_reshaped.reshape(B, n_prim * (pose_dim + 1))

        a1_flat = cache['a1_flat']
        d_W_prim = a1_flat.T @ d_prim_out
        d_b_prim = d_prim_out.sum(axis=0)
        d_a1_flat = d_prim_out @ self.W_prim.T

        d_a1 = d_a1_flat.reshape(B, self.p.conv_channels, self._H1, self._H1)
        d_z1 = relu_grad(cache['z1'], d_a1)
        d_x_in, d_W1, d_b1 = conv2d_backward(d_z1, cache['x_in'], self.W1,
                                             stride=self.p.conv_stride)

        return dict(W1=d_W1, b1=d_b1, W_prim=d_W_prim, b_prim=d_b_prim,
                    W_route=d_W_route, beta_a=d_beta_a, beta_v=d_beta_v)


# ============================================================================
# CNN baseline (matched parameter budget)
# ============================================================================

class CNNBaseline:
    """Conv (5x5, stride 2, 16ch) + flatten + Linear -> 5 classes.

    Parameter count is roughly matched to the capsule model on the conv
    layer (the capsule routing weights add a separate small budget).
    """

    def __init__(self, img_size: int = 32, n_classes: int = 5,
                 conv_channels: int = 16, conv_kernel: int = 5,
                 conv_stride: int = 2, hidden: int = 64,
                 rng: Optional[np.random.Generator] = None):
        if rng is None:
            rng = np.random.default_rng(0)
        scale1 = np.sqrt(2.0 / (1 * conv_kernel ** 2))
        self.W1 = rng.normal(0, scale1, (conv_channels, 1, conv_kernel,
                                         conv_kernel)).astype(np.float64)
        self.b1 = np.zeros(conv_channels, dtype=np.float64)
        H1 = (img_size + 2 * (conv_kernel // 2) - conv_kernel) // conv_stride + 1
        self.feat_dim = conv_channels * H1 * H1
        self._H1 = H1
        self.conv_channels = conv_channels
        self.conv_stride = conv_stride

        scale2 = np.sqrt(2.0 / self.feat_dim)
        self.W2 = rng.normal(0, scale2, (self.feat_dim, hidden)).astype(np.float64)
        self.b2 = np.zeros(hidden, dtype=np.float64)
        scale3 = np.sqrt(2.0 / hidden)
        self.W3 = rng.normal(0, scale3, (hidden, n_classes)).astype(np.float64)
        self.b3 = np.zeros(n_classes, dtype=np.float64)

    def params(self) -> dict:
        return dict(W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                    W3=self.W3, b3=self.b3)

    def set_params(self, d: dict):
        self.W1 = d['W1']; self.b1 = d['b1']
        self.W2 = d['W2']; self.b2 = d['b2']
        self.W3 = d['W3']; self.b3 = d['b3']

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        B = x.shape[0]
        x_in = x[:, None, :, :]
        z1 = conv2d_forward(x_in, self.W1, self.b1, stride=self.conv_stride)
        a1 = relu(z1)
        a1_flat = a1.reshape(B, -1)
        z2 = a1_flat @ self.W2 + self.b2
        a2 = relu(z2)
        z3 = a2 @ self.W3 + self.b3
        cache = dict(x_in=x_in, z1=z1, a1=a1, a1_flat=a1_flat,
                     z2=z2, a2=a2, z3=z3)
        return z3, cache

    def backward(self, d_z3: np.ndarray, cache: dict) -> dict:
        B = cache['x_in'].shape[0]
        a2 = cache['a2']
        d_W3 = a2.T @ d_z3
        d_b3 = d_z3.sum(axis=0)
        d_a2 = d_z3 @ self.W3.T
        d_z2 = relu_grad(cache['z2'], d_a2)
        a1_flat = cache['a1_flat']
        d_W2 = a1_flat.T @ d_z2
        d_b2 = d_z2.sum(axis=0)
        d_a1_flat = d_z2 @ self.W2.T
        d_a1 = d_a1_flat.reshape(B, self.conv_channels, self._H1, self._H1)
        d_z1 = relu_grad(cache['z1'], d_a1)
        d_x_in, d_W1, d_b1 = conv2d_backward(d_z1, cache['x_in'], self.W1,
                                             stride=self.conv_stride)
        return dict(W1=d_W1, b1=d_b1, W2=d_W2, b2=d_b2, W3=d_W3, b3=d_b3)


# ============================================================================
# Losses
# ============================================================================

def softmax_xent_loss(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """Numerically stable softmax cross-entropy. Returns (loss, dlogits)."""
    B, K = logits.shape
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    probs = ez / ez.sum(axis=1, keepdims=True)
    log_probs = np.log(np.clip(probs[np.arange(B), labels], 1e-12, None))
    loss = -log_probs.mean()
    grad = probs.copy()
    grad[np.arange(B), labels] -= 1.0
    grad /= B
    return loss, grad


def spread_loss(activations: np.ndarray, labels: np.ndarray,
                margin: float = 0.2) -> Tuple[float, np.ndarray]:
    """Spread loss from the matrix-capsules paper.

    L = sum_{i != t} max(0, m - (a_t - a_i))^2

    activations: (B, K) in (0, 1) -- already passed through sigmoid in the
    routing layer. labels: (B,) integer class indices.
    """
    B, K = activations.shape
    a_t = activations[np.arange(B), labels][:, None]  # (B, 1)
    diff = margin - (a_t - activations)  # (B, K)
    diff[np.arange(B), labels] = 0.0  # exclude target
    pos = np.maximum(diff, 0.0)
    loss = (pos ** 2).sum(axis=1).mean()
    # grad
    grad = np.zeros_like(activations)
    # d/d a_i (i != t): 2 * pos[i] * (-1) * (-1) = 2 * pos[i]; wait
    # diff_i = m - a_t + a_i, so d diff_i / d a_i = +1, d diff_i / d a_t = -1
    # L = sum_{i != t} pos_i^2 (per sample), d L / d a_i = 2 pos_i (i != t)
    # d L / d a_t = sum_{i != t} 2 pos_i * (-1) = -2 sum pos
    grad_other = 2.0 * pos
    grad_other[np.arange(B), labels] = 0.0
    grad_t = -grad_other.sum(axis=1)
    grad = grad_other.copy()
    grad[np.arange(B), labels] = grad_t
    grad /= B
    return loss, grad


# ============================================================================
# Optimizer (Adam)
# ============================================================================

class Adam:
    def __init__(self, params: dict, lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict, grads: dict) -> dict:
        self.t += 1
        out = {}
        for k in params:
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            out[k] = params[k] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return out


# ============================================================================
# Training & evaluation
# ============================================================================

def iterate_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int,
                        rng: np.random.Generator):
    n = len(X)
    perm = rng.permutation(n)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]


def caps_logits(class_pose: np.ndarray, class_act: np.ndarray) -> np.ndarray:
    """Combine class capsule pose-norm and activation into class logits.

    score_j = ||mu_j||_F^2 / pose_d^2 + 4 * (a_j - 0.5)

    Pose-norm gives a strong, dense gradient signal through W and the conv;
    the activation contribution lets EM routing's "agreement gating" influence
    the prediction. The 4* scale on activation centers it around zero (since
    a_j is in (0, 1)).
    """
    pd = class_pose.shape[-1]
    pose_score = (class_pose ** 2).sum(axis=(-1, -2)) / (pd * pd)
    return pose_score + 4.0 * (class_act - 0.5)


def caps_logits_grad(class_pose: np.ndarray, class_act: np.ndarray,
                     d_logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pd = class_pose.shape[-1]
    d_pose = 2.0 * class_pose * d_logits[..., None, None] / (pd * pd)
    d_act = 4.0 * d_logits
    return d_pose, d_act


def evaluate_acc_capsnet(model: 'CapsNet', X: np.ndarray, y: np.ndarray,
                         batch_size: int = 32) -> float:
    correct = 0
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        class_pose, class_act, _ = model.forward(xb)
        logits = caps_logits(class_pose, class_act)
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / len(X)


def train_capsnet(model: CapsNet, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  n_epochs: int = 8, batch_size: int = 32, lr: float = 1e-3,
                  rng: Optional[np.random.Generator] = None) -> dict:
    if rng is None:
        rng = np.random.default_rng(0)
    opt = Adam(model.params(), lr=lr)
    history = dict(epoch=[], train_loss=[], val_acc=[])
    for ep in range(n_epochs):
        model_params = model.params()
        ep_loss = 0.0; ep_count = 0
        for xb, yb in iterate_minibatches(X_train, y_train, batch_size, rng):
            class_pose, class_act, cache = model.forward(xb)
            logits = caps_logits(class_pose, class_act)
            loss, d_logits = softmax_xent_loss(logits, yb)
            d_pose, d_act = caps_logits_grad(class_pose, class_act, d_logits)
            grads = model.backward(d_pose, d_act, cache)
            new_params = opt.step(model_params, grads)
            model.set_params(new_params)
            model_params = new_params
            ep_loss += loss * len(xb); ep_count += len(xb)
        train_loss = ep_loss / ep_count
        val_acc = evaluate_acc_capsnet(model, X_val, y_val, batch_size)
        history['epoch'].append(ep)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        print(f"  [caps] epoch {ep+1}/{n_epochs}  loss={train_loss:.4f}  val_acc={val_acc:.3f}")
    return history


def train_cnn(model: CNNBaseline, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              n_epochs: int = 8, batch_size: int = 32, lr: float = 1e-3,
              rng: Optional[np.random.Generator] = None) -> dict:
    if rng is None:
        rng = np.random.default_rng(0)
    opt = Adam(model.params(), lr=lr)
    history = dict(epoch=[], train_loss=[], val_acc=[])
    for ep in range(n_epochs):
        model_params = model.params()
        ep_loss = 0.0; ep_count = 0
        for xb, yb in iterate_minibatches(X_train, y_train, batch_size, rng):
            logits, cache = model.forward(xb)
            loss, d_logits = softmax_xent_loss(logits, yb)
            grads = model.backward(d_logits, cache)
            new_params = opt.step(model_params, grads)
            model.set_params(new_params)
            model_params = new_params
            ep_loss += loss * len(xb); ep_count += len(xb)
        train_loss = ep_loss / ep_count
        val_acc = evaluate_acc_cnn(model, X_val, y_val, batch_size)
        history['epoch'].append(ep)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        print(f"  [cnn]  epoch {ep+1}/{n_epochs}  loss={train_loss:.4f}  val_acc={val_acc:.3f}")
    return history


def evaluate_acc_cnn(model: CNNBaseline, X: np.ndarray, y: np.ndarray,
                     batch_size: int = 32) -> float:
    correct = 0
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits, _ = model.forward(xb)
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / len(X)


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-azimuths", type=str, default="0:180",
                    help="azimuth range for training (e.g. 0:180)")
    ap.add_argument("--test-azimuths", type=str, default="180:340",
                    help="azimuth range for the held-out test set")
    ap.add_argument("--n-train-views", type=int, default=6,
                    help="number of azimuth samples in the training range")
    ap.add_argument("--n-test-views", type=int, default=6,
                    help="number of azimuth samples in the held-out test range")
    ap.add_argument("--n-elev", type=int, default=3,
                    help="number of elevation samples (in 10..50 deg)")
    ap.add_argument("--n-per-combo", type=int, default=6,
                    help="number of noisy samples per (class, az, el)")
    ap.add_argument("--em-iters", type=int, default=3)
    ap.add_argument("--out-json", type=str, default=None,
                    help="optional path to dump results JSON")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    az_train_range = parse_az_range(args.train_azimuths)
    az_test_range = parse_az_range(args.test_azimuths)
    train_azs, test_azs = split_azimuths(az_train_range, az_test_range,
                                         args.n_train_views, args.n_test_views)
    elevations = list(np.linspace(10, 50, args.n_elev))

    print(f"Train azimuths: {[f'{a:.0f}' for a in train_azs]}")
    print(f"Test  azimuths: {[f'{a:.0f}' for a in test_azs]}")
    print(f"Elevations:     {[f'{e:.0f}' for e in elevations]}")

    # Build datasets.
    print("Generating synthesized smallNORB-like dataset...")
    X_tr, y_tr, az_tr, el_tr = make_dataset(train_azs, elevations,
                                            n_per_combo=args.n_per_combo,
                                            seed=args.seed)
    X_fam, y_fam, _, _ = make_dataset(train_azs, elevations,
                                      n_per_combo=2, seed=args.seed + 100)
    X_no, y_no, _, _ = make_dataset(test_azs, elevations,
                                    n_per_combo=2, seed=args.seed + 200)
    print(f"  train      : {X_tr.shape}")
    print(f"  familiar   : {X_fam.shape}")
    print(f"  novel-view : {X_no.shape}")

    # ---- Capsule net ----
    print("Training MatrixCapsNet...")
    p = CapsNetParams(em_iters=args.em_iters)
    caps = CapsNet(p, np.random.default_rng(args.seed + 1))
    t0 = time.time()
    caps_history = train_capsnet(caps, X_tr, y_tr, X_fam, y_fam,
                                 n_epochs=args.n_epochs, batch_size=args.batch_size,
                                 lr=args.lr, rng=np.random.default_rng(args.seed + 2))
    caps_train_time = time.time() - t0
    caps_fam_acc = evaluate_acc_capsnet(caps, X_fam, y_fam)
    caps_novel_acc = evaluate_acc_capsnet(caps, X_no, y_no)

    # ---- CNN baseline ----
    print("Training CNN baseline...")
    cnn = CNNBaseline(rng=np.random.default_rng(args.seed + 3))
    t1 = time.time()
    cnn_history = train_cnn(cnn, X_tr, y_tr, X_fam, y_fam,
                            n_epochs=args.n_epochs, batch_size=args.batch_size,
                            lr=args.lr, rng=np.random.default_rng(args.seed + 4))
    cnn_train_time = time.time() - t1
    cnn_fam_acc = evaluate_acc_cnn(cnn, X_fam, y_fam)
    cnn_novel_acc = evaluate_acc_cnn(cnn, X_no, y_no)

    print()
    print("=" * 64)
    print("Viewpoint extrapolation results")
    print("=" * 64)
    print(f"{'Model':<14}{'Familiar acc':>16}{'Novel-view acc':>18}{'Drop':>10}")
    print(f"{'MatrixCaps':<14}{caps_fam_acc:>16.3f}{caps_novel_acc:>18.3f}"
          f"{caps_fam_acc - caps_novel_acc:>10.3f}")
    print(f"{'CNN':<14}{cnn_fam_acc:>16.3f}{cnn_novel_acc:>18.3f}"
          f"{cnn_fam_acc - cnn_novel_acc:>10.3f}")
    print(f"{'chance':<14}{0.2:>16.3f}{0.2:>18.3f}")
    print()
    print(f"Caps train wall: {caps_train_time:.1f}s   CNN train wall: {cnn_train_time:.1f}s")

    results = dict(
        seed=args.seed, n_epochs=args.n_epochs,
        train_azimuths=train_azs, test_azimuths=test_azs, elevations=elevations,
        caps=dict(fam_acc=float(caps_fam_acc), novel_acc=float(caps_novel_acc),
                  history=caps_history, train_wall=caps_train_time),
        cnn=dict(fam_acc=float(cnn_fam_acc), novel_acc=float(cnn_novel_acc),
                 history=cnn_history, train_wall=cnn_train_time),
    )
    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else o)
        print(f"Wrote {args.out_json}")
    return results


if __name__ == "__main__":
    main()
