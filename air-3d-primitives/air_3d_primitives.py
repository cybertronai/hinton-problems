"""
AIR with a programmable 3D renderer (Eslami, Heess, Weber, Tassa, Szepesvari,
Kavukcuoglu & Hinton, 2016).

Pipeline
--------
1. A pure-numpy Lambertian renderer that ray-casts up to 3 unit primitives
   (sphere, cube, cylinder) under an orthographic camera and a single
   camera-direction light.
2. A scene generator that samples (count, types, positions, Euler rotations)
   and renders the corresponding 64x64 grayscale image.
3. An AIR-style inference network: an MLP that maps the image to a
   per-slot tuple (presence, type one-hot, 3D position, 3D rotation). Slots
   are made permutation-free by sorting ground-truth primitives by their
   x-position, so the network learns a canonical decomposition.
4. Supervised training of the inference network on synthesized
   (image, ground-truth) pairs.

This is the inference half of AIR — we share the *encoder* idea (variable
count, factored what/where) and reuse the renderer as a known generative
model. We skip the REINFORCE step over discrete z_pres that the original
paper used to train end-to-end from pixels alone, since pure-numpy
implementations of REINFORCE-AIR are brittle. The renderer + supervised
encoder still tests the core claim of the paper: that the inverse of a
known programmable renderer is learnable from images.

Usage
-----
    python3 air_3d_primitives.py --seed 0 --image-size 64 \
        --max-primitives 3 --n-epochs 30

Outputs JSON results to ``results.json`` and learned weights to
``weights.npz`` so the visualization scripts can reload them.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

PRIMITIVE_TYPES = ("sphere", "cube", "cylinder")
TYPE_TO_INDEX = {t: i for i, t in enumerate(PRIMITIVE_TYPES)}


def euler_to_rotation(angles: np.ndarray) -> np.ndarray:
    """ZYX intrinsic Euler angles -> 3x3 rotation matrix.

    angles: shape (3,) with (alpha, beta, gamma) in radians. Each Euler angle
    rotates around z, y, x respectively.
    """
    alpha, beta, gamma = angles
    cz, sz = np.cos(alpha), np.sin(alpha)
    cy, sy = np.cos(beta), np.sin(beta)
    cx, sx = np.cos(gamma), np.sin(gamma)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return Rz @ Ry @ Rx


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

@dataclass
class Primitive:
    """One scene element. Type + pose."""
    type: str
    position: np.ndarray  # (3,) world coords in [-1, 1]
    euler: np.ndarray     # (3,) Euler angles in [0, pi]
    scale: float = 0.4

    def as_vector(self) -> np.ndarray:
        type_one_hot = np.zeros(len(PRIMITIVE_TYPES))
        type_one_hot[TYPE_TO_INDEX[self.type]] = 1.0
        return np.concatenate([type_one_hot, self.position, self.euler])

    @property
    def rotation(self) -> np.ndarray:
        return euler_to_rotation(self.euler)


def _ray_grid(image_size: int, extent: float = 1.6):
    """Build (image_size**2, 3) ray origins and directions for orthographic
    camera at z=+5 looking along -z. The image plane covers [-extent, extent]
    in x and y."""
    coords = np.linspace(-extent, extent, image_size)
    px, py = np.meshgrid(coords, coords)  # (H, W) each; py grows downward
    # Flip y so increasing pixel row = decreasing world y (standard image up)
    ray_o = np.stack(
        [px.ravel(), -py.ravel(), np.full(px.size, 5.0)], axis=1
    ).astype(np.float64)
    ray_d = np.tile(np.array([0.0, 0.0, -1.0]), (px.size, 1)).astype(np.float64)
    return ray_o, ray_d


def _intersect_sphere(local_o: np.ndarray, local_d: np.ndarray):
    """Ray-unit-sphere intersection. Returns (t, valid, local_normal)."""
    a = (local_d * local_d).sum(axis=1)
    b = (local_o * local_d).sum(axis=1)
    c = (local_o * local_o).sum(axis=1) - 1.0
    disc = b * b - a * c
    sqd = np.sqrt(np.maximum(disc, 0.0))
    a_safe = np.where(np.abs(a) < 1e-12, 1e-12, a)
    t = (-b - sqd) / a_safe
    valid = (disc >= 0.0) & (t > 1e-4)
    local_p = local_o + t[:, None] * local_d
    local_n = local_p  # outward normal of unit sphere = position
    return t, valid, local_n


def _intersect_cube(local_o: np.ndarray, local_d: np.ndarray):
    """Ray-unit-cube intersection (|x|, |y|, |z| <= 1) by slab method."""
    eps = 1e-9
    d_safe = np.where(np.abs(local_d) < eps, eps, local_d)
    t1 = (-1.0 - local_o) / d_safe
    t2 = (1.0 - local_o) / d_safe
    t_min = np.minimum(t1, t2)  # (N, 3)
    t_max = np.maximum(t1, t2)
    t_enter = t_min.max(axis=1)
    t_exit = t_max.min(axis=1)
    valid = (t_enter <= t_exit) & (t_enter > 1e-4)
    t = t_enter
    axis = np.argmax(t_min, axis=1)  # which slab determined the entry
    local_p = local_o + t[:, None] * local_d
    n = np.zeros_like(local_p)
    rows = np.arange(local_p.shape[0])
    sign = np.sign(local_p[rows, axis])
    # if exactly 0 (degenerate), default to +1
    sign = np.where(sign == 0, 1.0, sign)
    n[rows, axis] = sign
    return t, valid, n


def _intersect_cylinder(local_o: np.ndarray, local_d: np.ndarray):
    """Ray-unit-cylinder intersection with axis along y, |y| <= 1, x^2+z^2 <= 1.

    Picks the closest of: side-wall hit, top-cap hit, bottom-cap hit.
    """
    n_rays = local_o.shape[0]
    # ----- side -----
    a = local_d[:, 0] ** 2 + local_d[:, 2] ** 2
    b = local_o[:, 0] * local_d[:, 0] + local_o[:, 2] * local_d[:, 2]
    c = local_o[:, 0] ** 2 + local_o[:, 2] ** 2 - 1.0
    disc = b * b - a * c
    sqd = np.sqrt(np.maximum(disc, 0.0))
    a_safe = np.where(a < 1e-9, 1e-9, a)
    t_side = (-b - sqd) / a_safe
    y_at_side = local_o[:, 1] + t_side * local_d[:, 1]
    side_valid = (a >= 1e-9) & (disc >= 0.0) & (t_side > 1e-4) & (np.abs(y_at_side) <= 1.0)

    # ----- caps -----
    eps = 1e-9
    dy_safe = np.where(np.abs(local_d[:, 1]) < eps, eps, local_d[:, 1])
    t_top = (1.0 - local_o[:, 1]) / dy_safe
    x_top = local_o[:, 0] + t_top * local_d[:, 0]
    z_top = local_o[:, 2] + t_top * local_d[:, 2]
    top_valid = (np.abs(local_d[:, 1]) >= eps) & (t_top > 1e-4) & (x_top ** 2 + z_top ** 2 <= 1.0)

    t_bot = (-1.0 - local_o[:, 1]) / dy_safe
    x_bot = local_o[:, 0] + t_bot * local_d[:, 0]
    z_bot = local_o[:, 2] + t_bot * local_d[:, 2]
    bot_valid = (np.abs(local_d[:, 1]) >= eps) & (t_bot > 1e-4) & (x_bot ** 2 + z_bot ** 2 <= 1.0)

    inf = np.inf
    ts = np.stack([
        np.where(side_valid, t_side, inf),
        np.where(top_valid, t_top, inf),
        np.where(bot_valid, t_bot, inf),
    ], axis=1)
    which = np.argmin(ts, axis=1)
    t = ts[np.arange(n_rays), which]
    valid = np.isfinite(t)
    # Replace infinite t's with 0 so local_o + t*d does not produce NaN.
    # We still use `valid` to discard those pixels downstream.
    t_safe = np.where(valid, t, 0.0)
    local_p = local_o + t_safe[:, None] * local_d

    # normals per case
    side_norm = np.stack([local_p[:, 0], np.zeros(n_rays), local_p[:, 2]], axis=1)
    side_norm = side_norm / np.maximum(
        np.linalg.norm(side_norm, axis=1, keepdims=True), 1e-9
    )
    top_norm = np.tile(np.array([0.0, 1.0, 0.0]), (n_rays, 1))
    bot_norm = np.tile(np.array([0.0, -1.0, 0.0]), (n_rays, 1))
    n = np.where(which[:, None] == 0, side_norm,
                 np.where(which[:, None] == 1, top_norm, bot_norm))
    return t, valid, n


def render_3d_scene(primitives: Sequence[Primitive], image_size: int = 64,
                    light_dir: np.ndarray | None = None,
                    background: float = 0.0) -> np.ndarray:
    """Lambertian render of a 3D scene to a grayscale (image_size, image_size)
    array in [0, 1].

    Camera: orthographic, image plane covers [-1.6, 1.6]^2, looking down -z
    from z=+5. Light: from camera direction (0, 0, +1) by default.
    """
    if light_dir is None:
        light_dir = np.array([0.0, 0.0, 1.0])
    light_dir = light_dir / max(np.linalg.norm(light_dir), 1e-9)

    n_pixels = image_size * image_size
    ray_o, ray_d = _ray_grid(image_size)

    # Per-pixel z-buffer in world coords
    best_t = np.full(n_pixels, np.inf)
    best_intensity = np.full(n_pixels, background)

    for prim in primitives:
        R = prim.rotation
        s = prim.scale
        # Transform world ray into the unit-primitive local frame:
        #   world_pt = R * (s * local_pt) + position
        # =>  local_pt = R^T (world_pt - position) / s
        local_o = (ray_o - prim.position) @ R / s
        local_d = ray_d @ R / s

        if prim.type == "sphere":
            t, valid, local_n = _intersect_sphere(local_o, local_d)
        elif prim.type == "cube":
            t, valid, local_n = _intersect_cube(local_o, local_d)
        elif prim.type == "cylinder":
            t, valid, local_n = _intersect_cylinder(local_o, local_d)
        else:
            raise ValueError(f"unknown primitive type: {prim.type}")

        # World normal (rotation only; uniform scale doesn't affect normal direction)
        local_n_norm = local_n / np.maximum(
            np.linalg.norm(local_n, axis=1, keepdims=True), 1e-9
        )
        world_n = local_n_norm @ R.T

        intensity = np.maximum(0.0, world_n @ light_dir)
        # Boost ambient slightly so silhouettes are visible
        intensity = 0.15 + 0.85 * intensity

        # Z-buffer: smaller t = closer to camera (rays go in -z)
        closer = valid & (t < best_t)
        best_t = np.where(closer, t, best_t)
        best_intensity = np.where(closer, intensity, best_intensity)

    return best_intensity.reshape(image_size, image_size).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    primitives: list[Primitive] = field(default_factory=list)
    image: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.primitives)


def sample_scene(rng: np.random.Generator, max_primitives: int = 3,
                 image_size: int = 64, scale: float = 0.4) -> Scene:
    """Sample a scene with 1..max_primitives primitives. We exclude n=0 to keep
    every image non-empty for evaluation; the inference network still has to
    decide presence per slot."""
    n = int(rng.integers(1, max_primitives + 1))
    prims: list[Primitive] = []
    for _ in range(n):
        t = PRIMITIVE_TYPES[int(rng.integers(0, len(PRIMITIVE_TYPES)))]
        position = np.array([
            rng.uniform(-0.9, 0.9),
            rng.uniform(-0.9, 0.9),
            rng.uniform(-0.3, 0.3),
        ])
        euler = np.array([
            rng.uniform(0.0, np.pi),
            rng.uniform(0.0, np.pi),
            rng.uniform(0.0, np.pi),
        ])
        prims.append(Primitive(type=t, position=position, euler=euler, scale=scale))

    # Sort by x for canonical slot assignment
    prims.sort(key=lambda p: p.position[0])
    image = render_3d_scene(prims, image_size=image_size)
    return Scene(primitives=prims, image=image)


def encode_targets(scene: Scene, max_slots: int = 3):
    """Encode a scene as fixed-size ground-truth tensors.

    Returns
    -------
    presence : (max_slots,) {0, 1}
    type_idx : (max_slots,) int in [0, 3); -1 for absent slots (ignored in loss)
    position : (max_slots, 3) float
    rotation : (max_slots, 3) float
    """
    presence = np.zeros(max_slots, dtype=np.float32)
    type_idx = -np.ones(max_slots, dtype=np.int64)
    position = np.zeros((max_slots, 3), dtype=np.float32)
    rotation = np.zeros((max_slots, 3), dtype=np.float32)
    for i, p in enumerate(scene.primitives[:max_slots]):
        presence[i] = 1.0
        type_idx[i] = TYPE_TO_INDEX[p.type]
        position[i] = p.position
        rotation[i] = p.euler
    return presence, type_idx, position, rotation


def generate_dataset(n_scenes: int, max_primitives: int = 3,
                     image_size: int = 64, seed: int = 0):
    """Synthesize a dataset of (image, ground-truth) pairs.

    Returns
    -------
    images   : (n_scenes, H, W) float32 in [0, 1]
    presence : (n_scenes, max_primitives) float32
    types    : (n_scenes, max_primitives) int64
    positions: (n_scenes, max_primitives, 3) float32
    rotations: (n_scenes, max_primitives, 3) float32
    """
    rng = np.random.default_rng(seed)
    images = np.zeros((n_scenes, image_size, image_size), dtype=np.float32)
    presence = np.zeros((n_scenes, max_primitives), dtype=np.float32)
    types = np.zeros((n_scenes, max_primitives), dtype=np.int64)
    positions = np.zeros((n_scenes, max_primitives, 3), dtype=np.float32)
    rotations = np.zeros((n_scenes, max_primitives, 3), dtype=np.float32)
    for i in range(n_scenes):
        scene = sample_scene(rng, max_primitives=max_primitives, image_size=image_size)
        images[i] = scene.image
        p, t, pos, rot = encode_targets(scene, max_slots=max_primitives)
        presence[i] = p
        types[i] = t
        positions[i] = pos
        rotations[i] = rot
    return images, presence, types, positions, rotations


# ---------------------------------------------------------------------------
# AIR-style inference network (numpy MLP)
# ---------------------------------------------------------------------------

def _xavier(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    scale = np.sqrt(2.0 / fan_in)
    return rng.standard_normal((fan_in, fan_out)).astype(np.float32) * scale


def _avg_pool(x: np.ndarray, factor: int) -> np.ndarray:
    """Mean-pool a (B, H, W) image stack by an integer factor on H and W."""
    if factor <= 1:
        return x
    B, H, W = x.shape
    H2, W2 = H // factor, W // factor
    x = x[:, :H2 * factor, :W2 * factor]
    x = x.reshape(B, H2, factor, W2, factor).mean(axis=(2, 4))
    return x


class AIR3DEncoder:
    """MLP that produces, per slot, the AIR latents:

      - presence logit (1)
      - type logits (3)
      - position (3)
      - rotation (3)

    Total per slot = 10.

    The image is optionally average-pooled by ``input_pool`` before flattening
    to keep the first FC layer compact. Forward pass keeps activations for
    backprop. Three fully-connected layers: pooled-image -> hidden -> hidden
    -> 10 * max_slots.
    """

    def __init__(self, image_size: int, max_slots: int = 3, hidden: int = 128,
                 input_pool: int = 2, seed: int = 0):
        self.image_size = image_size
        self.max_slots = max_slots
        self.hidden = hidden
        self.input_pool = input_pool
        self.pooled_size = image_size // input_pool if input_pool > 0 else image_size
        rng = np.random.default_rng(seed)
        in_dim = self.pooled_size * self.pooled_size
        out_dim = max_slots * 10
        self.W1 = _xavier(in_dim, hidden, rng)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = _xavier(hidden, hidden, rng)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.W3 = _xavier(hidden, out_dim, rng)
        self.b3 = np.zeros(out_dim, dtype=np.float32)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    @staticmethod
    def _relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def _drelu(x):
        return (x > 0.0).astype(np.float32)

    def _preprocess(self, images: np.ndarray) -> np.ndarray:
        """Accept (B, H, W), (B, H*W), or (H, W) and return (B, pooled_dim)."""
        if images.ndim == 2 and images.shape == (self.image_size, self.image_size):
            images = images[None]
        if images.ndim == 2:
            # (B, H*W)
            B = images.shape[0]
            images = images.reshape(B, self.image_size, self.image_size)
        if self.input_pool > 1:
            images = _avg_pool(images, self.input_pool)
        return images.reshape(images.shape[0], -1).astype(np.float32)

    def forward(self, x: np.ndarray):
        """x: (B, H, W) or (B, H*W). Returns dict of activations."""
        x_flat = self._preprocess(x)
        z1 = x_flat @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._relu(z2)
        z3 = a2 @ self.W3 + self.b3  # (B, max_slots*10)
        out = z3.reshape(-1, self.max_slots, 10)
        return {
            "x": x_flat, "z1": z1, "a1": a1,
            "z2": z2, "a2": a2, "z3": z3, "out": out,
        }

    @staticmethod
    def split_heads(out: np.ndarray):
        # out: (B, S, 10)
        pres_logit = out[..., 0]               # (B, S)
        type_logits = out[..., 1:4]            # (B, S, 3)
        position = out[..., 4:7]               # (B, S, 3)
        rotation = out[..., 7:10]              # (B, S, 3)
        return pres_logit, type_logits, position, rotation

    def decode(self, image: np.ndarray) -> list[Primitive]:
        """Predict primitives for a single image (or list if batched).

        Decoding rule: keep slot k iff sigmoid(presence) >= 0.5. Type =
        argmax of softmax(type_logits). Position and rotation read directly
        from heads. We clip rotation to [0, pi] and position to [-1, 1] for
        sanity.
        """
        single = image.ndim == 2
        if single:
            image = image[None]
        out = self.forward(image)["out"]
        pres_logit, type_logits, position, rotation = self.split_heads(out)
        sigmoid = 1.0 / (1.0 + np.exp(-pres_logit))
        preds: list[list[Primitive]] = []
        for b in range(out.shape[0]):
            scene_prims: list[Primitive] = []
            for k in range(self.max_slots):
                if sigmoid[b, k] >= 0.5:
                    t = PRIMITIVE_TYPES[int(np.argmax(type_logits[b, k]))]
                    pos = np.clip(position[b, k], -1.0, 1.0)
                    rot = np.clip(rotation[b, k], 0.0, np.pi)
                    scene_prims.append(
                        Primitive(type=t, position=pos.astype(np.float64),
                                  euler=rot.astype(np.float64))
                    )
            preds.append(scene_prims)
        if single:
            return preds[0]
        return preds


# ---------------------------------------------------------------------------
# Loss + backward
# ---------------------------------------------------------------------------

def _bce_with_logits(logits: np.ndarray, target: np.ndarray):
    # log(1 + exp(-|x|)) + max(x, 0) - x*t
    max_l = np.maximum(logits, 0.0)
    loss = max_l - logits * target + np.log1p(np.exp(-np.abs(logits)))
    grad = (1.0 / (1.0 + np.exp(-logits))) - target
    return loss, grad


def _softmax_ce(logits: np.ndarray, target_idx: np.ndarray, mask: np.ndarray):
    """Cross-entropy with mask; logits (B, S, C), target_idx (B, S), mask (B, S).

    Returns
    -------
    loss_sum : scalar (sum over masked entries)
    grad     : (B, S, C); zero where mask is 0
    """
    B, S, C = logits.shape
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    probs = e / np.sum(e, axis=-1, keepdims=True)
    # gather log-probs at target indices (only where mask=1)
    safe_idx = np.where(mask.astype(bool), target_idx, 0)
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    nll = -np.take_along_axis(log_probs, safe_idx[..., None], axis=-1)[..., 0]
    nll = nll * mask
    loss_sum = nll.sum()
    one_hot = np.zeros_like(probs)
    np.put_along_axis(one_hot, safe_idx[..., None], 1.0, axis=-1)
    grad = (probs - one_hot) * mask[..., None]
    return loss_sum, grad


def compute_loss(out: np.ndarray, presence: np.ndarray, type_idx: np.ndarray,
                 position: np.ndarray, rotation: np.ndarray,
                 weights=(2.0, 2.0, 1.0, 0.3)):
    """Compute the supervised AIR-style loss.

    weights = (presence, type, position, rotation).

    Rotation loss is masked out for sphere slots since a unit sphere is
    rotationally symmetric -- the orientation is unrecoverable from any
    image and we should not penalize the network for guessing wrong.

    Returns (total_loss_scalar, grad_out_same_shape_as_out, components_dict).
    """
    B, S, _ = out.shape
    w_pres, w_type, w_pos, w_rot = weights
    pres_logit, type_logits, position_p, rotation_p = AIR3DEncoder.split_heads(out)

    # Presence: BCE for every slot (target 0/1)
    bce_loss, bce_grad = _bce_with_logits(pres_logit, presence)
    loss_pres = bce_loss.mean()
    grad_pres = bce_grad / (B * S)  # mean over (B, S)

    # Type: CE only where presence=1
    mask = presence
    ce_sum, ce_grad = _softmax_ce(type_logits, type_idx, mask)
    n_present = max(mask.sum(), 1.0)
    loss_type = ce_sum / n_present
    grad_type = ce_grad / n_present  # (B, S, 3)

    # Position: MSE only where presence=1
    diff_pos = position_p - position
    sq_pos = (diff_pos ** 2) * mask[..., None]
    loss_pos = sq_pos.sum() / (3.0 * n_present)
    grad_position = 2.0 * diff_pos * mask[..., None] / (3.0 * n_present)

    # Rotation: MSE only where presence=1 AND type != sphere
    sphere_idx = TYPE_TO_INDEX["sphere"]
    not_sphere = (type_idx != sphere_idx).astype(np.float32)
    rot_mask = mask * not_sphere
    n_rot = max(rot_mask.sum(), 1.0)
    diff_rot = rotation_p - rotation
    sq_rot = (diff_rot ** 2) * rot_mask[..., None]
    loss_rot = sq_rot.sum() / (3.0 * n_rot)
    grad_rotation = 2.0 * diff_rot * rot_mask[..., None] / (3.0 * n_rot)

    total = (
        w_pres * loss_pres
        + w_type * loss_type
        + w_pos * loss_pos
        + w_rot * loss_rot
    )

    grad_out = np.zeros_like(out)
    grad_out[..., 0] = w_pres * grad_pres
    grad_out[..., 1:4] = w_type * grad_type
    grad_out[..., 4:7] = w_pos * grad_position
    grad_out[..., 7:10] = w_rot * grad_rotation

    return total, grad_out, {
        "presence": float(loss_pres),
        "type": float(loss_type),
        "position": float(loss_pos),
        "rotation": float(loss_rot),
    }


def backward(model: AIR3DEncoder, cache: dict, grad_out: np.ndarray):
    """Backprop through the 3-layer MLP. Returns gradients for all params."""
    B = grad_out.shape[0]
    grad_z3 = grad_out.reshape(B, -1)  # (B, max_slots*10)
    a2 = cache["a2"]
    grad_W3 = a2.T @ grad_z3
    grad_b3 = grad_z3.sum(axis=0)
    grad_a2 = grad_z3 @ model.W3.T

    z2 = cache["z2"]
    grad_z2 = grad_a2 * AIR3DEncoder._drelu(z2)
    a1 = cache["a1"]
    grad_W2 = a1.T @ grad_z2
    grad_b2 = grad_z2.sum(axis=0)
    grad_a1 = grad_z2 @ model.W2.T

    z1 = cache["z1"]
    grad_z1 = grad_a1 * AIR3DEncoder._drelu(z1)
    x = cache["x"]
    grad_W1 = x.T @ grad_z1
    grad_b1 = grad_z1.sum(axis=0)

    return [grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3]


# ---------------------------------------------------------------------------
# Training loop (Adam)
# ---------------------------------------------------------------------------

def train(model: AIR3DEncoder, dataset: dict, n_epochs: int = 30,
          batch_size: int = 32, lr: float = 1e-3, seed: int = 0,
          val_split: float = 0.2, weight_decay: float = 5e-4,
          verbose: bool = True):
    """Train the AIR encoder with Adam + decoupled weight decay. Tracks the
    best val checkpoint and restores the network's params to it after the run.
    Returns (history, val_idx, tr_idx).
    """
    rng = np.random.default_rng(seed + 9_999)

    images = dataset["images"]
    presence = dataset["presence"].astype(np.float32)
    types = dataset["types"].astype(np.int64)
    positions = dataset["positions"].astype(np.float32)
    rotations = dataset["rotations"].astype(np.float32)

    n = images.shape[0]
    n_val = int(n * val_split)
    perm = rng.permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    n_train = tr_idx.size

    params = model.params()
    # Decay weight tensors only (W_*), not biases
    is_weight = [p.ndim >= 2 for p in params]
    m_state = [np.zeros_like(p) for p in params]
    v_state = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0

    history = {"epoch": [], "train_loss": [], "val_loss": [], "components": []}
    best_val = float("inf")
    best_params = [p.copy() for p in params]
    best_epoch = -1

    for epoch in range(n_epochs):
        order = rng.permutation(n_train)
        train_loss_sum = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = tr_idx[order[start:start + batch_size]]
            x = images[idx]
            cache = model.forward(x)
            loss, grad_out, _components = compute_loss(
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
            train_loss_sum += float(loss)
            n_batches += 1
        train_loss = train_loss_sum / max(n_batches, 1)

        # Val
        x_val = images[val_idx]
        cache_val = model.forward(x_val)
        val_loss, _, components = compute_loss(
            cache_val["out"], presence[val_idx], types[val_idx],
            positions[val_idx], rotations[val_idx],
        )
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(val_loss))
        history["components"].append({k: float(v) for k, v in components.items()})

        if float(val_loss) < best_val:
            best_val = float(val_loss)
            best_params = [p.copy() for p in params]
            best_epoch = epoch

        if verbose and (epoch < 3 or (epoch + 1) % max(1, n_epochs // 10) == 0
                        or epoch == n_epochs - 1):
            print(f"  epoch {epoch:3d}  train={train_loss:.4f}  "
                  f"val={float(val_loss):.4f}  "
                  f"[pres={components['presence']:.3f} type={components['type']:.3f} "
                  f"pos={components['position']:.3f} rot={components['rotation']:.3f}]")

    # Restore best params
    for p, bp in zip(params, best_params):
        p[...] = bp
    if verbose:
        print(f"      restored best val checkpoint from epoch {best_epoch} "
              f"(val={best_val:.4f})")
    history["best_epoch"] = best_epoch
    history["best_val"] = best_val
    return history, val_idx, tr_idx


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: AIR3DEncoder, dataset: dict, indices: np.ndarray):
    """Evaluate an AIR encoder on a subset of the dataset.

    Reports
    -------
    count_acc : exact match on number of primitives.
    presence_acc : per-slot binary accuracy.
    type_acc : type accuracy averaged over present slots.
    pos_mae : mean absolute error per axis on (x, y, z).
    rot_mae : mean absolute error per Euler axis (mod pi).
    """
    images = dataset["images"][indices]
    presence = dataset["presence"][indices]
    types = dataset["types"][indices]
    positions = dataset["positions"][indices]
    rotations = dataset["rotations"][indices]

    out = model.forward(images)["out"]
    pres_logit, type_logits, pos_p, rot_p = AIR3DEncoder.split_heads(out)
    pres_pred = (pres_logit >= 0.0).astype(np.float32)

    presence_acc = float((pres_pred == presence).mean())
    n_pred = pres_pred.sum(axis=1)
    n_true = presence.sum(axis=1)
    count_acc = float((n_pred == n_true).mean())

    type_pred = np.argmax(type_logits, axis=-1)
    mask = presence.astype(bool)
    if mask.sum() > 0:
        type_acc = float((type_pred[mask] == types[mask]).mean())
        pos_mae_per_axis = np.abs(pos_p - positions)[mask].mean(axis=0)
        # Rotation: angles are mod pi (since axis-angle ambiguity, but here we
        # have Euler angles in [0, pi] so it's the natural range)
        rot_diff = np.minimum(np.abs(rot_p - rotations),
                              np.pi - np.abs(rot_p - rotations))
        rot_mae_per_axis = rot_diff[mask].mean(axis=0)
    else:
        type_acc = float("nan")
        pos_mae_per_axis = np.full(3, np.nan)
        rot_mae_per_axis = np.full(3, np.nan)

    return {
        "count_acc": count_acc,
        "presence_acc": presence_acc,
        "type_acc": type_acc,
        "pos_mae_xyz": [float(v) for v in pos_mae_per_axis],
        "rot_mae_xyz": [float(v) for v in rot_mae_per_axis],
    }


# ---------------------------------------------------------------------------
# Public API + CLI
# ---------------------------------------------------------------------------

def build_air_model_3d(image_size: int, max_slots: int = 3, hidden: int = 128,
                       input_pool: int = 2, seed: int = 0) -> AIR3DEncoder:
    """Public constructor matching the spec."""
    return AIR3DEncoder(image_size=image_size, max_slots=max_slots,
                        hidden=hidden, input_pool=input_pool, seed=seed)


def run(seed: int = 0, image_size: int = 64, max_primitives: int = 3,
        n_epochs: int = 60, n_train: int = 2000, n_test: int = 400,
        hidden: int = 128, input_pool: int = 2,
        batch_size: int = 32, lr: float = 1e-3, weight_decay: float = 5e-4,
        save_weights: str | None = "weights.npz",
        save_results: str | None = "results.json",
        verbose: bool = True):
    """End-to-end: synthesize -> train -> evaluate -> persist."""
    t0 = time.time()
    if verbose:
        print(f"[1/4] Synthesizing {n_train + n_test} scenes "
              f"({image_size}x{image_size}, max {max_primitives} primitives)...")
    images, presence, types, positions, rotations = generate_dataset(
        n_train + n_test, max_primitives=max_primitives,
        image_size=image_size, seed=seed,
    )
    train_dataset = {
        "images": images[:n_train], "presence": presence[:n_train],
        "types": types[:n_train], "positions": positions[:n_train],
        "rotations": rotations[:n_train],
    }
    test_dataset = {
        "images": images[n_train:], "presence": presence[n_train:],
        "types": types[n_train:], "positions": positions[n_train:],
        "rotations": rotations[n_train:],
    }
    t_synth = time.time() - t0

    if verbose:
        print(f"      synth wallclock: {t_synth:.1f}s")
        print(f"[2/4] Building AIR-3D encoder (image_size={image_size}, "
              f"max_slots={max_primitives}, hidden={hidden}, "
              f"input_pool={input_pool})")
    model = build_air_model_3d(image_size=image_size, max_slots=max_primitives,
                               hidden=hidden, input_pool=input_pool, seed=seed)

    if verbose:
        print(f"[3/4] Training for {n_epochs} epochs...")
    t1 = time.time()
    history, val_idx, tr_idx = train(
        model, train_dataset, n_epochs=n_epochs,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay, seed=seed,
        verbose=verbose,
    )
    t_train = time.time() - t1

    if verbose:
        print(f"      train wallclock: {t_train:.1f}s")
        print("[4/4] Evaluating on held-out test set...")
    metrics = evaluate(model, test_dataset, np.arange(n_test))

    total = time.time() - t0
    result = {
        "config": {
            "seed": seed,
            "image_size": image_size,
            "max_primitives": max_primitives,
            "n_epochs": n_epochs,
            "n_train": n_train,
            "n_test": n_test,
            "hidden": hidden,
            "input_pool": input_pool,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
        },
        "metrics": metrics,
        "wallclock": {
            "synth_s": t_synth,
            "train_s": t_train,
            "total_s": total,
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
        },
        "history": history,
    }
    if verbose:
        print()
        print("Results:")
        print(f"  count accuracy     : {metrics['count_acc']:.3f}")
        print(f"  presence (per-slot): {metrics['presence_acc']:.3f}")
        print(f"  type accuracy      : {metrics['type_acc']:.3f}")
        print(f"  position MAE (x,y,z): {metrics['pos_mae_xyz']}")
        print(f"  rotation MAE (a,b,g): {metrics['rot_mae_xyz']}")
        print(f"  wallclock          : synth {t_synth:.1f}s + train "
              f"{t_train:.1f}s = {total:.1f}s")

    if save_weights is not None:
        np.savez(save_weights,
                 W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2,
                 W3=model.W3, b3=model.b3,
                 image_size=np.int64(image_size),
                 max_slots=np.int64(max_primitives),
                 hidden=np.int64(hidden),
                 input_pool=np.int64(input_pool))
    if save_results is not None:
        with open(save_results, "w") as f:
            json.dump(result, f, indent=2)
    return result, model


def load_model(path: str) -> AIR3DEncoder:
    z = np.load(path)
    input_pool = int(z["input_pool"]) if "input_pool" in z.files else 1
    model = AIR3DEncoder(
        image_size=int(z["image_size"]),
        max_slots=int(z["max_slots"]),
        hidden=int(z["hidden"]),
        input_pool=input_pool,
        seed=0,
    )
    model.W1 = z["W1"]; model.b1 = z["b1"]
    model.W2 = z["W2"]; model.b2 = z["b2"]
    model.W3 = z["W3"]; model.b3 = z["b3"]
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--max-primitives", type=int, default=3)
    ap.add_argument("--n-epochs", type=int, default=60)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--input-pool", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    args = ap.parse_args()

    run(
        seed=args.seed,
        image_size=args.image_size,
        max_primitives=args.max_primitives,
        n_epochs=args.n_epochs,
        n_train=args.n_train,
        n_test=args.n_test,
        hidden=args.hidden,
        input_pool=args.input_pool,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


if __name__ == "__main__":
    main()
