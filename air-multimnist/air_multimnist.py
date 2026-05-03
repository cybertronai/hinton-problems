"""AIR (Attend, Infer, Repeat) on Multi-MNIST scenes -- pure numpy.

Eslami, Heess, Weber, Tassa, Szepesvari, Kavukcuoglu & Hinton (NIPS 2016).

The full AIR model is recurrent: at each step t the model decides
  z_pres_t   -- whether to attend to another object (Bernoulli, drives count)
  z_where_t  -- 3-D affine (log_s, tx, ty) for spatial transformer
  z_what_t   -- VAE latent for object appearance
A spatial transformer renders each decoded patch back onto the canvas at
z_where_t. Total reconstruction = sum over active steps of rendered patches.

Pure-numpy pragmatic reduction (deviations from paper, all documented):
  - 32x32 canvas, 0-2 digits (paper: 50x50)
  - what_dim=20 (paper: 50)
  - Per-step heads with shared global encoder, NOT a recurrent RNN over
    image residuals. Pure-numpy backprop through an LSTM x 3 timesteps over
    the spatial transformer is too slow per step; per-step heads on a
    global encoder still demonstrate counting + per-digit decode.
  - Gumbel-softmax (sigmoid relaxation) for z_pres throughout (paper uses
    REINFORCE for the discrete count). This gives a continuous proxy whose
    gradient flows via reparameterization in the same spirit as IWAE.
  - Independent Bernoulli prior per step with rates p_t = (0.7, 0.5, 0.3),
    a coarse proxy for the geometric chain prior in the paper.
  - 1k pre-rendered training scenes (paper: 60M streamed)
  - Spatial transformer = inverse-affine bilinear sampler (no forward STN
    on the encoder side; the encoder is a global MLP).
"""
from __future__ import annotations
import argparse
import gzip
import json
import os
import platform
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# MNIST loader (urllib + gzip; cached at ~/.cache/hinton-mnist/)
# ----------------------------------------------------------------------

MNIST_URL_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}
CACHE_DIR = Path.home() / ".cache" / "hinton-mnist"


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"  downloading {url} -> {dest}", flush=True)
    with urllib.request.urlopen(url) as r:
        data = r.read()
    dest.write_bytes(data)


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    n = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    if magic != 2051:
        raise RuntimeError(f"bad MNIST images magic: {magic}")
    return np.frombuffer(data[16:], dtype=np.uint8).reshape(n, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    n = int.from_bytes(data[4:8], "big")
    if magic != 2049:
        raise RuntimeError(f"bad MNIST labels magic: {magic}")
    return np.frombuffer(data[8:], dtype=np.uint8)


def load_mnist(split: str = "train"):
    img_key = "train_images" if split == "train" else "test_images"
    lab_key = "train_labels" if split == "train" else "test_labels"
    images_dest = CACHE_DIR / MNIST_FILES[img_key]
    labels_dest = CACHE_DIR / MNIST_FILES[lab_key]
    _download(MNIST_URL_BASE + MNIST_FILES[img_key], images_dest)
    _download(MNIST_URL_BASE + MNIST_FILES[lab_key], labels_dest)
    images = _read_idx_images(images_dest).astype(np.float32) / 255.0
    labels = _read_idx_labels(labels_dest).astype(np.int64)
    return images, labels


# ----------------------------------------------------------------------
# Scene generation
# ----------------------------------------------------------------------

def _resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize a single 2D image."""
    h, w = img.shape
    if h == out_h and w == out_w:
        return img.copy()
    yy = np.linspace(0, h - 1, out_h, dtype=np.float32)
    xx = np.linspace(0, w - 1, out_w, dtype=np.float32)
    y0 = np.floor(yy).astype(np.int32); y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.floor(xx).astype(np.int32); x1 = np.minimum(x0 + 1, w - 1)
    wy = (yy - y0)[:, None]
    wx = (xx - x0)[None, :]
    p00 = img[np.ix_(y0, x0)]
    p01 = img[np.ix_(y0, x1)]
    p10 = img[np.ix_(y1, x0)]
    p11 = img[np.ix_(y1, x1)]
    return ((1 - wy) * (1 - wx) * p00 +
            (1 - wy) *      wx  * p01 +
                 wy  * (1 - wx) * p10 +
                 wy  *      wx  * p11).astype(np.float32)


def _paste(canvas: np.ndarray, sprite: np.ndarray, top: int, left: int):
    """In-place pixel-wise max of sprite onto canvas at (top, left)."""
    Hc, Wc = canvas.shape
    Hs, Ws = sprite.shape
    src_y0 = max(0, -top); src_x0 = max(0, -left)
    src_y1 = min(Hs, Hc - top); src_x1 = min(Ws, Wc - left)
    dst_y0 = max(0, top); dst_x0 = max(0, left)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    if src_y1 > src_y0 and src_x1 > src_x0:
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = np.maximum(
            canvas[dst_y0:dst_y1, dst_x0:dst_x1],
            sprite[src_y0:src_y1, src_x0:src_x1],
        )


def render_scene(images: np.ndarray, labels: np.ndarray,
                 canvas_size: int = 32,
                 n_digits_options=(0, 1, 2),
                 min_digit_size: int = 12,
                 max_digit_size: int = 18,
                 rng: np.random.Generator | None = None):
    """Place 0/1/2 MNIST digits at random scale & position on a blank canvas.

    Returns (canvas (Hc, Wc) float32 in [0, 1], list of dicts with the
    ground-truth (label, top, left, size) per placed digit, and n_digits).
    """
    rng = rng or np.random.default_rng()
    n_digits = int(rng.choice(n_digits_options))
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    placements = []
    for _ in range(n_digits):
        idx = int(rng.integers(0, images.shape[0]))
        size = int(rng.integers(min_digit_size, max_digit_size + 1))
        sprite = _resize_bilinear(images[idx], size, size)
        top = int(rng.integers(0, canvas_size - size + 1))
        left = int(rng.integers(0, canvas_size - size + 1))
        _paste(canvas, sprite, top, left)
        placements.append(dict(label=int(labels[idx]),
                               top=top, left=left, size=size))
    return canvas, placements, n_digits


def generate_scenes(n_scenes: int, images: np.ndarray, labels: np.ndarray,
                    canvas_size: int = 32, rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng()
    canvases = np.zeros((n_scenes, canvas_size, canvas_size), dtype=np.float32)
    counts = np.zeros((n_scenes,), dtype=np.int64)
    placements = []
    for i in range(n_scenes):
        c, plc, n = render_scene(images, labels, canvas_size, rng=rng)
        canvases[i] = c
        counts[i] = n
        placements.append(plc)
    return canvases, counts, placements


# ----------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


# ----------------------------------------------------------------------
# Spatial transformer (inverse-affine bilinear sampler)
# ----------------------------------------------------------------------
# Given a patch (B, Hp, Wp) and z_where (B, 3) = (log_s, tx, ty), render
# onto a canvas (B, Hc, Wc). The transform maps each canvas pixel at
# normalized coord (u, v) in [-1, 1] to a patch normalized coord:
#
#   up = (u - tx) / s
#   vp = (v - ty) / s              with s = exp(log_s) > 0
#
# which is then converted to patch pixel coord and bilinearly sampled.
# The patch covers a [tx-s, tx+s] x [ty-s, ty+s] region of the canvas
# (in normalized coords). s is the half-width.

def render_forward(patch: np.ndarray, z_where: np.ndarray, canvas_size: int):
    B, Hp, Wp = patch.shape
    Hc = Wc = canvas_size

    log_s = z_where[:, 0]
    tx    = z_where[:, 1]
    ty    = z_where[:, 2]
    s = np.exp(np.clip(log_s, -3.0, 1.5)).astype(np.float32)  # s in ~[0.05, 4.5]

    # canvas grid in normalized coords
    u = np.linspace(-1, 1, Wc, dtype=np.float32)
    v = np.linspace(-1, 1, Hc, dtype=np.float32)
    U, V = np.meshgrid(u, v)        # (Hc, Wc) each

    s_b  = s [:, None, None]
    tx_b = tx[:, None, None]
    ty_b = ty[:, None, None]

    up = (U[None] - tx_b) / s_b      # (B, Hc, Wc)
    vp = (V[None] - ty_b) / s_b

    # patch pixel coord (continuous)
    x = (up + 1.0) * (Wp - 1) / 2.0
    y = (vp + 1.0) * (Hp - 1) / 2.0

    x0 = np.floor(x).astype(np.int32); y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1;                       y1 = y0 + 1
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)

    m00 = ((x0 >= 0) & (x0 < Wp) & (y0 >= 0) & (y0 < Hp)).astype(np.float32)
    m01 = ((x1 >= 0) & (x1 < Wp) & (y0 >= 0) & (y0 < Hp)).astype(np.float32)
    m10 = ((x0 >= 0) & (x0 < Wp) & (y1 >= 0) & (y1 < Hp)).astype(np.float32)
    m11 = ((x1 >= 0) & (x1 < Wp) & (y1 >= 0) & (y1 < Hp)).astype(np.float32)

    x0c = np.clip(x0, 0, Wp - 1)
    x1c = np.clip(x1, 0, Wp - 1)
    y0c = np.clip(y0, 0, Hp - 1)
    y1c = np.clip(y1, 0, Hp - 1)

    bidx = np.arange(B)[:, None, None]
    P00 = patch[bidx, y0c, x0c] * m00
    P01 = patch[bidx, y0c, x1c] * m01
    P10 = patch[bidx, y1c, x0c] * m10
    P11 = patch[bidx, y1c, x1c] * m11

    canvas = ((1 - wy) * (1 - wx) * P00 +
              (1 - wy) *      wx  * P01 +
                   wy  * (1 - wx) * P10 +
                   wy  *      wx  * P11).astype(np.float32)

    cache = dict(patch=patch, s=s, up=up, vp=vp,
                 wx=wx, wy=wy, x0c=x0c, x1c=x1c, y0c=y0c, y1c=y1c,
                 m00=m00, m01=m01, m10=m10, m11=m11,
                 P00=P00, P01=P01, P10=P10, P11=P11,
                 Hp=Hp, Wp=Wp)
    return canvas, cache


def render_backward(d_canvas: np.ndarray, cache: dict):
    """Returns (d_patch (B, Hp, Wp), d_z_where (B, 3))."""
    patch = cache["patch"]
    B, Hp, Wp = patch.shape
    wx, wy = cache["wx"], cache["wy"]
    P00, P01, P10, P11 = cache["P00"], cache["P01"], cache["P10"], cache["P11"]
    m00, m01, m10, m11 = cache["m00"], cache["m01"], cache["m10"], cache["m11"]
    x0c, x1c, y0c, y1c = cache["x0c"], cache["x1c"], cache["y0c"], cache["y1c"]
    s = cache["s"]; up = cache["up"]; vp = cache["vp"]

    # gradient wrt the four corner samples (each pre-masked: P0x already 0 outside)
    dP00 = d_canvas * (1 - wy) * (1 - wx) * m00
    dP01 = d_canvas * (1 - wy) *      wx  * m01
    dP10 = d_canvas *      wy  * (1 - wx) * m10
    dP11 = d_canvas *      wy  *      wx  * m11

    d_patch = np.zeros_like(patch)
    bidx = np.arange(B)[:, None, None]
    np.add.at(d_patch, (bidx, y0c, x0c), dP00)
    np.add.at(d_patch, (bidx, y0c, x1c), dP01)
    np.add.at(d_patch, (bidx, y1c, x0c), dP10)
    np.add.at(d_patch, (bidx, y1c, x1c), dP11)

    # gradient wrt sub-pixel offsets
    d_wx = d_canvas * (-(1 - wy) * P00 + (1 - wy) * P01 - wy * P10 + wy * P11)
    d_wy = d_canvas * (-(1 - wx) * P00 - wx * P01 + (1 - wx) * P10 + wx * P11)

    # x = (up + 1) * (Wp - 1) / 2; wx = x - x0; treating x0 as constant of x for wx
    d_x = d_wx
    d_y = d_wy
    d_up = d_x * (Wp - 1) / 2.0
    d_vp = d_y * (Hp - 1) / 2.0

    s_b = s[:, None, None]
    d_tx_grid = -d_up / s_b
    d_ty_grid = -d_vp / s_b
    d_s_grid  = -(d_up * up + d_vp * vp) / s_b

    d_tx = d_tx_grid.sum(axis=(1, 2))
    d_ty = d_ty_grid.sum(axis=(1, 2))
    d_s  = d_s_grid.sum(axis=(1, 2))
    d_log_s = d_s * s  # s = exp(log_s)

    d_z_where = np.stack([d_log_s, d_tx, d_ty], axis=1).astype(np.float32)
    return d_patch.astype(np.float32), d_z_where


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

class AIR:
    """Pure-numpy AIR model.

    Architecture:
      Encoder           : 1024 -> 200 -> 100  (ReLU, ReLU)
      Per-step heads    : 100 -> 1 (z_pres_logit), 100 -> 3 (z_where),
                          100 -> 20 (z_what_mu), 100 -> 20 (z_what_logvar)
      VAE decoder       : 20 -> 100 -> 256 (ReLU, sigmoid) reshaped to 16x16
      Spatial transformer renders each step's 16x16 patch to the 32x32 canvas
      at (z_where_t).

    Sampling and reparameterization:
      z_what_t = mu_t + exp(0.5 * logvar_t) * eps,  eps ~ N(0, I)
      z_pres_t = sigmoid((z_pres_logit_t + g) / temp)  # Gumbel relaxation
                  with g = log(u) - log(1 - u),  u ~ Uniform(0, 1)

    Reconstruction:
      cumulative_pres_0 = 1
      cumulative_pres_t = cumulative_pres_{t-1} * z_pres_t
      recon = sum_t  cumulative_pres_t  *  STN(decode(z_what_t), z_where_t)

    The cumulative-pres factor encodes "use earlier slots first": once an
    earlier z_pres becomes near 0, all later slots are masked out.
    """

    def __init__(self,
                 canvas_size: int = 32,
                 max_steps: int = 3,
                 what_dim: int = 20,
                 patch_size: int = 16,
                 enc_h1: int = 200,
                 enc_h2: int = 100,
                 dec_h: int = 100,
                 init_log_s: float = -0.7,
                 pres_priors=(0.5, 0.4, 0.2),
                 seed: int = 0):
        self.canvas_size = canvas_size
        self.max_steps = max_steps
        self.what_dim = what_dim
        self.patch_size = patch_size
        self.enc_h1 = enc_h1
        self.enc_h2 = enc_h2
        self.dec_h  = dec_h
        self.init_log_s = init_log_s
        self.pres_priors = list(pres_priors)
        assert len(self.pres_priors) == max_steps
        self.rng = np.random.default_rng(seed)

        D_in = canvas_size * canvas_size
        D_patch = patch_size * patch_size
        H1, H2 = enc_h1, enc_h2

        def he(shape, fan_in):
            return (self.rng.standard_normal(shape) *
                    np.sqrt(2.0 / fan_in)).astype(np.float32)

        # Encoder
        self.W_e1 = he((D_in, H1), D_in)
        self.b_e1 = np.zeros((H1,), dtype=np.float32)
        self.W_e2 = he((H1, H2), H1)
        self.b_e2 = np.zeros((H2,), dtype=np.float32)

        # Per-step heads (T independent linear heads on the H2 features)
        T = max_steps
        # z_pres logit
        self.W_pres = he((T, H2, 1), H2)
        self.b_pres = np.zeros((T, 1), dtype=np.float32)
        # z_where: (log_s, tx, ty); init bias for (log_s) so patch starts smaller
        self.W_where = (0.01 * self.rng.standard_normal((T, H2, 3))).astype(np.float32)
        self.b_where = np.zeros((T, 3), dtype=np.float32)
        self.b_where[:, 0] = init_log_s
        # z_what: mu, logvar
        self.W_what_mu = he((T, H2, what_dim), H2) * 0.5
        self.b_what_mu = np.zeros((T, what_dim), dtype=np.float32)
        self.W_what_lv = (0.01 * self.rng.standard_normal((T, H2, what_dim))).astype(np.float32)
        self.b_what_lv = np.zeros((T, what_dim), dtype=np.float32)

        # Decoder (shared across steps)
        self.W_d1 = he((what_dim, dec_h), what_dim)
        self.b_d1 = np.zeros((dec_h,), dtype=np.float32)
        self.W_d2 = he((dec_h, D_patch), dec_h)
        self.b_d2 = np.zeros((D_patch,), dtype=np.float32) - 2.0  # init dark patch

    @property
    def param_names(self):
        return ("W_e1", "b_e1", "W_e2", "b_e2",
                "W_pres", "b_pres",
                "W_where", "b_where",
                "W_what_mu", "b_what_mu",
                "W_what_lv", "b_what_lv",
                "W_d1", "b_d1", "W_d2", "b_d2")

    def zero_like_params(self):
        return {k: np.zeros_like(getattr(self, k)) for k in self.param_names}

    # -- forward ----------------------------------------------------------

    def encode(self, x: np.ndarray):
        """x: (B, D_in). Returns (h2 (B, H2), cache)."""
        a1 = x @ self.W_e1 + self.b_e1
        h1 = relu(a1)
        a2 = h1 @ self.W_e2 + self.b_e2
        h2 = relu(a2)
        return h2, dict(x=x, a1=a1, h1=h1, a2=a2, h2=h2)

    def decode(self, z_what: np.ndarray):
        """z_what: (B, what_dim). Returns (patch (B, P, P), cache)."""
        d1 = z_what @ self.W_d1 + self.b_d1
        h1 = relu(d1)
        d2 = h1 @ self.W_d2 + self.b_d2
        p_flat = sigmoid(d2)
        patch = p_flat.reshape(-1, self.patch_size, self.patch_size)
        return patch, dict(z_what=z_what, d1=d1, h1=h1, d2=d2, p_flat=p_flat)

    def forward(self, x_img: np.ndarray, *, eps_what=None, gumbel=None,
                temp: float = 0.5, deterministic: bool = False):
        """x_img: (B, Hc, Wc). Returns (recon, cache).

        If `deterministic`, z_pres = sigmoid(logit) and z_what = mu (no noise).
        Otherwise eps_what and gumbel must be supplied (B, T, what_dim) and
        (B, T) respectively, or will be drawn from rng.
        """
        B, Hc, Wc = x_img.shape
        assert Hc == self.canvas_size and Wc == self.canvas_size
        x_flat = x_img.reshape(B, -1)
        h2, enc_cache = self.encode(x_flat)

        T = self.max_steps
        D = self.what_dim

        if eps_what is None:
            eps_what = self.rng.standard_normal((B, T, D)).astype(np.float32)
        if gumbel is None:
            u = self.rng.uniform(1e-6, 1 - 1e-6, size=(B, T)).astype(np.float32)
            gumbel = (np.log(u) - np.log(1 - u)).astype(np.float32)

        recon = np.zeros((B, Hc, Wc), dtype=np.float32)
        cumpres = np.ones((B,), dtype=np.float32)
        step_caches = []
        cumpres_history = []
        eff_pres_history = []
        z_pres_history = []
        for t in range(T):
            # Heads
            pres_logit = (h2 @ self.W_pres[t] + self.b_pres[t])[:, 0]   # (B,)
            where      =  h2 @ self.W_where[t] + self.b_where[t]         # (B, 3)
            what_mu    =  h2 @ self.W_what_mu[t] + self.b_what_mu[t]    # (B, D)
            what_lv    =  h2 @ self.W_what_lv[t] + self.b_what_lv[t]    # (B, D)
            what_lv = np.clip(what_lv, -8.0, 4.0)

            if deterministic:
                z_what = what_mu
                z_pres = sigmoid(pres_logit)
            else:
                z_what = what_mu + np.exp(0.5 * what_lv) * eps_what[:, t]
                z_pres = sigmoid((pres_logit + gumbel[:, t]) / temp)

            patch, dec_cache = self.decode(z_what)
            canvas_t, stn_cache = render_forward(patch, where, self.canvas_size)

            cumpres_prev = cumpres.copy()
            eff_pres = cumpres_prev * z_pres
            recon = recon + eff_pres[:, None, None] * canvas_t
            cumpres = eff_pres  # for next step

            cumpres_history.append(cumpres_prev)
            eff_pres_history.append(eff_pres)
            z_pres_history.append(z_pres)

            step_caches.append(dict(t=t, pres_logit=pres_logit, where=where,
                                    what_mu=what_mu, what_lv=what_lv,
                                    z_what=z_what, z_pres=z_pres,
                                    eps_what=eps_what[:, t], gumbel=gumbel[:, t],
                                    cumpres_prev=cumpres_prev, eff_pres=eff_pres,
                                    canvas_t=canvas_t,
                                    dec_cache=dec_cache, stn_cache=stn_cache))

        cache = dict(x_img=x_img, x_flat=x_flat, enc_cache=enc_cache,
                     step_caches=step_caches, recon=recon,
                     cumpres_history=cumpres_history,
                     eff_pres_history=eff_pres_history,
                     z_pres_history=z_pres_history,
                     temp=temp, deterministic=deterministic)
        return recon, cache

    # -- backward ---------------------------------------------------------

    def backward(self, cache: dict, recon_weight: float = 1.0,
                 kl_what_weight: float = 1.0,
                 kl_pres_weight: float = 0.1):
        """Compute ELBO = recon_weight * MSE(recon, x) + KL_what + KL_pres.

        Returns (loss_dict, grads).
        """
        x_img = cache["x_img"]
        recon = cache["recon"]
        B, Hc, Wc = x_img.shape
        D_pix = Hc * Wc

        # Reconstruction loss: per-pixel BCE ish? Use Gaussian likelihood with
        # fixed variance for simplicity -- equivalent to MSE up to constants.
        diff = recon - x_img
        recon_loss = float(np.mean(diff ** 2)) * D_pix * recon_weight  # so per-image
        # gradient wrt recon: d/d_recon of recon_weight * (1/B) * sum (diff^2)
        # we compute scalar loss as mean over batch & pixels then scale by D_pix
        d_recon = (recon_weight * 2.0 / B) * diff   # (B, Hc, Wc); per-image MSE summed

        # KL on z_what (Gaussian to N(0, I)): collected across all steps
        kl_what_total = 0.0
        d_what_mu_per_step = []
        d_what_lv_per_step = []
        for sc in cache["step_caches"]:
            mu = sc["what_mu"]; lv = sc["what_lv"]
            # KL(N(mu, var) || N(0,I)) = -0.5 * sum (1 + lv - mu^2 - exp(lv))
            kl = -0.5 * np.sum(1.0 + lv - mu ** 2 - np.exp(lv), axis=1)  # (B,)
            kl_what_total += float(np.mean(kl)) * kl_what_weight
            # gradient wrt mu: kl_what_weight * mu / B
            d_what_mu_per_step.append((kl_what_weight / B) * mu)
            # gradient wrt lv: kl_what_weight * 0.5 * (exp(lv) - 1) / B
            d_what_lv_per_step.append((kl_what_weight * 0.5 / B) * (np.exp(lv) - 1.0))

        # KL on z_pres (Bernoulli relaxation -> use cross-entropy of sigmoid(logit) vs prior)
        # Treat z_pres as Bernoulli with prob q = sigmoid(logit), prior p = pres_prior_t.
        # KL(Bern(q) || Bern(p)) = q*log(q/p) + (1-q)*log((1-q)/(1-p))
        # We use a small weight to avoid swamping reconstruction.
        kl_pres_total = 0.0
        d_pres_logit_per_step = []
        for t, sc in enumerate(cache["step_caches"]):
            q = sigmoid(sc["pres_logit"])
            p = float(self.pres_priors[t])
            q_ = np.clip(q, 1e-6, 1 - 1e-6)
            kl = (q_ * (np.log(q_) - np.log(p)) +
                  (1 - q_) * (np.log(1 - q_) - np.log(1 - p)))
            kl_pres_total += float(np.mean(kl)) * kl_pres_weight
            # d_kl / d_logit = (q - p) * 1   (standard logistic CE form)
            d_pres_logit_per_step.append((kl_pres_weight / B) * (q - p))

        total_loss = recon_loss + kl_what_total + kl_pres_total

        # Initialize gradient accumulators
        grads = self.zero_like_params()
        # Gradient on h2 from all steps will be accumulated here:
        d_h2 = np.zeros_like(cache["enc_cache"]["h2"])

        T = self.max_steps
        # We need to backprop through cumpres = prod_{i<=t} z_pres_i for each t.
        # Strategy: store contribution from recon to each (eff_pres_t, canvas_t).
        # eff_pres_t gradient propagates back into z_pres_t and z_pres_{i<t}
        # via the product rule. We accumulate into d_z_pres_per_step then into
        # d_pres_logit (Gumbel sigmoid) and d_h2.

        # Pass 1: compute d_eff_pres_t and d_canvas_t from d_recon
        d_eff_pres = []
        d_canvas_t_list = []
        for t in range(T):
            sc = cache["step_caches"][t]
            canvas_t = sc["canvas_t"]
            eff_pres = sc["eff_pres"]
            # recon += eff_pres * canvas_t   (per pixel, broadcast eff_pres over (Hc, Wc))
            d_canvas_t = d_recon * eff_pres[:, None, None]
            d_eff = (d_recon * canvas_t).sum(axis=(1, 2))  # (B,)
            d_eff_pres.append(d_eff)
            d_canvas_t_list.append(d_canvas_t)

        # Pass 2: convert d_eff_pres to d_z_pres for each step via product rule
        # eff_pres_t = prod_{i<=t} z_pres_i
        # d eff_pres_t / d z_pres_j = (prod_{i<=t, i!=j} z_pres_i) for j <= t else 0
        # We accumulate into d_z_pres[j] for each j.
        z_pres_per_step = [sc["z_pres"] for sc in cache["step_caches"]]
        d_z_pres = [np.zeros_like(zp) for zp in z_pres_per_step]
        for t in range(T):
            d_eff = d_eff_pres[t]
            # contribution from eff_pres_t to z_pres_j for j <= t
            for j in range(t + 1):
                # partial = d_eff * prod_{i<=t, i!=j} z_pres_i
                # This equals d_eff * eff_pres_t / z_pres_j (with safe guard)
                eff = cache["step_caches"][t]["eff_pres"]
                z_j = z_pres_per_step[j]
                # safe division
                partial = d_eff * eff / np.clip(z_j, 1e-6, None)
                d_z_pres[j] += partial

        # Now backprop for each step
        for t in range(T):
            sc = cache["step_caches"][t]
            d_canvas_t = d_canvas_t_list[t]

            # STN backward: d_canvas_t -> d_patch, d_where
            d_patch, d_where = render_backward(d_canvas_t, sc["stn_cache"])

            # Decoder backward: d_patch -> d_z_what
            dec_cache = sc["dec_cache"]
            d_p_flat = d_patch.reshape(B, -1)
            # sigmoid' = p_flat * (1 - p_flat)
            d_d2 = d_p_flat * dec_cache["p_flat"] * (1.0 - dec_cache["p_flat"])
            grads["W_d2"] += dec_cache["h1"].T @ d_d2
            grads["b_d2"] += d_d2.sum(axis=0)
            d_h1 = d_d2 @ self.W_d2.T
            d_d1 = d_h1 * (dec_cache["d1"] > 0).astype(np.float32)
            grads["W_d1"] += dec_cache["z_what"].T @ d_d1
            grads["b_d1"] += d_d1.sum(axis=0)
            d_z_what = d_d1 @ self.W_d1.T

            # z_what = mu + exp(0.5 * lv) * eps  (or just mu if deterministic)
            if cache["deterministic"]:
                d_what_mu = d_z_what + d_what_mu_per_step[t]
                d_what_lv = d_what_lv_per_step[t]
            else:
                d_what_mu = d_z_what + d_what_mu_per_step[t]
                d_what_lv = (d_z_what * sc["eps_what"] *
                             0.5 * np.exp(0.5 * sc["what_lv"]) +
                             d_what_lv_per_step[t])

            # z_pres = sigmoid((logit + g) / temp)
            if cache["deterministic"]:
                z_pres = sc["z_pres"]
                d_pres_logit_from_recon = d_z_pres[t] * z_pres * (1.0 - z_pres)
            else:
                z_pres = sc["z_pres"]
                d_pres_logit_from_recon = (d_z_pres[t] * z_pres * (1.0 - z_pres)
                                           / cache["temp"])
            d_pres_logit = d_pres_logit_from_recon + d_pres_logit_per_step[t]

            # Heads backward (linear): d/d_W = h2.T @ d_out, d/d_b = sum d_out, d/d_h2 = d_out @ W.T
            # z_pres head
            grads["W_pres"][t] += cache["enc_cache"]["h2"].T @ d_pres_logit[:, None]
            grads["b_pres"][t] += d_pres_logit.sum(axis=0, keepdims=True)
            d_h2 += d_pres_logit[:, None] @ self.W_pres[t].T

            # z_where head
            grads["W_where"][t] += cache["enc_cache"]["h2"].T @ d_where
            grads["b_where"][t] += d_where.sum(axis=0)
            d_h2 += d_where @ self.W_where[t].T

            # z_what mu head
            grads["W_what_mu"][t] += cache["enc_cache"]["h2"].T @ d_what_mu
            grads["b_what_mu"][t] += d_what_mu.sum(axis=0)
            d_h2 += d_what_mu @ self.W_what_mu[t].T

            # z_what logvar head (clipped: zero out grad outside the clip range)
            mask_lv = ((sc["what_lv"] > -7.99) & (sc["what_lv"] < 3.99)).astype(np.float32)
            d_what_lv_eff = d_what_lv * mask_lv
            grads["W_what_lv"][t] += cache["enc_cache"]["h2"].T @ d_what_lv_eff
            grads["b_what_lv"][t] += d_what_lv_eff.sum(axis=0)
            d_h2 += d_what_lv_eff @ self.W_what_lv[t].T

        # Encoder backward
        ec = cache["enc_cache"]
        d_a2 = d_h2 * (ec["a2"] > 0).astype(np.float32)
        grads["W_e2"] += ec["h1"].T @ d_a2
        grads["b_e2"] += d_a2.sum(axis=0)
        d_h1 = d_a2 @ self.W_e2.T
        d_a1 = d_h1 * (ec["a1"] > 0).astype(np.float32)
        grads["W_e1"] += ec["x"].T @ d_a1
        grads["b_e1"] += d_a1.sum(axis=0)

        loss_dict = dict(total=total_loss,
                         recon=recon_loss,
                         kl_what=kl_what_total,
                         kl_pres=kl_pres_total)
        return loss_dict, grads

    # -- inference --------------------------------------------------------

    def parse_scene(self, image: np.ndarray, threshold: float = 0.5):
        """Parse a single scene. Returns dict with z_pres, z_where, z_what,
        predicted reconstruction, and per-step patches.

        `image`: (Hc, Wc) or (B, Hc, Wc).
        """
        single = (image.ndim == 2)
        x = image[None] if single else image
        recon, cache = self.forward(x, deterministic=True)
        per_step = []
        for sc in cache["step_caches"]:
            per_step.append(dict(
                z_pres=sc["z_pres"],
                z_where=sc["where"],
                z_what=sc["what_mu"],
                patch=sc["dec_cache"]["p_flat"].reshape(-1, self.patch_size, self.patch_size),
                canvas_t=sc["canvas_t"],
            ))
        z_pres_stack = np.stack([sc["z_pres"] for sc in cache["step_caches"]], axis=1)  # (B, T)
        # cumulative-pres masking: count = sum_t (cumprod_{<=t} > thr)
        cumprod = np.cumprod(z_pres_stack, axis=1)
        active_mask = (cumprod > threshold).astype(np.int64)
        count = active_mask.sum(axis=1)
        result = dict(
            recon=recon[0] if single else recon,
            z_pres=z_pres_stack[0] if single else z_pres_stack,
            cum_pres=cumprod[0] if single else cumprod,
            count=int(count[0]) if single else count,
            per_step=per_step,
        )
        return result


# ----------------------------------------------------------------------
# Public-facing helpers (matches problem.py signature)
# ----------------------------------------------------------------------

def build_air_model(canvas_size: int = 32, max_steps: int = 3,
                    what_dim: int = 20, where_dim: int = 3,
                    seed: int = 0):
    if where_dim != 3:
        raise ValueError("where_dim must be 3 (log_s, tx, ty).")
    return AIR(canvas_size=canvas_size, max_steps=max_steps,
               what_dim=what_dim, seed=seed)


def elbo_loss(model: AIR, batch_images: np.ndarray, **kwargs):
    """Compute ELBO loss and gradients on a batch of (B, Hc, Wc) scenes."""
    recon, cache = model.forward(batch_images)
    return model.backward(cache, **kwargs)


def parse_scene(model: AIR, image: np.ndarray) -> dict:
    return model.parse_scene(image)


# ----------------------------------------------------------------------
# Adam optimizer
# ----------------------------------------------------------------------

class Adam:
    def __init__(self, model: AIR, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = model.zero_like_params()
        self.v = model.zero_like_params()

    def step(self, grads: dict, lr: float | None = None):
        lr = self.lr if lr is None else lr
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for k, g in grads.items():
            m = self.m[k]; v = self.v[k]
            m[...] = self.beta1 * m + (1.0 - self.beta1) * g
            v[...] = self.beta2 * v + (1.0 - self.beta2) * (g * g)
            update = lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)
            getattr(self.model, k)[...] -= update


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def _count_accuracy(model: AIR, scenes: np.ndarray, counts: np.ndarray,
                    threshold: float = 0.5):
    out = model.parse_scene(scenes, threshold=threshold)
    return float(np.mean(out["count"] == counts))


def train(model: AIR | None = None,
          n_train: int = 1000,
          n_val: int = 200,
          canvas_size: int = 32,
          max_steps: int = 3,
          what_dim: int = 20,
          n_epochs: int = 8,
          batch_size: int = 32,
          lr: float = 2e-3,
          temp_init: float = 1.0,
          temp_final: float = 0.2,
          recon_weight: float = 1.0,
          kl_what_weight: float = 0.05,
          kl_pres_weight: float = 0.3,
          warmup_kl_epochs: int = 2,
          seed: int = 0,
          verbose: bool = True,
          snapshot_callback=None,
          snapshot_every: int = 100):
    """Train AIR on Multi-MNIST scenes.

    Returns (model, history, train_data, val_data).
    """
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"# loading MNIST...", flush=True)
    images, labels = load_mnist("train")
    if verbose:
        print(f"# generating {n_train} train + {n_val} val scenes "
              f"({canvas_size}x{canvas_size}, 0-2 digits)...", flush=True)
    train_scenes, train_counts, train_placements = generate_scenes(
        n_train, images, labels, canvas_size, rng=rng)
    val_scenes, val_counts, val_placements = generate_scenes(
        n_val, images, labels, canvas_size,
        rng=np.random.default_rng(seed + 9999))

    if model is None:
        model = AIR(canvas_size=canvas_size, max_steps=max_steps,
                    what_dim=what_dim, seed=seed)
    opt = Adam(model, lr=lr)

    if verbose:
        print(f"# AIR: max_steps={max_steps}, what_dim={what_dim}, "
              f"patch_size={model.patch_size}", flush=True)
        print(f"# training: {n_epochs} epochs, batch_size={batch_size}, "
              f"lr={lr}, temp {temp_init}->{temp_final}", flush=True)

    history = {"step": [], "epoch": [],
               "loss": [], "recon": [], "kl_what": [], "kl_pres": [],
               "val_recon": [], "val_count_acc": [], "val_count_mae": []}

    best_count_acc = -1.0
    best_state = None
    best_epoch = 0

    n_steps_total = n_epochs * max(1, n_train // batch_size)
    step = 0
    t0 = time.time()
    for epoch in range(n_epochs):
        # KL warmup
        if epoch < warmup_kl_epochs:
            kw = kl_what_weight * (epoch + 1) / warmup_kl_epochs
            kp = kl_pres_weight * (epoch + 1) / warmup_kl_epochs
        else:
            kw = kl_what_weight
            kp = kl_pres_weight

        idx_shuffle = rng.permutation(n_train)
        epoch_recon = 0.0
        epoch_klw = 0.0
        epoch_klp = 0.0
        epoch_total = 0.0
        n_batches = n_train // batch_size

        for b in range(n_batches):
            sel = idx_shuffle[b * batch_size:(b + 1) * batch_size]
            batch = train_scenes[sel]

            # anneal temperature
            frac = step / max(1, n_steps_total - 1)
            temp = float(temp_init + (temp_final - temp_init) * frac)

            recon, cache = model.forward(batch, temp=temp)
            losses, grads = model.backward(cache, recon_weight=recon_weight,
                                           kl_what_weight=kw,
                                           kl_pres_weight=kp)
            opt.step(grads)

            epoch_recon += losses["recon"]
            epoch_klw   += losses["kl_what"]
            epoch_klp   += losses["kl_pres"]
            epoch_total += losses["total"]
            step += 1

            if snapshot_callback is not None and (step % snapshot_every == 0):
                snapshot_callback(step, model, history,
                                  train_scenes[:8], val_scenes, val_counts)

        # Epoch summary + validation
        avg_recon = epoch_recon / n_batches
        avg_klw   = epoch_klw   / n_batches
        avg_klp   = epoch_klp   / n_batches
        avg_total = epoch_total / n_batches

        val_out = model.parse_scene(val_scenes, threshold=0.5)
        val_count_acc = float(np.mean(val_out["count"] == val_counts))
        val_count_mae = float(np.mean(np.abs(val_out["count"] - val_counts)))
        # validation reconstruction MSE (deterministic)
        val_recon, _ = model.forward(val_scenes, deterministic=True)
        val_recon_mse = float(np.mean((val_recon - val_scenes) ** 2))

        history["step"].append(step)
        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_total)
        history["recon"].append(avg_recon)
        history["kl_what"].append(avg_klw)
        history["kl_pres"].append(avg_klp)
        history["val_recon"].append(val_recon_mse)
        history["val_count_acc"].append(val_count_acc)
        history["val_count_mae"].append(val_count_mae)

        if val_count_acc > best_count_acc:
            best_count_acc = val_count_acc
            best_state = {k: getattr(model, k).copy() for k in model.param_names}
            best_epoch = epoch + 1

        if verbose:
            elapsed = time.time() - t0
            print(f"epoch {epoch+1:2d}/{n_epochs}  "
                  f"recon={avg_recon:.3f}  "
                  f"kl_what={avg_klw:.3f}  kl_pres={avg_klp:.3f}  "
                  f"val_mse={val_recon_mse:.4f}  "
                  f"val_count_acc={val_count_acc:.3f}  "
                  f"val_count_mae={val_count_mae:.3f}  "
                  f"({elapsed:.1f}s)", flush=True)

    # Restore best params for the returned model.
    if best_state is not None:
        if verbose:
            print(f"# restoring best epoch {best_epoch} (val_count_acc={best_count_acc:.3f})",
                  flush=True)
        for k, v in best_state.items():
            getattr(model, k)[...] = v

    train_data = dict(scenes=train_scenes, counts=train_counts,
                      placements=train_placements)
    val_data   = dict(scenes=val_scenes,   counts=val_counts,
                      placements=val_placements)
    history["best_epoch"] = best_epoch
    history["best_count_acc"] = best_count_acc
    return model, history, train_data, val_data


# ----------------------------------------------------------------------
# Environment / reproducibility
# ----------------------------------------------------------------------

def env_info() -> dict:
    return dict(
        python=sys.version.split()[0],
        numpy=np.__version__,
        platform=platform.platform(),
        processor=platform.processor() or "unknown",
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--canvas-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=3)
    p.add_argument("--what-dim", type=int, default=20)
    p.add_argument("--n-epochs", type=int, default=8)
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-val",   type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--kl-what-weight", type=float, default=0.05)
    p.add_argument("--kl-pres-weight", type=float, default=0.3)
    p.add_argument("--out-dir", type=str, default=str(Path(__file__).parent / "viz"))
    args = p.parse_args()

    t0 = time.time()
    model, history, train_data, val_data = train(
        n_train=args.n_train, n_val=args.n_val,
        canvas_size=args.canvas_size, max_steps=args.max_steps,
        what_dim=args.what_dim, n_epochs=args.n_epochs,
        batch_size=args.batch_size, lr=args.lr,
        kl_what_weight=args.kl_what_weight,
        kl_pres_weight=args.kl_pres_weight,
        seed=args.seed)
    elapsed = time.time() - t0

    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best epoch: {history['best_epoch']} "
          f"(val_count_acc={history['best_count_acc']:.3f})")
    print(f"Best count accuracy (model restored): {history['best_count_acc']:.3f} "
          f"(chance = {1.0/3:.3f})")
    print(f"Final-epoch val MSE: {history['val_recon'][-1]:.4f}")
    print(f"Env: {env_info()}")


if __name__ == "__main__":
    main()
