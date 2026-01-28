#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import random
import math
import logging
import pickle
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pytorch_lightning.utilities.seed import seed_everything
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semi_synthetic_dataset import (
    MIMIC3SyntheticDatasetCollection,
    MIMIC3SyntheticTimeSeriesDataset,
)
from evaluation import (
    compute_n_step_rmse_nll,
    _prep_out_scaling,
    unscale_gaussian,
    unscale_y,
    masked_rmse_per_tau_avg_dims,
    masked_gaussian_nll_per_tau_avg_dims,
)
from vis_func import vis_kalman_fusion, vis_patients

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# Basics
# =============================================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def max_step_in_taus_str(taus_str: str) -> int:
    """Returns the maximum 1-based step referenced in taus_str (or 0 if none/all)."""
    if taus_str is None:
        return 0
    s = str(taus_str).strip().lower()
    if s in ("", "all"):
        return 0
    m = 0
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            a = int(a.strip()); b = int(b.strip())
            m = max(m, a, b)
        else:
            m = max(m, int(chunk))
    return int(m)

def parse_taus_arg(taus_str: str, horizon: int):
    """
    Parse eval steps as 1-based, return 0-based tau indices.
    Example: "1,2,5-7" -> [0,1,4,5,6]
    """
    if taus_str is None:
        return list(range(horizon))
    s = str(taus_str).strip().lower()
    if s in ("", "all"):
        return list(range(horizon))

    items = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            a = int(a.strip()); b = int(b.strip())
            lo, hi = min(a, b), max(a, b)
            items.extend(list(range(lo, hi + 1)))
        else:
            items.append(int(chunk))

    taus = []
    for step in items:
        tau = step - 1
        if 0 <= tau < horizon:
            taus.append(tau)
    taus = sorted(set(taus))
    return taus if taus else list(range(horizon))

def get_future_past_split(subset, horizon: int) -> np.ndarray:
    d = subset.data
    if "future_past_split" in d:
        return np.asarray(d["future_past_split"]).astype(int)
    L = np.asarray(d["sequence_lengths"]).astype(int)
    t0 = np.maximum(L - int(horizon), 1)
    return t0.astype(int)

def _onehot_from_code(code: int, A_dim: int) -> np.ndarray:
    v = np.zeros((A_dim,), dtype=np.float32)
    if 0 <= int(code) < A_dim:
        v[int(code)] = 1.0
    return v

def _safe_impute_nan_with_mean(X: np.ndarray) -> np.ndarray:
    """
    Replace non-finite entries with column means (or 0 if a column mean is NaN).
    Works for 2D arrays.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"_safe_impute_nan_with_mean expects 2D array, got {X.shape}")
    col_mean = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    X2 = X.copy()
    bad = ~np.isfinite(X2)
    if bad.any():
        X2[bad] = col_mean[np.where(bad)[1]]
    return X2

# =============================================================================
# Transformer prior (Gaussian head)
# =============================================================================

def build_causal_mask(T: int, device: torch.device):
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

def mask_to_padding(mask_bt: torch.Tensor) -> torch.Tensor:
    if mask_bt.dim() == 3:
        mask_bt = mask_bt[..., 0]
    mask_bt = mask_bt.float()
    B, T = mask_bt.shape
    lengths = mask_bt.sum(dim=1).long().clamp(min=0, max=T)
    ar = torch.arange(T, device=mask_bt.device).unsqueeze(0).expand(B, T)
    pad = ar >= lengths.unsqueeze(1)
    return pad

class GaussianCausalTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dim_outcomes: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 256,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.dim_outcomes = int(dim_outcomes)
        self.d_model = int(d_model)
        self.max_len = int(max_len)

        self.in_proj = nn.Linear(self.input_dim, self.d_model)
        self.pos_emb = nn.Embedding(self.max_len, self.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.mean_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.dim_outcomes),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.dim_outcomes),
        )

    def encode(self, x, src_key_padding_mask=None):
        """
        Returns encoder hidden states h: (B,T,d_model)
        """
        B, T, _ = x.shape
        device = x.device
        if T > self.max_len:
            raise ValueError(f"T={T} exceeds max_len={self.max_len}. Increase max_len.")
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        h = self.in_proj(x) + self.pos_emb(pos)
        causal_mask = build_causal_mask(T, device=device)
        h = self.encoder(h, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        return h

    def forward(self, x, src_key_padding_mask=None, return_hidden: bool = False):
        h = self.encode(x, src_key_padding_mask=src_key_padding_mask)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        if return_hidden:
            return mean, logvar, h
        return mean, logvar

# =============================================================================
# Token layout inference (Transformer only)
# =============================================================================

def build_token_from_parts(y_prev, a_onehot, x_static, vitals_t, order, D_vitals: int):
    parts = []
    for name in order:
        if name == "vitals":
            if D_vitals <= 0:
                continue
            if vitals_t is None:
                parts.append(np.zeros((D_vitals,), dtype=np.float32))
            else:
                parts.append(np.asarray(vitals_t, dtype=np.float32).reshape(-1))
        elif name == "prev_outputs":
            parts.append(np.asarray(y_prev, dtype=np.float32).reshape(-1))
        elif name == "treatments":
            parts.append(np.asarray(a_onehot, dtype=np.float32).reshape(-1))
        elif name == "static":
            parts.append(np.asarray(x_static, dtype=np.float32).reshape(-1))
        else:
            raise ValueError(f"Unknown part name: {name}")
    return np.concatenate(parts, axis=0).astype(np.float32)

def infer_layout_by_matching(subset):
    ds = MIMIC3SyntheticTimeSeriesDataset(subset)
    x_seq, y_seq, mask = ds[0]
    x_seq = np.asarray(x_seq)
    y_seq = np.asarray(y_seq)

    d = subset.data
    Dy = int(np.asarray(d["outputs"]).shape[-1])
    D_vitals = int(np.asarray(d["vitals"]).shape[-1]) if "vitals" in d else 0

    candidates = [
        ("prev_outputs+treat+static+vitals", ("prev_outputs", "treatments", "static", "vitals")),
        ("vitals+prev_outputs+treat+static", ("vitals", "prev_outputs", "treatments", "static")),
    ]

    best = None
    best_err = float("inf")
    T = x_seq.shape[0]
    max_t = min(T, 8)

    for name, order in candidates:
        errs = []
        y_prev = np.zeros((Dy,), dtype=np.float32)
        for t in range(max_t):
            if float(mask[t]) <= 0.5:
                break
            a_onehot = np.asarray(d["current_treatments"][0, t], dtype=np.float32)
            x_static = np.asarray(d["static_features"][0], dtype=np.float32)
            vitals_t = np.asarray(d["vitals"][0, t], dtype=np.float32) if D_vitals > 0 else None
            tok = build_token_from_parts(y_prev, a_onehot, x_static, vitals_t, order, D_vitals)
            xt = x_seq[t].astype(np.float32)
            errs.append(1e9 if tok.shape[-1] != xt.shape[-1] else float(np.mean((tok - xt) ** 2)))
            y_prev = y_seq[t].astype(np.float32).reshape(-1)
        err = float(np.mean(errs)) if errs else 1e9
        print(f"[layout] candidate={name:28s} mse={err:.6e}")
        if err < best_err:
            best_err = err
            best = {"name": name, "order": order, "D_vitals": D_vitals}

    if best is None:
        raise RuntimeError("Could not infer token layout.")
    print(f"[layout] selected: {best['name']} (mse={best_err:.6e})")
    return best

# =============================================================================
# MMD utilities (simple RBF MMD, multi-class one-vs-rest)
# =============================================================================

def compute_token_slices(layout: dict, Dy: int, A_dim: int, D_static: int, D_vitals: int, input_dim: int):
    """
    Build (start,end) slices for each token part, consistent with build_token_from_parts().
    Returns dict: {"prev_outputs": (s,e), "treatments": (s,e), "static": (s,e), "vitals": (s,e)}
    """
    order = layout["order"]
    idx = 0
    slices = {}
    for name in order:
        if name == "vitals":
            size = int(D_vitals)
        elif name == "prev_outputs":
            size = int(Dy)
        elif name == "treatments":
            size = int(A_dim)
        elif name == "static":
            size = int(D_static)
        else:
            raise ValueError(f"Unknown part name in layout order: {name}")
        slices[name] = (idx, idx + size)
        idx += size

    if idx != int(input_dim):
        raise ValueError(
            f"Token slice sum mismatch: computed={idx} vs input_dim={input_dim}. "
            f"Check layout/order and dims (Dy={Dy}, A_dim={A_dim}, D_static={D_static}, D_vitals={D_vitals})."
        )
    return slices

def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).T
    return torch.clamp(x2 + y2 - 2.0 * (x @ y.T), min=0.0)

def _rbf_kernel_from_d2(d2: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return torch.exp(-d2 / (2.0 * (h * h) + 1e-12))

def _median_heuristic_bandwidth(z: torch.Tensor, max_pairs: int = 4096, eps: float = 1e-6) -> torch.Tensor:
    N = z.shape[0]
    if N <= 1:
        return torch.tensor(1.0, device=z.device, dtype=z.dtype)

    rng = torch.randperm(N, device=z.device)
    m = min(N, int(math.sqrt(max_pairs)) + 1)
    idx = rng[:m]
    zz = z[idx]

    d2 = _pairwise_sq_dists(zz, zz)
    triu = torch.triu_indices(m, m, offset=1, device=z.device)
    vals = d2[triu[0], triu[1]]
    if vals.numel() == 0:
        return torch.tensor(1.0, device=z.device, dtype=z.dtype)

    med = torch.median(vals)
    h = torch.sqrt(torch.clamp(med, min=0.0) + eps)
    return torch.clamp(h, min=eps)

def mmd2_unbiased_rbf(x: torch.Tensor, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    m = y.shape[0]
    if n < 2 or m < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)

    d2_xx = _pairwise_sq_dists(x, x)
    d2_yy = _pairwise_sq_dists(y, y)
    d2_xy = _pairwise_sq_dists(x, y)

    K_xx = _rbf_kernel_from_d2(d2_xx, h)
    K_yy = _rbf_kernel_from_d2(d2_yy, h)
    K_xy = _rbf_kernel_from_d2(d2_xy, h)

    sum_xx = (K_xx.sum() - K_xx.diag().sum()) / (n * (n - 1) + 1e-12)
    sum_yy = (K_yy.sum() - K_yy.diag().sum()) / (m * (m - 1) + 1e-12)
    sum_xy = K_xy.mean()

    return sum_xx + sum_yy - 2.0 * sum_xy

def mmd_multiclass_one_vs_rest(
    feats: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    min_per_group: int = 8,
    max_per_group: int = 256,
    bw_mode: str = "median",
    bw_fixed: float = None,
    bw_scale: float = 1.0,
) -> torch.Tensor:
    device = feats.device
    dtype = feats.dtype

    losses = []
    for a in range(int(num_classes)):
        idx_a = (labels == a).nonzero(as_tuple=False).view(-1)
        idx_r = (labels != a).nonzero(as_tuple=False).view(-1)

        if idx_a.numel() < int(min_per_group) or idx_r.numel() < int(min_per_group):
            continue

        if idx_a.numel() > int(max_per_group):
            idx_a = idx_a[torch.randperm(idx_a.numel(), device=device)[:int(max_per_group)]]
        if idx_r.numel() > int(max_per_group):
            idx_r = idx_r[torch.randperm(idx_r.numel(), device=device)[:int(max_per_group)]]

        xa = feats.index_select(0, idx_a)
        xr = feats.index_select(0, idx_r)

        if str(bw_mode) == "fixed":
            if bw_fixed is None:
                raise ValueError("mmd_bw must be set when mmd_bw_mode='fixed'")
            h = torch.tensor(float(bw_fixed), device=device, dtype=dtype)
        elif str(bw_mode) == "median":
            z = torch.cat([xa, xr], dim=0)
            h = _median_heuristic_bandwidth(z) * float(bw_scale)
        else:
            raise ValueError("mmd_bw_mode must be 'median' or 'fixed'")

        h = torch.clamp(h, min=1e-6)
        losses.append(mmd2_unbiased_rbf(xa, xr, h))

    if not losses:
        return torch.zeros((), device=device, dtype=dtype)
    return torch.stack(losses).mean()

# =============================================================================
# Transformer training with MMD penalty
# =============================================================================

def train_gaussian_prior(
    model: nn.Module,
    train_ds,
    num_epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    logvar_clip=(-10.0, 8.0),
    lambda_mmd: float = 0.0,
    treat_slice: tuple = None,
    A_dim: int = None,
    mmd_min_per_group: int = 8,
    mmd_max_per_group: int = 256,
    mmd_bw_mode: str = "median",
    mmd_bw: float = None,
    mmd_bw_scale: float = 1.0,
):
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    use_mmd = (
        float(lambda_mmd) > 0.0
        and treat_slice is not None
        and A_dim is not None
        and int(A_dim) > 1
    )

    for epoch in range(1, num_epochs + 1):
        model.train()
        loss_sum = 0.0
        nll_sum = 0.0
        mmd_sum = 0.0
        nb = 0

        for x, y, mask in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            pad_mask = mask_to_padding(mask)
            mean, logvar, _h = model(x, src_key_padding_mask=pad_mask, return_hidden=True)

            logvar = torch.clamp(logvar, min=logvar_clip[0], max=logvar_clip[1])
            var = torch.exp(logvar) + 1e-6

            m = mask
            if m.dim() == 3:
                m = m[..., 0]
            m = m.float().unsqueeze(-1)

            nll = 0.5 * (((y - mean) ** 2) / var + logvar)
            nll = nll * m
            denom = (m.sum() * y.shape[-1]) + 1e-8
            loss_nll = nll.sum() / denom

            loss_mmd = torch.zeros((), device=device, dtype=loss_nll.dtype)
            if use_mmd:
                s, e = int(treat_slice[0]), int(treat_slice[1])
                if e > s:
                    x_cov = x.detach().clone()
                    x_cov[:, :, s:e] = 0.0
                    h_cov = model.encode(x_cov, src_key_padding_mask=pad_mask)

                    a_onehot = x[:, :, s:e]
                    a_lab = torch.argmax(a_onehot, dim=-1)

                    valid = (m.squeeze(-1) > 0.0)
                    feats = h_cov[valid]
                    labs = a_lab[valid]

                    if feats.shape[0] >= 2:
                        loss_mmd = mmd_multiclass_one_vs_rest(
                            feats=feats,
                            labels=labs,
                            num_classes=int(A_dim),
                            min_per_group=int(mmd_min_per_group),
                            max_per_group=int(mmd_max_per_group),
                            bw_mode=str(mmd_bw_mode),
                            bw_fixed=mmd_bw,
                            bw_scale=float(mmd_bw_scale),
                        )

            loss = loss_nll + float(lambda_mmd) * loss_mmd

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            loss_sum += float(loss.item())
            nll_sum += float(loss_nll.item())
            if use_mmd:
                mmd_sum += float(loss_mmd.item())
            nb += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == num_epochs:
            if use_mmd:
                print(
                    f"[prior] epoch {epoch:03d}/{num_epochs} | loss={loss_sum/max(nb,1):.4f} | "
                    f"NLL={nll_sum/max(nb,1):.4f} | MMD={mmd_sum/max(nb,1):.6f} | lambda_mmd={float(lambda_mmd):.3g}"
                )
            else:
                print(f"[prior] epoch {epoch:03d}/{num_epochs} | train NLL={loss_sum / max(nb,1):.4f}")

# =============================================================================
# PCA (no sklearn)
# =============================================================================

class PCAProjector:
    def __init__(self, n_components: int = 16, whiten: bool = True, eps: float = 1e-8):
        self.n_components = int(n_components)
        self.whiten = bool(whiten)
        self.eps = float(eps)
        self.fitted = False

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"PCAProjector.fit expects 2D array, got shape {X.shape}")
        X = _safe_impute_nan_with_mean(X)

        N, D = X.shape
        if N < 2:
            raise ValueError("PCAProjector.fit needs at least 2 samples.")
        K = int(min(self.n_components, D, max(1, N - 1)))

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        V = Vt[:K, :]

        eigvals = (S**2) / max(N - 1, 1)
        eigvals_k = eigvals[:K]

        self.components_ = V
        self.eigvals_ = eigvals_k
        self.K_ = K
        self.fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("PCAProjector not fitted.")
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.mean_.size:
            raise ValueError(f"PCAProjector.transform dim mismatch: got {x.size}, expected {self.mean_.size}")
        x = np.where(np.isfinite(x), x, self.mean_)
        z = (x - self.mean_) @ self.components_.T
        if self.whiten:
            z = z / np.sqrt(self.eigvals_ + self.eps)
        return z.astype(float)

# =============================================================================
# Common SC utilities
# =============================================================================

def weighted_mean_var(u: np.ndarray, w: np.ndarray, min_var: float = 1e-8):
    mu = float(np.sum(w * u))
    var = float(np.sum(w * (u - mu) ** 2))
    return mu, max(var, min_var)

def effective_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float).reshape(-1)
    s2 = float(np.sum(w * w)) + 1e-12
    return float(1.0 / s2)

def kernel_weights_from_d2(
    d2: np.ndarray,
    kernel: str = "rbf",
    bandwidth_mode: str = "median",
    bandwidth: float = None,
    bandwidth_scale: float = 1.0,
    min_bandwidth: float = 1e-6,
):
    d2 = np.asarray(d2, dtype=float).reshape(-1)
    J = d2.size
    if J == 0:
        return np.zeros((0,), dtype=float), float("nan")

    if bandwidth_mode == "median":
        med = float(np.median(d2))
        h = np.sqrt(max(med, 0.0) + 1e-12) * float(bandwidth_scale)
    elif bandwidth_mode == "fixed":
        if bandwidth is None:
            raise ValueError("bandwidth must be set when bandwidth_mode='fixed'")
        h = float(bandwidth)
    else:
        raise ValueError("bandwidth_mode must be 'median' or 'fixed'")
    h = max(h, float(min_bandwidth))

    if kernel == "rbf":
        s = -d2 / (2.0 * h * h)
    elif kernel == "laplace":
        s = -np.sqrt(np.maximum(d2, 0.0)) / h
    else:
        raise ValueError("kernel must be 'rbf' or 'laplace'")

    s = s - float(np.max(s))
    w = np.exp(s)
    sw = float(np.sum(w))
    if not np.isfinite(sw) or sw <= 0:
        return (np.ones((J,), dtype=float) / max(J, 1)), h
    return (w / sw).astype(float), h

# =============================================================================
# SC Feature builder: HistDeltaPlus_1D base (shared by kernel/attention)
# =============================================================================

class HistDeltaPlus_Features_1D:
    """
    Base for SC per outcome dim.
    """

    def __init__(
        self,
        outcome_idx: int,
        m_hist: int = 10,
        m_dhist: int = 6,
        target_mode: str = "delta",
        eps_rel: float = 1e-3,
        max_donors_per_action: int = 20000,
        knn: int = 200,
        max_steps_total: int = 300000,
        seed: int = 0,
        use_prev_treat: bool = True,
        clip_q: float = 0.01,
        tau_dist: float = 0.05,
        static_pca_dim: int = 16,
        vitals_pca_dim: int = 8,
        lambda_vitals: float = 1.0,
        # --- (1) stabilization: z-std floor ---
        zstd_floor: float = 5e-2,
        sc_debug: bool = False,
    ):
        self.outcome_idx = int(outcome_idx)
        self.m_hist = int(m_hist)
        self.m_dhist = int(m_dhist)
        assert target_mode in ("delta", "rel_delta", "level")
        self.target_mode = target_mode
        self.eps_rel = float(eps_rel)

        self.max_donors_per_action = int(max_donors_per_action)
        self.knn = int(knn)
        self.max_steps_total = int(max_steps_total)
        self.seed = int(seed)
        self.use_prev_treat = bool(use_prev_treat)
        self.clip_q = clip_q if clip_q is None else float(clip_q)

        self.tau_dist = float(tau_dist)

        self.static_pca_dim = int(static_pca_dim)
        self.vitals_pca_dim = int(vitals_pca_dim)
        self.lambda_vitals = float(lambda_vitals)

        self.zstd_floor = float(zstd_floor)
        self.sc_debug = bool(sc_debug)

        self.static_pca = None
        self.vitals_pca = None
        self.fitted = False

    def set_static_pca(self, pca: PCAProjector):
        self.static_pca = pca

    def set_vitals_pca(self, pca: PCAProjector):
        self.vitals_pca = pca

    def _pt_feat(self, prev_code: int, A_dim: int):
        if not self.use_prev_treat:
            return np.zeros((A_dim,), dtype=float)
        v = np.zeros((A_dim,), dtype=np.float32)
        if 0 <= int(prev_code) < A_dim:
            v[int(prev_code)] = 1.0
        return v.astype(float)

    def _static_feat(self, x_static: np.ndarray):
        if self.static_pca_dim == 0:
            # Use raw static features when pca_dim=0
            if x_static is None:
                return np.zeros((0,), dtype=float)
            return np.asarray(x_static, dtype=float).reshape(-1)
        if self.static_pca is None or x_static is None:
            return np.zeros((self.static_pca_dim,), dtype=float)
        z = self.static_pca.transform(x_static)
        if z.size < self.static_pca_dim:
            z = np.pad(z, (0, self.static_pca_dim - z.size), mode="constant", constant_values=0.0)
        elif z.size > self.static_pca_dim:
            z = z[:self.static_pca_dim]
        return z.astype(float)

    def _vitals_feat(self, vit_prev: np.ndarray):
        if self.vitals_pca_dim == 0:
            # Use raw vitals features when pca_dim=0
            if vit_prev is None:
                return np.zeros((0,), dtype=float)
            return (self.lambda_vitals * np.asarray(vit_prev, dtype=float).reshape(-1))
        if self.vitals_pca is None or vit_prev is None:
            return np.zeros((self.vitals_pca_dim,), dtype=float)
        z = self.vitals_pca.transform(vit_prev)
        if z.size < self.vitals_pca_dim:
            z = np.pad(z, (0, self.vitals_pca_dim - z.size), mode="constant", constant_values=0.0)
        elif z.size > self.vitals_pca_dim:
            z = z[:self.vitals_pca_dim]
        return (self.lambda_vitals * z).astype(float)

    def _level_hist(self, y, t, m):
        start = max(0, t - m)
        h = y[start:t][::-1]
        if len(h) == 0:
            return np.zeros((m,), dtype=float)
        if len(h) < m:
            h = np.pad(h, (0, m - len(h)), mode="constant", constant_values=h[0])
        return h.astype(float)

    def _delta_hist(self, y, t, m):
        if m <= 0:
            return np.zeros((0,), dtype=float)
        deltas = []
        for k in range(t - 1, max(0, t - 1 - m), -1):
            if k <= 0:
                break
            deltas.append(float(y[k] - y[k - 1]))
        if len(deltas) == 0:
            return np.zeros((m,), dtype=float)
        if len(deltas) < m:
            deltas = deltas + [deltas[0]] * (m - len(deltas))
        return np.asarray(deltas[:m], dtype=float)

    def _build_z(self, level_hist, delta_hist, pt_feat, static_feat, vitals_feat):
        return np.concatenate(
            [np.asarray(level_hist, float).reshape(-1),
             np.asarray(delta_hist, float).reshape(-1),
             np.asarray(pt_feat, float).reshape(-1),
             np.asarray(static_feat, float).reshape(-1),
             np.asarray(vitals_feat, float).reshape(-1)],
            axis=0
        )

    def _make_target(self, y_prev, y_cur):
        if self.target_mode == "level":
            return float(y_cur)
        delta = float(y_cur - y_prev)
        if self.target_mode == "delta":
            return delta
        denom = max(abs(float(y_prev)), self.eps_rel)
        return delta / denom

    def _build_z_target(self, y_hist_buffer, prev_code: int, x_static: np.ndarray, vit_prev: np.ndarray):
        lvl = np.asarray(y_hist_buffer[: self.m_hist], dtype=float)
        if lvl.size < self.m_hist:
            fill = float(lvl[0]) if lvl.size else 0.0
            lvl = np.pad(lvl, (0, self.m_hist - lvl.size), mode="constant", constant_values=fill)

        dh = []
        for k in range(min(self.m_dhist, len(y_hist_buffer) - 1)):
            dh.append(float(y_hist_buffer[k] - y_hist_buffer[k + 1]))
        if self.m_dhist > 0:
            if len(dh) == 0:
                dh = [0.0] * self.m_dhist
            elif len(dh) < self.m_dhist:
                dh = dh + [dh[0]] * (self.m_dhist - len(dh))
        dh = np.asarray(dh[: self.m_dhist], dtype=float)

        pt = self._pt_feat(prev_code, self.A_dim)
        s = self._static_feat(x_static)
        v = self._vitals_feat(vit_prev)

        z = self._build_z(lvl, dh, pt, s, v)
        z = (z - self.z_mean) / self.z_std
        return z.astype(np.float32)

    def fit(self, train_subset):
        d = train_subset.data
        Y = np.asarray(d["outputs"]).astype(float)
        A_onehot = np.asarray(d["current_treatments"]).astype(float)
        X_static = np.asarray(d["static_features"]).astype(float) if "static_features" in d else None
        V = np.asarray(d["vitals"]).astype(float) if "vitals" in d else None
        seq_lens = np.asarray(d["sequence_lengths"]).astype(int)

        A_codes = A_onehot.argmax(axis=-1).astype(int)
        A_dim = int(A_onehot.shape[-1])
        rng = np.random.default_rng(self.seed)

        Z_list, tgt_list, a_list = [], [], []
        N, T, Dy = Y.shape
        for i in range(N):
            L = int(seq_lens[i])
            if L <= 1:
                continue
            y_i = Y[i, :L, self.outcome_idx].astype(float)

            x_i = X_static[i] if X_static is not None else None
            s_i = self._static_feat(x_i)

            for t in range(1, L):
                a_t = int(A_codes[i, t])
                prev_code = int(A_codes[i, t - 1])

                vit_prev = V[i, t - 1, :] if V is not None else None
                v_i = self._vitals_feat(vit_prev)

                lvl = self._level_hist(y_i, t, self.m_hist)
                dh = self._delta_hist(y_i, t, self.m_dhist)
                pt = self._pt_feat(prev_code, A_dim)

                z = self._build_z(lvl, dh, pt, s_i, v_i).astype(np.float32)
                tgt = float(self._make_target(float(y_i[t - 1]), float(y_i[t])))

                Z_list.append(z); tgt_list.append(tgt); a_list.append(a_t)

        if not Z_list:
            raise RuntimeError("SC: no training steps extracted (check sequence lengths).")

        Z = np.stack(Z_list, axis=0).astype(np.float32)
        tgt = np.asarray(tgt_list, dtype=np.float32)
        a_step = np.asarray(a_list, dtype=np.int64)

        if Z.shape[0] > self.max_steps_total:
            idx = rng.choice(Z.shape[0], size=self.max_steps_total, replace=False)
            Z, tgt, a_step = Z[idx], tgt[idx], a_step[idx]

        self.z_mean = Z.mean(axis=0).astype(np.float32)

        # ==============================
        # (1) Stabilization: z_std floor
        # ==============================
        z_std = Z.std(axis=0)
        z_std = np.maximum(z_std, float(self.zstd_floor))
        self.z_std = z_std.astype(np.float32)

        Zs = ((Z - self.z_mean) / self.z_std).astype(np.float32)

        if self.sc_debug:
            print(
                f"[sc-debug dim{self.outcome_idx}] z_std min/med/max="
                f"{float(np.min(self.z_std)):.4g}/{float(np.median(self.z_std)):.4g}/{float(np.max(self.z_std)):.4g} | "
                f"#(z_std<1e-3)={int(np.sum(self.z_std < 1e-3))} | Zs_abs_max={float(np.max(np.abs(Zs))):.4g}"
            )

        self.Z_by_a = {}
        self.tgt_by_a = {}
        self.clip_by_a = {}
        for a in range(A_dim):
            idx_a = np.where(a_step == a)[0]
            if idx_a.size == 0:
                continue
            if idx_a.size > self.max_donors_per_action:
                idx_a = rng.choice(idx_a, size=self.max_donors_per_action, replace=False)
            Za = Zs[idx_a].astype(np.float32)
            ta = tgt[idx_a].astype(np.float32)
            self.Z_by_a[a] = Za
            self.tgt_by_a[a] = ta
            if self.clip_q is None:
                self.clip_by_a[a] = None
            else:
                lo, hi = np.quantile(ta, [self.clip_q, 1.0 - self.clip_q])
                self.clip_by_a[a] = (float(lo), float(hi))

        self.A_dim = A_dim
        self.z_dim = int(Zs.shape[1])
        self.available_actions = sorted(list(self.Z_by_a.keys()))
        self.fitted = True
        print(
            f"[sc-base] fit outcome_idx={self.outcome_idx} | steps={Zs.shape[0]} | z_dim={self.z_dim} | "
            f"A_dim={A_dim} | static_pca_dim={self.static_pca_dim} | vitals_pca_dim={self.vitals_pca_dim} | lambda_vitals={self.lambda_vitals} | "
            f"zstd_floor={self.zstd_floor:g}"
        )
        return self

    def predict_next_with_var_and_diag(self, *args, **kwargs):
        raise NotImplementedError("Base class: use DTSC_HistDeltaPlus_1D or AttnSC_HistDeltaPlus_1D")

# =============================================================================
# Option 1: DTSC Kernel
# =============================================================================

class DTSC_HistDeltaPlus_1D(HistDeltaPlus_Features_1D):
    def __init__(
        self,
        outcome_idx: int,
        m_hist: int = 10,
        m_dhist: int = 6,
        target_mode: str = "delta",
        eps_rel: float = 1e-3,
        max_donors_per_action: int = 20000,
        knn: int = 200,
        max_steps_total: int = 300000,
        seed: int = 0,
        use_prev_treat: bool = True,
        clip_q: float = 0.01,
        tau_dist: float = 0.05,
        kernel: str = "rbf",
        kernel_bw_mode: str = "median",
        kernel_bw: float = None,
        kernel_bw_scale: float = 2.0,
        static_pca_dim: int = 16,
        vitals_pca_dim: int = 8,
        lambda_vitals: float = 1.0,
        zstd_floor: float = 5e-2,
        sc_debug: bool = False,
    ):
        super().__init__(
            outcome_idx=outcome_idx,
            m_hist=m_hist,
            m_dhist=m_dhist,
            target_mode=target_mode,
            eps_rel=eps_rel,
            max_donors_per_action=max_donors_per_action,
            knn=knn,
            max_steps_total=max_steps_total,
            seed=seed,
            use_prev_treat=use_prev_treat,
            clip_q=clip_q,
            tau_dist=tau_dist,
            static_pca_dim=static_pca_dim,
            vitals_pca_dim=vitals_pca_dim,
            lambda_vitals=lambda_vitals,
            zstd_floor=zstd_floor,
            sc_debug=sc_debug,
        )
        self.kernel = str(kernel)
        self.kernel_bw_mode = str(kernel_bw_mode)
        self.kernel_bw = kernel_bw
        self.kernel_bw_scale = float(kernel_bw_scale)

    def predict_next_with_var_and_diag(
        self,
        y_hist_buffer,
        a_t: int,
        prev_code: int,
        tau_idx: int,
        x_static: np.ndarray = None,
        vit_prev: np.ndarray = None
    ):
        assert self.fitted
        y_prev = float(y_hist_buffer[0])
        a_t = int(a_t)

        if a_t not in self.Z_by_a:
            diag = {"ok": False, "reason": "no_donors_for_action", "ess": 0.0, "w_max": 1.0, "d2z_mean": np.nan}
            return y_prev, 1e-3, diag

        Z_pool = self.Z_by_a[a_t]
        tgt_pool = self.tgt_by_a[a_t]
        zt = self._build_z_target(y_hist_buffer, prev_code, x_static=x_static, vit_prev=vit_prev)

        # (1) Stabilization: compute distances in float64
        diff = Z_pool.astype(np.float64) - zt[None, :].astype(np.float64)
        d2 = np.sum(diff * diff, axis=1)

        K = min(self.knn, d2.size)
        nn_idx = np.argpartition(d2, K - 1)[:K]
        d2_knn = d2[nn_idx]
        tgt_knn = tgt_pool[nn_idx]

        d2_eff = d2_knn * max(self.tau_dist, 1e-8)
        w, h = kernel_weights_from_d2(
            d2=d2_eff,
            kernel=self.kernel,
            bandwidth_mode=self.kernel_bw_mode,
            bandwidth=self.kernel_bw,
            bandwidth_scale=self.kernel_bw_scale,
        )
        mu_u, var_u = weighted_mean_var(tgt_knn, w, min_var=1e-8)

        clip = self.clip_by_a.get(a_t, None)
        if clip is not None:
            lo, hi = clip
            mu_u = float(np.clip(mu_u, lo, hi))

        if self.target_mode == "delta":
            y_sc = y_prev + mu_u
            var_y = var_u
        elif self.target_mode == "rel_delta":
            c = max(abs(y_prev), self.eps_rel)
            y_sc = y_prev + mu_u * c
            var_y = (c ** 2) * var_u
        else:  # level
            y_sc = mu_u
            var_y = var_u

        ess = effective_sample_size(w) if w.size else 0.0
        w_max = float(np.max(w)) if w.size else 1.0
        diag = {
            "ok": True,
            "tau": int(tau_idx),
            "ess": float(ess),
            "w_max": float(w_max),
            "d2z_mean": float(np.mean(d2_knn)) if d2_knn.size else float("nan"),
            "kernel_h": float(h),
        }
        return float(y_sc), float(max(var_y, 1e-8)), diag

# =============================================================================
# Option 2: Cross-unit Attention SC
# =============================================================================

class DonorEmbedNet(nn.Module):
    def __init__(self, z_dim: int, emb_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(z_dim), int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(emb_dim)),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        e = self.net(z)
        e = e / (torch.norm(e, dim=-1, keepdim=True) + 1e-8)
        return e

class AttnSC_HistDeltaPlus_1D(HistDeltaPlus_Features_1D):
    def __init__(
        self,
        outcome_idx: int,
        m_hist: int = 10,
        m_dhist: int = 6,
        target_mode: str = "delta",
        eps_rel: float = 1e-3,
        max_donors_per_action: int = 20000,
        knn: int = 200,
        max_steps_total: int = 300000,
        seed: int = 0,
        use_prev_treat: bool = True,
        clip_q: float = 0.01,
        tau_dist: float = 0.05,
        static_pca_dim: int = 16,
        vitals_pca_dim: int = 8,
        lambda_vitals: float = 1.0,
        zstd_floor: float = 5e-2,
        sc_debug: bool = False,
        # attention hyperparams
        attn_emb_dim: int = 64,
        attn_hidden: int = 256,
        attn_dropout: float = 0.1,
        attn_temp: float = 1.0,
        attn_candidates_train: int = 128,
        attn_steps_per_epoch: int = 1500,
        attn_entropy_reg: float = -1e-3,
        attn_weight_decay: float = 1e-4,
    ):
        super().__init__(
            outcome_idx=outcome_idx,
            m_hist=m_hist,
            m_dhist=m_dhist,
            target_mode=target_mode,
            eps_rel=eps_rel,
            max_donors_per_action=max_donors_per_action,
            knn=knn,
            max_steps_total=max_steps_total,
            seed=seed,
            use_prev_treat=use_prev_treat,
            clip_q=clip_q,
            tau_dist=tau_dist,
            static_pca_dim=static_pca_dim,
            vitals_pca_dim=vitals_pca_dim,
            lambda_vitals=lambda_vitals,
            zstd_floor=zstd_floor,
            sc_debug=sc_debug,
        )

        self.attn_emb_dim = int(attn_emb_dim)
        self.attn_hidden = int(attn_hidden)
        self.attn_dropout = float(attn_dropout)
        self.attn_temp = float(attn_temp)
        self.attn_candidates_train = int(attn_candidates_train)
        self.attn_steps_per_epoch = int(attn_steps_per_epoch)
        self.attn_entropy_reg = float(attn_entropy_reg)
        self.attn_weight_decay = float(attn_weight_decay)

        self.embed_net = None
        self._trained = False

    def fit(self, train_subset):
        super().fit(train_subset)
        self.embed_net = DonorEmbedNet(
            z_dim=self.z_dim,
            emb_dim=self.attn_emb_dim,
            hidden=self.attn_hidden,
            dropout=self.attn_dropout,
        )
        self._trained = False
        return self

    def train_attention(
        self,
        device: str,
        epochs: int = 10,
        lr: float = 2e-3,
        batch_size: int = 256,
        candidates: int = None,
        steps_per_epoch: int = None,
        temp: float = None,
        entropy_reg: float = None,
        seed: int = 0,
        verbose_every: int = 1,
    ):
        assert self.fitted
        assert self.embed_net is not None

        device = str(device)
        rng = np.random.default_rng(int(seed))
        self.embed_net.to(device)

        candidates = int(self.attn_candidates_train if candidates is None else candidates)
        steps_per_epoch = int(self.attn_steps_per_epoch if steps_per_epoch is None else steps_per_epoch)
        temp = float(self.attn_temp if temp is None else temp)
        entropy_reg = float(self.attn_entropy_reg if entropy_reg is None else entropy_reg)

        opt = torch.optim.AdamW(self.embed_net.parameters(), lr=float(lr), weight_decay=float(self.attn_weight_decay))

        actions = list(self.available_actions)
        if not actions:
            print(f"[attn-sc dim{self.outcome_idx}] WARNING: no available actions, skip training.")
            return

        pools = {a: self.Z_by_a[a] for a in actions}
        tgts = {a: self.tgt_by_a[a] for a in actions}
        pool_sizes = {a: int(pools[a].shape[0]) for a in actions}

        for ep in range(1, int(epochs) + 1):
            self.embed_net.train()
            loss_sum = 0.0
            mse_sum = 0.0
            ent_sum = 0.0
            nb = 0

            for _ in range(int(steps_per_epoch)):
                a_batch = rng.choice(actions, size=int(batch_size), replace=True)

                z_q = np.zeros((batch_size, self.z_dim), dtype=np.float32)
                y_true = np.zeros((batch_size,), dtype=np.float32)
                z_c = np.zeros((batch_size, candidates, self.z_dim), dtype=np.float32)
                y_c = np.zeros((batch_size, candidates), dtype=np.float32)

                for b in range(batch_size):
                    a = int(a_batch[b])
                    n = pool_sizes[a]
                    if n <= 1:
                        continue
                    q_idx = int(rng.integers(0, n))
                    rep = (n < candidates)
                    c_idx = rng.choice(n, size=candidates, replace=rep)

                    z_q[b] = pools[a][q_idx]
                    y_true[b] = tgts[a][q_idx]
                    z_c[b] = pools[a][c_idx]
                    y_c[b] = tgts[a][c_idx]

                z_q_t = torch.tensor(z_q, dtype=torch.float32, device=device)
                z_c_t = torch.tensor(z_c, dtype=torch.float32, device=device)
                y_true_t = torch.tensor(y_true, dtype=torch.float32, device=device)
                y_c_t = torch.tensor(y_c, dtype=torch.float32, device=device)

                e_q = self.embed_net(z_q_t)
                e_c = self.embed_net(z_c_t.view(-1, self.z_dim)).view(batch_size, candidates, -1)

                scores = (e_q[:, None, :] * e_c).sum(dim=-1) / math.sqrt(max(e_q.shape[-1], 1))
                scores = scores / max(temp, 1e-6)
                w = torch.softmax(scores, dim=1)

                pred = (w * y_c_t).sum(dim=1)
                mse = (pred - y_true_t) ** 2
                ent = -(w * torch.log(w + 1e-12)).sum(dim=1)

                loss = mse.mean() + float(entropy_reg) * ent.mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.embed_net.parameters(), max_norm=5.0)
                opt.step()

                loss_sum += float(loss.item())
                mse_sum += float(mse.mean().item())
                ent_sum += float(ent.mean().item())
                nb += 1

            if (ep == 1) or (ep % max(int(verbose_every), 1) == 0) or (ep == epochs):
                print(
                    f"[attn-sc dim{self.outcome_idx}] ep {ep:03d}/{epochs} | "
                    f"loss={loss_sum/max(nb,1):.6f} | mse={mse_sum/max(nb,1):.6f} | ent={ent_sum/max(nb,1):.6f}"
                )

        self.embed_net.eval()
        self._trained = True

    @torch.no_grad()
    def predict_next_with_var_and_diag(
        self,
        y_hist_buffer,
        a_t: int,
        prev_code: int,
        tau_idx: int,
        x_static: np.ndarray = None,
        vit_prev: np.ndarray = None,
        device: str = "cpu",
    ):
        assert self.fitted
        y_prev = float(y_hist_buffer[0])
        a_t = int(a_t)

        if a_t not in self.Z_by_a:
            diag = {"ok": False, "reason": "no_donors_for_action", "ess": 0.0, "w_max": 1.0, "d2z_mean": np.nan}
            return y_prev, 1e-3, diag

        Z_pool = self.Z_by_a[a_t]
        tgt_pool = self.tgt_by_a[a_t]
        zt = self._build_z_target(y_hist_buffer, prev_code, x_static=x_static, vit_prev=vit_prev)

        # (1) Stabilization: compute distances in float64
        diff = Z_pool.astype(np.float64) - zt[None, :].astype(np.float64)
        d2 = np.sum(diff * diff, axis=1)

        K = min(self.knn, d2.size)
        nn_idx = np.argpartition(d2, K - 1)[:K]
        d2_knn = d2[nn_idx]
        Z_knn = Z_pool[nn_idx]
        tgt_knn = tgt_pool[nn_idx]

        if self.embed_net is None:
            w = np.ones((K,), dtype=np.float32) / max(K, 1)
            entropy = float(-np.sum(w * np.log(w + 1e-12)))
            score_gap = float("nan")
        else:
            self.embed_net.to(device)
            self.embed_net.eval()

            zt_t = torch.tensor(zt[None, :], dtype=torch.float32, device=device)
            Zt_t = torch.tensor(Z_knn, dtype=torch.float32, device=device)

            e_q = self.embed_net(zt_t)
            e_k = self.embed_net(Zt_t)

            scores = (e_k * e_q[0]).sum(dim=-1) / math.sqrt(max(e_k.shape[-1], 1))
            scores = scores / max(float(self.attn_temp), 1e-6)
            w_t = torch.softmax(scores, dim=0)
            w = w_t.detach().cpu().numpy().astype(np.float32)

            entropy = float(-(w * np.log(w + 1e-12)).sum())
            s_np = scores.detach().cpu().numpy().astype(np.float32)
            if s_np.size >= 2:
                top2 = np.partition(s_np, -2)[-2:]
                score_gap = float(np.max(top2) - np.min(top2))
            else:
                score_gap = float("nan")

        mu_u, var_u = weighted_mean_var(tgt_knn, w, min_var=1e-8)

        clip = self.clip_by_a.get(a_t, None)
        if clip is not None:
            lo, hi = clip
            mu_u = float(np.clip(mu_u, lo, hi))

        if self.target_mode == "delta":
            y_sc = y_prev + mu_u
            var_y = var_u
        elif self.target_mode == "rel_delta":
            c = max(abs(y_prev), self.eps_rel)
            y_sc = y_prev + mu_u * c
            var_y = (c ** 2) * var_u
        else:  # level
            y_sc = mu_u
            var_y = var_u

        ess = effective_sample_size(w) if w.size else 0.0
        w_max = float(np.max(w)) if w.size else 1.0
        diag = {
            "ok": True,
            "tau": int(tau_idx),
            "ess": float(ess),
            "w_max": float(w_max),
            "d2z_mean": float(np.mean(d2_knn)) if d2_knn.size else float("nan"),
            "attn_entropy": float(entropy),
            "attn_score_gap": float(score_gap),
        }
        return float(y_sc), float(max(var_y, 1e-8)), diag

# =============================================================================
# Multi-outcome wrappers
# =============================================================================

class DTSC_MultiOutcome:
    def __init__(
        self,
        Dy: int,
        static_pca_dim: int = 16,
        static_pca_whiten: bool = True,
        vitals_pca_dim: int = 8,
        vitals_pca_whiten: bool = True,
        lambda_vitals: float = 1.0,
        max_vitals_pca_samples: int = 20000,
        zstd_floor: float = 5e-2,
        sc_debug: bool = False,
        **kwargs
    ):
        self.Dy = int(Dy)
        self.static_pca_dim = int(static_pca_dim)
        self.static_pca_whiten = bool(static_pca_whiten)
        self.vitals_pca_dim = int(vitals_pca_dim)
        self.vitals_pca_whiten = bool(vitals_pca_whiten)
        self.lambda_vitals = float(lambda_vitals)
        self.max_vitals_pca_samples = int(max_vitals_pca_samples)
        self.zstd_floor = float(zstd_floor)
        self.sc_debug = bool(sc_debug)

        self.models = [
            DTSC_HistDeltaPlus_1D(
                outcome_idx=j,
                static_pca_dim=self.static_pca_dim,
                vitals_pca_dim=self.vitals_pca_dim,
                lambda_vitals=self.lambda_vitals,
                zstd_floor=self.zstd_floor,
                sc_debug=self.sc_debug,
                **kwargs
            )
            for j in range(self.Dy)
        ]

        self.fitted = False
        self.static_pca = None
        self.vitals_pca = None

    def _fit_static_pca(self, train_subset):
        d = train_subset.data
        if "static_features" not in d:
            print("[static-pca] WARNING: no static_features found; using zeros in SC.")
            return None
        Xs = np.asarray(d["static_features"]).astype(float)
        Xs = _safe_impute_nan_with_mean(Xs)
        pca = PCAProjector(n_components=self.static_pca_dim, whiten=self.static_pca_whiten).fit(Xs)
        print(f"[static-pca] fit on train_f | D_in={Xs.shape[1]} -> K={pca.K_} | whiten={self.static_pca_whiten}")
        return pca

    def _fit_vitals_pca(self, train_subset):
        d = train_subset.data
        if "vitals" not in d:
            print("[vitals-pca] WARNING: no vitals found; using zeros in SC.")
            return None

        V = np.asarray(d["vitals"]).astype(float)
        seq_lens = np.asarray(d["sequence_lengths"]).astype(int)
        N, T, Dv = V.shape

        rows = []
        for i in range(N):
            L = int(seq_lens[i])
            if L <= 1:
                continue
            rows.append(V[i, :max(L-1, 1), :])

        if not rows:
            print("[vitals-pca] WARNING: no valid vitals rows; using zeros in SC.")
            return None

        Xv = np.concatenate(rows, axis=0)
        if Xv.shape[0] > self.max_vitals_pca_samples:
            rng = np.random.default_rng(0)
            idx = rng.choice(Xv.shape[0], size=self.max_vitals_pca_samples, replace=False)
            Xv = Xv[idx]

        Xv = _safe_impute_nan_with_mean(Xv)
        pca = PCAProjector(n_components=self.vitals_pca_dim, whiten=self.vitals_pca_whiten).fit(Xv)
        print(f"[vitals-pca] fit on train_f | D_in={Dv} -> K={pca.K_} | whiten={self.vitals_pca_whiten} | samples={Xv.shape[0]}")
        return pca

    def fit(self, train_subset):
        self.static_pca = self._fit_static_pca(train_subset)
        self.vitals_pca = self._fit_vitals_pca(train_subset)

        for m in self.models:
            m.set_static_pca(self.static_pca)
            m.set_vitals_pca(self.vitals_pca)
            m.fit(train_subset)

        self.fitted = True
        return self

    def predict_one_step(
        self,
        y_hist_vec,
        a_t: int,
        prev_code: int,
        tau_idx: int,
        x_static: np.ndarray = None,
        vit_prev: np.ndarray = None,
        process_noise: float = 1e-6
    ):
        assert self.fitted
        if not y_hist_vec:
            y_hist_vec = [np.zeros((self.Dy,), dtype=float)]

        mu = np.zeros((self.Dy,), dtype=float)
        var = np.zeros((self.Dy,), dtype=float)
        diags = [None for _ in range(self.Dy)]
        for j in range(self.Dy):
            y_hist_j = [float(v[j]) for v in y_hist_vec]
            muj, varj, diagj = self.models[j].predict_next_with_var_and_diag(
                y_hist_buffer=y_hist_j,
                a_t=int(a_t),
                prev_code=int(prev_code),
                tau_idx=int(tau_idx),
                x_static=x_static,
                vit_prev=vit_prev,
            )
            mu[j] = muj
            var[j] = max(varj + float(process_noise), 1e-8)
            diags[j] = diagj
        return mu, var, diags

class AttentionSC_MultiOutcome:
    def __init__(
        self,
        Dy: int,
        static_pca_dim: int = 16,
        static_pca_whiten: bool = True,
        vitals_pca_dim: int = 8,
        vitals_pca_whiten: bool = True,
        lambda_vitals: float = 1.0,
        max_vitals_pca_samples: int = 20000,
        zstd_floor: float = 5e-2,
        sc_debug: bool = False,
        attn_device: str = "cpu",
        attn_epochs: int = 10,
        attn_lr: float = 2e-3,
        attn_batch_size: int = 256,
        attn_candidates: int = 128,
        attn_steps_per_epoch: int = 1500,
        attn_temp: float = 1.0,
        attn_entropy_reg: float = -1e-3,
        attn_verbose_every: int = 1,
        **kwargs
    ):
        self.Dy = int(Dy)
        self.static_pca_dim = int(static_pca_dim)
        self.static_pca_whiten = bool(static_pca_whiten)
        self.vitals_pca_dim = int(vitals_pca_dim)
        self.vitals_pca_whiten = bool(vitals_pca_whiten)
        self.lambda_vitals = float(lambda_vitals)
        self.max_vitals_pca_samples = int(max_vitals_pca_samples)
        self.zstd_floor = float(zstd_floor)
        self.sc_debug = bool(sc_debug)

        self.attn_device = str(attn_device)
        self.attn_epochs = int(attn_epochs)
        self.attn_lr = float(attn_lr)
        self.attn_batch_size = int(attn_batch_size)
        self.attn_candidates = int(attn_candidates)
        self.attn_steps_per_epoch = int(attn_steps_per_epoch)
        self.attn_temp = float(attn_temp)
        self.attn_entropy_reg = float(attn_entropy_reg)
        self.attn_verbose_every = int(attn_verbose_every)

        self.models = [
            AttnSC_HistDeltaPlus_1D(
                outcome_idx=j,
                static_pca_dim=self.static_pca_dim,
                vitals_pca_dim=self.vitals_pca_dim,
                lambda_vitals=self.lambda_vitals,
                zstd_floor=self.zstd_floor,
                sc_debug=self.sc_debug,
                attn_temp=self.attn_temp,
                attn_entropy_reg=self.attn_entropy_reg,
                attn_candidates_train=self.attn_candidates,
                attn_steps_per_epoch=self.attn_steps_per_epoch,
                **kwargs
            )
            for j in range(self.Dy)
        ]

        self.fitted = False
        self.static_pca = None
        self.vitals_pca = None

    def _fit_static_pca(self, train_subset):
        d = train_subset.data
        if "static_features" not in d:
            print("[static-pca] WARNING: no static_features found; using zeros in SC.")
            return None
        Xs = np.asarray(d["static_features"]).astype(float)
        Xs = _safe_impute_nan_with_mean(Xs)
        pca = PCAProjector(n_components=self.static_pca_dim, whiten=self.static_pca_whiten).fit(Xs)
        print(f"[static-pca] fit on train_f | D_in={Xs.shape[1]} -> K={pca.K_} | whiten={self.static_pca_whiten}")
        return pca

    def _fit_vitals_pca(self, train_subset):
        d = train_subset.data
        if "vitals" not in d:
            print("[vitals-pca] WARNING: no vitals found; using zeros in SC.")
            return None

        V = np.asarray(d["vitals"]).astype(float)
        seq_lens = np.asarray(d["sequence_lengths"]).astype(int)
        N, T, Dv = V.shape

        rows = []
        for i in range(N):
            L = int(seq_lens[i])
            if L <= 1:
                continue
            rows.append(V[i, :max(L-1, 1), :])

        if not rows:
            print("[vitals-pca] WARNING: no valid vitals rows; using zeros in SC.")
            return None

        Xv = np.concatenate(rows, axis=0)
        if Xv.shape[0] > self.max_vitals_pca_samples:
            rng = np.random.default_rng(0)
            idx = rng.choice(Xv.shape[0], size=self.max_vitals_pca_samples, replace=False)
            Xv = Xv[idx]

        Xv = _safe_impute_nan_with_mean(Xv)
        pca = PCAProjector(n_components=self.vitals_pca_dim, whiten=self.vitals_pca_whiten).fit(Xv)
        print(f"[vitals-pca] fit on train_f | D_in={Dv} -> K={pca.K_} | whiten={self.vitals_pca_whiten} | samples={Xv.shape[0]}")
        return pca

    def fit(self, train_subset, device_for_training: str = None):
        self.static_pca = self._fit_static_pca(train_subset)
        self.vitals_pca = self._fit_vitals_pca(train_subset)

        for m in self.models:
            m.set_static_pca(self.static_pca)
            m.set_vitals_pca(self.vitals_pca)
            m.fit(train_subset)

        dev = self.attn_device if device_for_training is None else str(device_for_training)
        print(f"[attn-sc] train attention on device={dev} | epochs={self.attn_epochs} | lr={self.attn_lr} | bs={self.attn_batch_size} | cand={self.attn_candidates}")
        for m in self.models:
            m.train_attention(
                device=dev,
                epochs=self.attn_epochs,
                lr=self.attn_lr,
                batch_size=self.attn_batch_size,
                candidates=self.attn_candidates,
                steps_per_epoch=self.attn_steps_per_epoch,
                temp=self.attn_temp,
                entropy_reg=self.attn_entropy_reg,
                seed=123 + int(m.outcome_idx),
                verbose_every=self.attn_verbose_every,
            )

        self.fitted = True
        return self

    def predict_one_step(
        self,
        y_hist_vec,
        a_t: int,
        prev_code: int,
        tau_idx: int,
        x_static: np.ndarray = None,
        vit_prev: np.ndarray = None,
        process_noise: float = 1e-6,
        device: str = "cpu",
    ):
        assert self.fitted
        if not y_hist_vec:
            y_hist_vec = [np.zeros((self.Dy,), dtype=float)]

        mu = np.zeros((self.Dy,), dtype=float)
        var = np.zeros((self.Dy,), dtype=float)
        diags = [None for _ in range(self.Dy)]
        for j in range(self.Dy):
            y_hist_j = [float(v[j]) for v in y_hist_vec]
            muj, varj, diagj = self.models[j].predict_next_with_var_and_diag(
                y_hist_buffer=y_hist_j,
                a_t=int(a_t),
                prev_code=int(prev_code),
                tau_idx=int(tau_idx),
                x_static=x_static,
                vit_prev=vit_prev,
                device=device,
            )
            mu[j] = muj
            var[j] = max(varj + float(process_noise), 1e-8)
            diags[j] = diagj
        return mu, var, diags

# =============================================================================
# Reliability r(phi) per outcome dim
# =============================================================================

def sc_diag_to_features(diag: dict, tau: int, horizon: int):
    if not isinstance(diag, dict) or not diag.get("ok", False):
        return np.array([1.0, 0.0, 0.0, 10.0, 1.0, float(tau / max(horizon - 1, 1))], dtype=np.float32)
    ess = float(diag.get("ess", 0.0))
    w_max = float(diag.get("w_max", 0.0))
    d2z = float(diag.get("d2z_mean", 10.0))
    tau_n = float(tau / max(horizon - 1, 1))
    return np.array(
        [1.0, np.log1p(max(ess, 0.0)), float(np.clip(w_max, 0.0, 1.0)), max(d2z, 0.0), 0.0, tau_n],
        dtype=np.float32
    )

class SCReliabilityCalibrator(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.lin = nn.Linear(int(feat_dim), 1)

    def forward(self, phi: torch.Tensor):
        logits = self.lin(phi).squeeze(-1)
        r = torch.sigmoid(logits)
        return torch.clamp(r, 1e-4, 1.0 - 1e-4)

@torch.no_grad()
def reliability_from_diag(diag: dict, tau: int, horizon: int, model: nn.Module, device: str):
    if model is None:
        return 0.0
    phi = sc_diag_to_features(diag, tau=tau, horizon=horizon)
    x = torch.tensor(phi[None, :], dtype=torch.float32, device=device)
    r = float(model(x).detach().cpu().numpy().reshape(-1)[0])
    return float(np.clip(r, 0.0, 1.0))

def fit_reliability_models(
    sc_model,
    subset,
    horizon: int,
    taus: list,
    max_windows: int,
    device: str,
    lr: float = 5e-2,
    epochs: int = 250,
    batch_size: int = 512,
    R_min: float = 1e-4,
    eps: float = 1e-3,
    process_noise: float = 1e-6,
    seed: int = 123,
    sc_var_scale2: np.ndarray = None,
):
    """
    Fit per-outcome reliability r(phi) by minimizing Gaussian NLL for the measurement model:
      y ~ N(mu_sc, R_eff), where R_eff = var_sc/(r+eps) + R_min

    Rollout-consistent: history update uses SC rollout (mu_sc).
    """
    d = subset.data
    Y = np.asarray(d["outputs"]).astype(float)
    A_codes = np.asarray(d["current_treatments"]).argmax(axis=-1).astype(int)
    X_static = np.asarray(d["static_features"]).astype(float) if "static_features" in d else None
    V = np.asarray(d["vitals"]).astype(float) if "vitals" in d else None
    seq_lens = np.asarray(d["sequence_lengths"]).astype(int)

    N, T, Dy = Y.shape
    H = int(horizon)
    taus = sorted(set(int(t) for t in taus if 0 <= int(t) < H))
    taus_set = set(taus)

    rng = np.random.default_rng(seed)
    idxs = np.arange(N)
    if max_windows is not None and N > int(max_windows):
        idxs = rng.choice(idxs, size=int(max_windows), replace=False)

    models = [SCReliabilityCalibrator(feat_dim=6).to(device) for _ in range(Dy)]
    opts = [torch.optim.AdamW(m.parameters(), lr=float(lr), weight_decay=1e-4) for m in models]

    feats = [[] for _ in range(Dy)]
    mu_list = [[] for _ in range(Dy)]
    var_list = [[] for _ in range(Dy)]
    ytrue_list = [[] for _ in range(Dy)]

    split = get_future_past_split(subset, horizon=H)
    hist_keep = sc_model.models[0].m_hist + 5

    for i in idxs:
        L = int(seq_lens[i])
        t0 = int(split[i])
        if t0 <= 0 or t0 >= L:
            continue
        H_i = min(H, L - t0)
        if H_i <= 0:
            continue

        policy_codes = A_codes[i, t0:t0 + H_i]
        x_i = X_static[i] if X_static is not None else None

        y_hist_vec = []
        for tt in range(t0 - 1, -1, -1):
            y_hist_vec.append(Y[i, tt, :].astype(float).reshape(-1))
            if len(y_hist_vec) >= hist_keep:
                break
        if not y_hist_vec:
            y_hist_vec = [np.zeros((Dy,), dtype=float)]

        for tau in range(H_i):
            if tau not in taus_set:
                continue

            prev_code = int(A_codes[i, t0 - 1]) if tau == 0 else int(policy_codes[tau - 1])
            a_t = int(policy_codes[tau])

            vit_prev = None
            if V is not None:
                idx_v = (t0 + tau - 1)
                if 0 <= idx_v < L:
                    vit_prev = V[i, idx_v, :]

            if isinstance(sc_model, AttentionSC_MultiOutcome):
                mu_sc, var_sc, diags = sc_model.predict_one_step(
                    y_hist_vec=y_hist_vec,
                    a_t=a_t,
                    prev_code=prev_code,
                    tau_idx=tau,
                    x_static=x_i,
                    vit_prev=vit_prev,
                    process_noise=process_noise,
                    device=device,
                )
            else:
                mu_sc, var_sc, diags = sc_model.predict_one_step(
                    y_hist_vec=y_hist_vec,
                    a_t=a_t,
                    prev_code=prev_code,
                    tau_idx=tau,
                    x_static=x_i,
                    vit_prev=vit_prev,
                    process_noise=process_noise,
                )

            if sc_var_scale2 is not None and tau < sc_var_scale2.shape[0]:
                var_sc = var_sc * sc_var_scale2[tau, :]

            y_true = Y[i, t0 + tau, :].astype(float).reshape(-1)

            for j in range(Dy):
                if not np.isfinite(mu_sc[j]) or not np.isfinite(var_sc[j]) or not np.isfinite(y_true[j]):
                    continue
                phi = sc_diag_to_features(diags[j], tau=tau, horizon=H)
                feats[j].append(phi)
                mu_list[j].append(float(mu_sc[j]))
                var_list[j].append(float(max(var_sc[j], 1e-8)))
                ytrue_list[j].append(float(y_true[j]))

            y_hist_vec = [mu_sc.astype(float).reshape(-1)] + y_hist_vec
            if len(y_hist_vec) > hist_keep:
                y_hist_vec = y_hist_vec[:hist_keep]

    for j in range(Dy):
        if len(feats[j]) == 0:
            print(f"[calib-r] dim{j}: no points, leaving model None (r=0).")
            models[j] = None
            continue

        X = torch.tensor(np.stack(feats[j], axis=0), dtype=torch.float32, device=device)
        MU = torch.tensor(np.asarray(mu_list[j]), dtype=torch.float32, device=device)
        VAR = torch.tensor(np.asarray(var_list[j]), dtype=torch.float32, device=device)
        YT = torch.tensor(np.asarray(ytrue_list[j]), dtype=torch.float32, device=device)

        ds = TensorDataset(X, MU, VAR, YT)
        loader = DataLoader(ds, batch_size=min(int(batch_size), len(ds)), shuffle=True)

        best = float("inf")
        best_state = None

        for ep in range(1, int(epochs) + 1):
            models[j].train()
            loss_sum = 0.0
            nb = 0
            for xb, mub, varb, yb in loader:
                r = models[j](xb)
                R_eff = varb / (r + float(eps)) + float(R_min)
                nll = 0.5 * (((yb - mub) ** 2) / R_eff + torch.log(R_eff))
                loss = nll.mean()

                opts[j].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(models[j].parameters(), max_norm=5.0)
                opts[j].step()

                loss_sum += float(loss.item())
                nb += 1

            avg = loss_sum / max(nb, 1)
            if avg < best:
                best = avg
                best_state = {k: v.detach().cpu().clone() for k, v in models[j].state_dict().items()}

            if ep in (1, epochs) or ep % 50 == 0:
                with torch.no_grad():
                    models[j].eval()
                    r_mean = float(models[j](X).mean().item())
                print(f"[calib-r dim{j}] ep {ep:03d}/{epochs} | meas-NLL={avg:.4f} | r_mean={r_mean:.3f}")

        if best_state is not None:
            models[j].load_state_dict(best_state)

    return models

# =============================================================================
# [B] Variance calibration utilities
# =============================================================================

def fit_var_scale2_per_tau(mu, var, y, mask, H: int, clip=(0.25, 25.0), min_points: int = 10):
    """
    Returns scale2[tau, dim] such that calibrated var = var * scale2.
    scale2 ≈ E[(err^2)/var] over masked valid points.
    Shapes: (N,H,Dy).
    """
    mu = np.asarray(mu, float)
    var = np.maximum(np.asarray(var, float), 1e-8)
    y  = np.asarray(y, float)
    m  = (np.asarray(mask, float) > 0.0)

    N, H2, Dy = mu.shape
    assert H2 == H

    scale2 = np.ones((H, Dy), dtype=float)
    for t in range(H):
        for j in range(Dy):
            ok = m[:, t, j] & np.isfinite(mu[:, t, j]) & np.isfinite(var[:, t, j]) & np.isfinite(y[:, t, j])
            if int(ok.sum()) < int(min_points):
                continue
            err2 = (y[ok, t, j] - mu[ok, t, j]) ** 2
            ratio = float(np.mean(err2 / var[ok, t, j]))
            if np.isfinite(ratio) and ratio > 0.0:
                scale2[t, j] = float(np.clip(ratio, float(clip[0]), float(clip[1])))
    return scale2

def apply_var_scale2(var, scale2, H: int):
    """
    var: (N,H,Dy) or (H,Dy) or (Dy,)
    scale2: (H,Dy)
    """
    var = np.asarray(var, float)
    scale2 = np.asarray(scale2, float)
    if scale2.ndim != 2:
        raise ValueError("scale2 must be (H,Dy)")
    if scale2.shape[0] < H:
        raise ValueError(f"scale2 H={scale2.shape[0]} < required H={H}")
    if var.ndim == 3:
        return var * scale2[None, :H, :]
    if var.ndim == 2:
        return var[:H, :] * scale2[:H, :]
    if var.ndim == 1:
        return var * scale2[0, :]
    raise ValueError(f"Unexpected var shape {var.shape}")

# =============================================================================
# CI fusion (diagonal)
# =============================================================================

def _ci_fuse_1d(mu1, var1, mu2, var2, omega, min_var: float = 1e-8):
    """
    CI for 1D Gaussians with unknown correlation.
      P^{-1} = ω P1^{-1} + (1-ω) P2^{-1}
      μ      = P ( ω P1^{-1} μ1 + (1-ω) P2^{-1} μ2 )

    Returns:
      mu_f, var_f, w2_eff
    where w2_eff is the effective contribution of μ2 in μ_f (information-weighted).
    """
    v1 = float(max(var1, min_var))
    v2 = float(max(var2, min_var))
    w = float(np.clip(omega, 0.0, 1.0))

    inv1 = 1.0 / v1
    inv2 = 1.0 / v2
    invP = w * inv1 + (1.0 - w) * inv2
    invP = float(max(invP, 1e-18))
    var_f = 1.0 / invP

    num = w * inv1 * float(mu1) + (1.0 - w) * inv2 * float(mu2)
    mu_f = var_f * num

    w2_eff = ((1.0 - w) * inv2) / invP
    return float(mu_f), float(max(var_f, min_var)), float(np.clip(w2_eff, 0.0, 1.0))

def _ci_choose_omega_diag(var1_vec, var2_vec, grid_size: int = 51, objective: str = "logdet"):
    """
    Choose ω ∈ [0,1] for diagonal CI by minimizing a conservative objective.
    objective:
      - "logdet": minimize sum_j log(P_j)   (det for diagonal cov)
      - "trace" : minimize sum_j P_j
    """
    v1 = np.maximum(np.asarray(var1_vec, float).reshape(-1), 1e-8)
    v2 = np.maximum(np.asarray(var2_vec, float).reshape(-1), 1e-8)
    omegas = np.linspace(0.0, 1.0, int(max(grid_size, 2)))

    best_w = 0.5
    best_obj = float("inf")

    inv1 = 1.0 / v1
    inv2 = 1.0 / v2

    for w in omegas:
        invP = w * inv1 + (1.0 - w) * inv2
        invP = np.maximum(invP, 1e-18)
        P = 1.0 / invP
        if objective == "trace":
            obj = float(np.sum(P))
        else:
            obj = float(np.sum(np.log(P)))
        if obj < best_obj:
            best_obj = obj
            best_w = float(w)

    return float(np.clip(best_w, 0.0, 1.0))

# =============================================================================
# Rollout: prior + SC + CI fusion
# =============================================================================

@torch.no_grad()
def rollout_cf_future(
    transformer: nn.Module,
    sc_model,
    subset,
    i: int,
    t0: int,
    H: int,
    layout: dict,
    device: str,
    reliability_models=None,
    R_min: float = 1e-4,
    process_noise_sc: float = 1e-6,
    prior_var_scale2: np.ndarray = None,   # (H_global, Dy)
    sc_var_scale2: np.ndarray = None,      # (H_global, Dy)
    ci_grid_size: int = 51,
    ci_objective: str = "logdet",
):
    d = subset.data
    Y = np.asarray(d["outputs"])[i, :, :].astype(float)
    A_onehot = np.asarray(d["current_treatments"])[i, :, :].astype(float)
    V = np.asarray(d["vitals"])[i, :, :].astype(float) if "vitals" in d else None
    Xs = np.asarray(d["static_features"])[i, :].astype(float) if "static_features" in d else None
    L = int(np.asarray(d["sequence_lengths"])[i])

    Dy = int(Y.shape[-1])
    A_dim = int(A_onehot.shape[-1])

    H = min(int(H), max(0, L - t0))
    if H <= 0:
        return None

    A_codes_obs = A_onehot.argmax(axis=-1).astype(int)
    policy_codes = A_codes_obs[t0:t0 + H]

    tokens = []
    y_prev = np.zeros((Dy,), dtype=np.float32)
    for t in range(0, t0):
        a_t = A_onehot[t]
        vitals_t = V[t] if V is not None else None
        tok = build_token_from_parts(y_prev, a_t, Xs, vitals_t, layout["order"], layout["D_vitals"])
        tokens.append(tok)
        y_prev = Y[t].astype(np.float32).reshape(-1)

    prior_mean = np.zeros((H, Dy), dtype=float)
    prior_var = np.zeros((H, Dy), dtype=float)
    sc_mean = np.zeros((H, Dy), dtype=float)
    sc_var = np.zeros((H, Dy), dtype=float)
    fused_mean = np.zeros((H, Dy), dtype=float)
    fused_var = np.zeros((H, Dy), dtype=float)

    r_arr = np.zeros((H, Dy), dtype=float)
    K_arr = np.zeros((H, Dy), dtype=float)
    R_eff_arr = np.zeros((H, Dy), dtype=float)
    omega_arr = np.zeros((H,), dtype=float)

    hist_keep = sc_model.models[0].m_hist + 5
    y_hist_vec = []
    for tt in range(t0 - 1, -1, -1):
        y_hist_vec.append(Y[tt].astype(float).reshape(-1))
        if len(y_hist_vec) >= hist_keep:
            break
    if not y_hist_vec:
        y_hist_vec = [np.zeros((Dy,), dtype=float)]

    y_prev_roll = Y[t0 - 1].astype(float).reshape(-1) if t0 - 1 >= 0 else np.zeros((Dy,), dtype=float)

    transformer.eval()

    for k in range(H):
        t = t0 + k

        a_onehot = _onehot_from_code(int(policy_codes[k]), A_dim)
        vitals_t = V[t] if V is not None else None
        tok = build_token_from_parts(y_prev_roll, a_onehot, Xs, vitals_t, layout["order"], layout["D_vitals"])
        tokens.append(tok)

        x_seq = torch.tensor(np.stack(tokens, axis=0), dtype=torch.float32, device=device).unsqueeze(0)
        pad_mask = torch.zeros((1, x_seq.shape[1]), dtype=torch.bool, device=device)
        mean_bt, logvar_bt = transformer(x_seq, src_key_padding_mask=pad_mask)
        mu_T = mean_bt[0, -1, :].detach().cpu().numpy().astype(float)
        logvar_T = np.clip(logvar_bt[0, -1, :].detach().cpu().numpy().astype(float), -10.0, 8.0)
        var_T = np.exp(logvar_T) + 1e-6

        if prior_var_scale2 is not None and k < prior_var_scale2.shape[0]:
            var_T = var_T * prior_var_scale2[k, :]

        prior_mean[k] = mu_T
        prior_var[k] = var_T

        prev_code = int(A_codes_obs[t0 - 1]) if k == 0 else int(policy_codes[k - 1])

        vit_prev = None
        if V is not None:
            idx_v = (t0 + k - 1)
            if 0 <= idx_v < L:
                vit_prev = V[idx_v]

        if isinstance(sc_model, AttentionSC_MultiOutcome):
            mu_sc, var_sc_step, diags = sc_model.predict_one_step(
                y_hist_vec=y_hist_vec,
                a_t=int(policy_codes[k]),
                prev_code=prev_code,
                tau_idx=k,
                x_static=Xs,
                vit_prev=vit_prev,
                process_noise=process_noise_sc,
                device=device,
            )
        else:
            mu_sc, var_sc_step, diags = sc_model.predict_one_step(
                y_hist_vec=y_hist_vec,
                a_t=int(policy_codes[k]),
                prev_code=prev_code,
                tau_idx=k,
                x_static=Xs,
                vit_prev=vit_prev,
                process_noise=process_noise_sc,
            )

        if sc_var_scale2 is not None and k < sc_var_scale2.shape[0]:
            var_sc_step = var_sc_step * sc_var_scale2[k, :]

        sc_mean[k] = mu_sc
        sc_var[k] = var_sc_step

        # ---- CI fusion ----
        R_eff_vec = np.zeros((Dy,), dtype=float)
        r_vec = np.zeros((Dy,), dtype=float)
        for j in range(Dy):
            r_j = 0.0
            if reliability_models is not None and j < len(reliability_models) and reliability_models[j] is not None:
                r_j = reliability_from_diag(diags[j], tau=k, horizon=H, model=reliability_models[j], device=device)
            r_j = float(np.clip(r_j, 0.0, 1.0))
            r_vec[j] = r_j
            R_eff_vec[j] = float(max(var_sc_step[j], 1e-8) / max(r_j, 1e-3) + float(R_min))

        omega_k = _ci_choose_omega_diag(var_T, R_eff_vec, grid_size=int(ci_grid_size), objective=str(ci_objective))
        omega_arr[k] = omega_k

        mu_F = np.zeros((Dy,), dtype=float)
        var_F = np.zeros((Dy,), dtype=float)

        for j in range(Dy):
            muF_j, varF_j, w_sc_eff = _ci_fuse_1d(
                mu1=mu_T[j],
                var1=var_T[j],
                mu2=mu_sc[j],
                var2=R_eff_vec[j],
                omega=omega_k,
            )
            mu_F[j] = muF_j
            var_F[j] = varF_j
            r_arr[k, j] = r_vec[j]
            K_arr[k, j] = w_sc_eff
            R_eff_arr[k, j] = R_eff_vec[j]

        fused_mean[k] = mu_F
        fused_var[k] = var_F

        # feedback
        y_prev_roll = mu_F.astype(float).reshape(-1)
        y_hist_vec = [mu_F.astype(float).reshape(-1)] + y_hist_vec
        if len(y_hist_vec) > hist_keep:
            y_hist_vec = y_hist_vec[:hist_keep]

    return {
        "prior_mean": prior_mean, "prior_var": np.maximum(prior_var, 1e-8),
        "sc_mean": sc_mean, "sc_var": np.maximum(sc_var, 1e-8),
        "fused_mean": fused_mean, "fused_var": np.maximum(fused_var, 1e-8),
        "r": r_arr, "K": K_arr, "R_eff": R_eff_arr, "omega": omega_arr,
    }

# =============================================================================
# Collect forecasts (RMSE / NLL)
# =============================================================================

@torch.no_grad()
def collect_gaussian_forecasts(
    transformer,
    sc_model,
    subset,
    horizon: int,
    layout: dict,
    max_windows: int,
    device: str,
    reliability_models=None,
    R_min: float = 1e-4,
    process_noise_sc: float = 1e-6,
    prior_var_scale2: np.ndarray = None,
    sc_var_scale2: np.ndarray = None,
):
    d = subset.data
    Y_all = np.asarray(d["outputs"]).astype(float)
    seq_lens = np.asarray(d["sequence_lengths"]).astype(int)

    H = int(horizon)
    split = get_future_past_split(subset, horizon=H)
    N0, T, Dy = Y_all.shape

    idxs = np.arange(N0)
    if max_windows is not None and N0 > int(max_windows):
        rng = np.random.default_rng(0)
        idxs = rng.choice(idxs, size=int(max_windows), replace=False)

    ds = MIMIC3SyntheticTimeSeriesDataset(subset)

    mu_T = np.full((len(idxs), H, Dy), np.nan, float)
    var_T = np.full((len(idxs), H, Dy), np.nan, float)

    # SC evidence from fused rollout
    mu_SC = np.full((len(idxs), H, Dy), np.nan, float)
    var_SC = np.full((len(idxs), H, Dy), np.nan, float)

    mu_F = np.full((len(idxs), H, Dy), np.nan, float)
    var_F = np.full((len(idxs), H, Dy), np.nan, float)

    y_true = np.full((len(idxs), H, Dy), np.nan, float)
    mask = np.zeros((len(idxs), H, Dy), float)

    used = 0
    skipped = 0

    for row, i in enumerate(idxs):
        L = int(seq_lens[i])
        t0 = int(split[i])
        if t0 <= 0 or t0 >= L:
            skipped += 1
            continue
        H_i = min(H, L - t0)
        if H_i <= 0:
            skipped += 1
            continue

        fused = rollout_cf_future(
            transformer=transformer,
            sc_model=sc_model,
            subset=subset,
            i=i,
            t0=t0,
            H=H_i,
            layout=layout,
            device=device,
            reliability_models=reliability_models,
            R_min=R_min,
            process_noise_sc=process_noise_sc,
            prior_var_scale2=prior_var_scale2,
            sc_var_scale2=sc_var_scale2,
        )
        if fused is None:
            skipped += 1
            continue

        mu_T[row, :H_i, :] = fused["prior_mean"]
        var_T[row, :H_i, :] = fused["prior_var"]
        mu_SC[row, :H_i, :] = fused["sc_mean"]
        var_SC[row, :H_i, :] = fused["sc_var"]
        mu_F[row, :H_i, :] = fused["fused_mean"]
        var_F[row, :H_i, :] = fused["fused_var"]

        yt = Y_all[i, t0:t0 + H_i, :]
        y_true[row, :H_i, :] = yt

        _, _, mseq = ds[i]
        mseq = np.asarray(mseq, float).reshape(-1)
        mf = mseq[t0:t0 + H_i].reshape(-1, 1)
        mf = np.repeat(mf, Dy, axis=1)
        mf = mf * np.isfinite(yt).astype(float)
        mask[row, :H_i, :] = mf

        used += 1

    print(f"[collect] used={used} skipped={skipped} H={H} Dy={Dy}")
    return {
        "mu_T": mu_T, "var_T": np.maximum(var_T, 1e-8),
        "mu_SC": mu_SC, "var_SC": np.maximum(var_SC, 1e-8),
        "mu_F": mu_F, "var_F": np.maximum(var_F, 1e-8),
        "y": y_true, "mask": mask,
    }

# =============================================================================
# Metrics: RMSE (accuracy) + NLL (uncertainty)
# =============================================================================


# =============================================================================
# Synthetic dataset generation 
# =============================================================================

def generate_dataset_on_fly(seed: int, ood_mode: bool = False):
    """
    Generate semi-synthetic dataset on-the-fly using config.
    
    Args:
        seed: Random seed for reproducibility
        ood_mode: If True, generate OOD test data with shifted parameters
    Returns:
        dataset_collection: Generated MIMIC3SyntheticDatasetCollection
    """
    # Set seed
    seed_everything(seed)
    
    # Load config 
    config_dir = os.path.abspath('./config')
    
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        overrides = [
            '+dataset=mimic3_synthetic',
            "+backbone/ct_hparams/mimic3_synthetic='1000'",
            f'exp.seed={seed}',
            f'dataset.seed={seed}',
            f'dataset.data_seed={seed}',
        ]

        args = compose(
            config_name='config.yaml',
            overrides=overrides
        )
    
    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
    
    logger.info("=" * 80)
    logger.info(f"Generating Semi-Synthetic MIMIC-III Dataset ({'OOD' if ood_mode else 'ID'})")
    logger.info("=" * 80)
    logger.info(f"Seed: {seed}")
    
    # Generate dataset
    logger.info("Instantiating Dataset Collection...")
    
    dataset_collection = instantiate(args.dataset, _recursive_=True)
    
    if ood_mode:
        logger.info("Generating OOD test datasets with modified parameters...")
        
        # Create modified config for OOD dataset
        ood_config = OmegaConf.create(OmegaConf.to_container(args.dataset, resolve=True))
        
        for i, treatment in enumerate(ood_config.synth_treatments_list):
            orig_conf_outcome = treatment.conf_outcome_weight
            orig_conf_vars = treatment.conf_vars_weight
            orig_effect = treatment.full_effect
            
            treatment.conf_outcome_weight *= 1.4
            treatment.conf_vars_weight *= 1.4
            treatment.full_effect *= 1.2
            
            logger.info(f"  Treatment {i}: conf_outcome {orig_conf_outcome:.2f} -> {treatment.conf_outcome_weight:.2f}")
            logger.info(f"  Treatment {i}: conf_vars {orig_conf_vars:.2f} -> {treatment.conf_vars_weight:.2f}")
            logger.info(f"  Treatment {i}: full_effect {orig_effect:.2f} -> {treatment.full_effect:.2f}")
        
        # Instantiate OOD dataset collection 
        ood_dataset_collection = instantiate(ood_config, _recursive_=True)
        
        # Replace test datasets
        dataset_collection.test_cf_one_step = ood_dataset_collection.test_cf_one_step
        dataset_collection.test_cf_treatment_seq = ood_dataset_collection.test_cf_treatment_seq
    
    dataset_collection.process_data_multi()
    
    logger.info("Dataset generation complete!")
    
    return dataset_collection

# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()

    # data / splits
    p.add_argument("--seed", type=int, default=None, help="Seed for both data generation and training (overrides --data_seed and --train_seed).")
    p.add_argument("--data_seed", type=int, default=10, help="Seed used to generate the semi-synthetic dataset.")
    p.add_argument("--train_seed", type=int, default=10, help="Seed used for training/optimization randomness (init, dropout, minibatch order).")
    p.add_argument("--ood", action="store_true", help="Generate OOD test dataset with modified parameters")
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--eval_taus", type=str, default="all")
    p.add_argument("--calib_taus", type=str, default="all")
    p.add_argument("--max_eval_windows", type=int, default=None)
    p.add_argument("--max_calib_windows", type=int, default=800)

    # prior training
    p.add_argument("--prior_epochs", type=int, default=250)
    p.add_argument("--prior_batch_size", type=int, default=64)
    p.add_argument("--prior_lr", type=float, default=1e-3)

    # MMD balancing regularization (Transformer prior)
    p.add_argument("--lambda_mmd", type=float, default=0.01)
    p.add_argument("--mmd_min_per_group", type=int, default=8)
    p.add_argument("--mmd_max_per_group", type=int, default=256)
    p.add_argument("--mmd_bw_mode", type=str, default="median", choices=["median", "fixed"])
    p.add_argument("--mmd_bw", type=float, default=None)
    p.add_argument("--mmd_bw_scale", type=float, default=1.0)

    # transformer arch
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--ff_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=256)

    # SC mode
    p.add_argument("--sc_mode", type=str, default="attn", choices=["kernel", "attn"])

    # DTSC / shared SC params
    p.add_argument("--knn", type=int, default=50)
    p.add_argument("--max_donors_per_action", type=int, default=20000)
    p.add_argument("--max_steps_total", type=int, default=300000)
    p.add_argument("--tau_dist", type=float, default=0.25)

    # (1) stabilization knobs
    p.add_argument("--zstd_floor", type=float, default=5e-2)
    p.add_argument("--sc_debug", type=int, default=0)

    # kernel-only params
    p.add_argument("--kernel", type=str, default="rbf", choices=["rbf", "laplace"])
    p.add_argument("--kernel_bw_mode", type=str, default="median", choices=["median", "fixed"])
    p.add_argument("--kernel_bw", type=float, default=None)
    p.add_argument("--kernel_bw_scale", type=float, default=0.5)

    # PCA for SC
    p.add_argument("--static_pca_dim", type=int, default=16)
    p.add_argument("--static_pca_whiten", type=int, default=0)
    p.add_argument("--vitals_pca_dim", type=int, default=8)
    p.add_argument("--vitals_pca_whiten", type=int, default=0)
    p.add_argument("--lambda_vitals", type=float, default=0.5)
    p.add_argument("--max_vitals_pca_samples", type=int, default=20000)

    # target mode
    p.add_argument("--target_mode", type=str, default="delta", choices=["delta", "rel_delta", "level"])

    # attention-only params
    p.add_argument("--attn_emb_dim", type=int, default=64)
    p.add_argument("--attn_hidden", type=int, default=256)
    p.add_argument("--attn_dropout", type=float, default=0.1)
    p.add_argument("--attn_temp", type=float, default=0.2)
    p.add_argument("--attn_epochs", type=int, default=10)
    p.add_argument("--attn_lr", type=float, default=2e-3)
    p.add_argument("--attn_batch_size", type=int, default=256)
    p.add_argument("--attn_candidates", type=int, default=128)
    p.add_argument("--attn_steps_per_epoch", type=int, default=1500)
    p.add_argument("--attn_entropy_reg", type=float, default=0.0)
    p.add_argument("--attn_verbose_every", type=int, default=1)

    # reliability + fusion
    p.add_argument("--calib_epochs", type=int, default=250)
    p.add_argument("--calib_lr", type=float, default=5e-2)
    p.add_argument("--R_min", type=float, default=1e-4)
    p.add_argument("--process_noise_sc", type=float, default=1e-6)

    # plotting
    p.add_argument("--plot_examples", type=int, default=10)
    p.add_argument("--plot_dim", type=int, default=0)
    p.add_argument("--plot_dir", type=str, default="plots")
    p.add_argument("--plot_n_sigma", type=float, default=1.0)
    p.add_argument("--plot_ylim", type=str, default="")
    p.add_argument("--viz_split_time", type=int, default=20)
    
    # device
    p.add_argument("--device", type=str, default="auto", help="Device: 'cuda', 'cuda:0', 'cuda:1', 'cpu', or 'auto' (auto-detect)")
    
    # checkpoint and visualization
    p.add_argument("--viz", action="store_true", help="Load pretrained model and run visualization only (no training)")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save/load model checkpoints")

    args = p.parse_args()

    if args.seed is not None:
        data_seed = int(args.seed)
        train_seed = int(args.seed)
        logger.info(f"Using --seed={args.seed} for both data generation and training")
    else:
        data_seed = int(args.data_seed)
        train_seed = int(args.train_seed)
        logger.info(f"Using separate seeds: data_seed={data_seed}, train_seed={train_seed}")

    # Generate dataset on-the-fly
    logger.info(f"\nGenerating dataset on-the-fly with seed={data_seed}, ood={args.ood}")
    collection = generate_dataset_on_fly(seed=data_seed, ood_mode=args.ood)

    seed_everything(train_seed)

    need_eval = max_step_in_taus_str(args.eval_taus)
    need_calib = max_step_in_taus_str(args.calib_taus)
    horizon = max(int(args.horizon), int(need_eval), int(need_calib))
    if horizon != int(args.horizon):
        print(f"[cfg] bump horizon {args.horizon} -> {horizon} to cover taus")

    scaling_params = collection.train_scaling_params

    calib_subset = getattr(collection, "val_cf_treatment_seq", None)
    if calib_subset is None:
        calib_subset = collection.test_cf_treatment_seq
        print("[cfg] val_cf_treatment_seq missing; using test_cf_treatment_seq for calibration (not ideal).")

    train_ds = MIMIC3SyntheticTimeSeriesDataset(collection.train_f)
    layout = infer_layout_by_matching(collection.train_f)

    x0, y0, _ = train_ds[0]
    input_dim = int(np.asarray(x0).shape[-1])
    Dy = int(np.asarray(y0).shape[-1])

    A_dim = int(np.asarray(collection.train_f.data["current_treatments"]).shape[-1])
    D_static = int(np.asarray(collection.train_f.data["static_features"]).shape[-1]) if "static_features" in collection.train_f.data else 0
    D_vitals = int(np.asarray(collection.train_f.data["vitals"]).shape[-1]) if "vitals" in collection.train_f.data else 0

    slices = compute_token_slices(
        layout=layout,
        Dy=Dy,
        A_dim=A_dim,
        D_static=D_static,
        D_vitals=D_vitals,
        input_dim=input_dim,
    )
    treat_slice = slices.get("treatments", None)
    print(f"[token-slices] treatments={treat_slice} | A_dim={A_dim} | D_static={D_static} | D_vitals={D_vitals}")

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[device] {device} | input_dim={input_dim} | Dy={Dy} | A_dim={A_dim} | horizon={horizon}")
    
    # Checkpoint path
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, f"model_data{data_seed}_train{train_seed}.pt")

    eval_taus = parse_taus_arg(args.eval_taus, horizon=horizon)
    calib_taus = parse_taus_arg(args.calib_taus, horizon=horizon)
    print(f"[taus] eval(1-based)={[t+1 for t in eval_taus]} | calib(1-based)={[t+1 for t in calib_taus]}")

    transformer = GaussianCausalTransformer(
        input_dim=input_dim,
        dim_outcomes=Dy,
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        num_layers=int(args.n_layers),
        dim_feedforward=int(args.ff_dim),
        dropout=float(args.dropout),
        max_len=int(args.max_len),
    ).to(device)

    # ==========================================================
    # Load or Train Models
    # ==========================================================
    
    if args.viz:
        # VIZ: Load all models and calibration from checkpoint
        print(f"\n[VIZ] Loading checkpoint from {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Load transformer
        transformer.load_state_dict(checkpoint['transformer'])
        print("[VIZ] Transformer loaded successfully")
        
        # Load SC model
        sc_model = checkpoint['sc_model']
        print("[VIZ] SC model loaded successfully")
        
        # Load calibration results
        prior_var_scale2 = checkpoint['prior_var_scale2']
        sc_var_scale2 = checkpoint['sc_var_scale2']
        reliability_models = checkpoint['reliability_models']
        print("[VIZ] Calibration data loaded successfully")
        
    else:
        # TRAIN: Train all models from scratch
        print("\n[1] Train transformer prior")
        train_gaussian_prior(
            transformer,
            train_ds=train_ds,
            num_epochs=int(args.prior_epochs),
            batch_size=int(args.prior_batch_size),
            lr=float(args.prior_lr),
            device=device,
            lambda_mmd=float(args.lambda_mmd),
            treat_slice=treat_slice,
            A_dim=A_dim,
            mmd_min_per_group=int(args.mmd_min_per_group),
            mmd_max_per_group=int(args.mmd_max_per_group),
            mmd_bw_mode=str(args.mmd_bw_mode),
            mmd_bw=args.mmd_bw,
            mmd_bw_scale=float(args.mmd_bw_scale),
        )

        print(f"\n[2] Fit SC on train_f | mode={args.sc_mode}")
        common_kwargs = dict(
            m_hist=10,
            m_dhist=6,
            target_mode=str(args.target_mode),
            eps_rel=1e-3,
            max_donors_per_action=int(args.max_donors_per_action),
            knn=int(args.knn),
            max_steps_total=int(args.max_steps_total),
            seed=0,
            use_prev_treat=True,
            clip_q=0.01,
            tau_dist=float(args.tau_dist),
            zstd_floor=float(args.zstd_floor),
            sc_debug=bool(int(args.sc_debug)),
        )

        if str(args.sc_mode) == "kernel":
            sc_model = DTSC_MultiOutcome(
                Dy=Dy,
                static_pca_dim=int(args.static_pca_dim),
                static_pca_whiten=bool(int(args.static_pca_whiten)),
                vitals_pca_dim=int(args.vitals_pca_dim),
                vitals_pca_whiten=bool(int(args.vitals_pca_whiten)),
                lambda_vitals=float(args.lambda_vitals),
                max_vitals_pca_samples=int(args.max_vitals_pca_samples),
                kernel=str(args.kernel),
                kernel_bw_mode=str(args.kernel_bw_mode),
                kernel_bw=args.kernel_bw,
                kernel_bw_scale=float(args.kernel_bw_scale),
                **common_kwargs
            ).fit(collection.train_f)
        else:
            sc_model = AttentionSC_MultiOutcome(
                Dy=Dy,
                static_pca_dim=int(args.static_pca_dim),
                static_pca_whiten=bool(int(args.static_pca_whiten)),
                vitals_pca_dim=int(args.vitals_pca_dim),
                vitals_pca_whiten=bool(int(args.vitals_pca_whiten)),
                lambda_vitals=float(args.lambda_vitals),
                max_vitals_pca_samples=int(args.max_vitals_pca_samples),
                attn_device=device,
                attn_epochs=int(args.attn_epochs),
                attn_lr=float(args.attn_lr),
                attn_batch_size=int(args.attn_batch_size),
                attn_candidates=int(args.attn_candidates),
                attn_steps_per_epoch=int(args.attn_steps_per_epoch),
                attn_temp=float(args.attn_temp),
                attn_entropy_reg=float(args.attn_entropy_reg),
                attn_verbose_every=int(args.attn_verbose_every),
                attn_emb_dim=int(args.attn_emb_dim),
                attn_hidden=int(args.attn_hidden),
                attn_dropout=float(args.attn_dropout),
                **common_kwargs
            ).fit(collection.train_f, device_for_training=device)

        print("\n[2.5] Fit variance calibration scales on calibration subset (Fix NLL)")
        pack_cal = collect_gaussian_forecasts(
            transformer=transformer,
            sc_model=sc_model,
            subset=calib_subset,
            horizon=horizon,
            layout=layout,
            max_windows=int(args.max_calib_windows),
            device=device,
            reliability_models=None,
            R_min=float(args.R_min),
            process_noise_sc=float(args.process_noise_sc),
            prior_var_scale2=None,
            sc_var_scale2=None,
        )

        prior_var_scale2 = fit_var_scale2_per_tau(
            mu=pack_cal["mu_T"],
            var=pack_cal["var_T"],
            y=pack_cal["y"],
            mask=pack_cal["mask"],
            H=horizon,
            clip=(0.001, 1000.0),
            min_points=10,
        )
        sc_var_scale2 = fit_var_scale2_per_tau(
            mu=pack_cal["mu_SC"],
            var=pack_cal["var_SC"],
            y=pack_cal["y"],
            mask=pack_cal["mask"],
            H=horizon,
            clip=(0.001, 1000.0),
            min_points=10,
        )
        print(
            f"[var-calib] prior_scale2 min/med/max="
            f"{prior_var_scale2.min():.3f}/{np.median(prior_var_scale2):.3f}/{prior_var_scale2.max():.3f}"
        )
        print(
            f"[var-calib] sc_scale2    min/med/max="
            f"{sc_var_scale2.min():.3f}/{np.median(sc_var_scale2):.3f}/{sc_var_scale2.max():.3f}"
        )

        print("\n[3] Fit reliability models r(phi) on calibration subset (rollout-consistent history)")
        reliability_models = fit_reliability_models(
            sc_model=sc_model,
            subset=calib_subset,
            horizon=horizon,
            taus=calib_taus,
            max_windows=int(args.max_calib_windows),
            device=device,
            lr=float(args.calib_lr),
            epochs=int(args.calib_epochs),
            batch_size=512,
            R_min=float(args.R_min),
            eps=1e-3,
            process_noise=float(args.process_noise_sc),
            sc_var_scale2=sc_var_scale2,
        )
        
        # Save checkpoint
        print(f"\n[CHECKPOINT] Saving models to {checkpoint_path}")
        torch.save({
            'transformer': transformer.state_dict(),
            'sc_model': sc_model,
            'prior_var_scale2': prior_var_scale2,
            'sc_var_scale2': sc_var_scale2,
            'reliability_models': reliability_models,
            'args': vars(args),
            'data_seed': data_seed,
            'train_seed': train_seed,
        }, checkpoint_path)
        print(f"[CHECKPOINT] Models saved successfully")

    # ==========================================================
    # Evaluation and Visualization
    # ==========================================================
    
    # Evaluate multi-step counterfactual (test_cf_treatment_seq)
    if not args.viz:
        print("\n[4] Evaluate on test_cf_treatment_seq: multi-step RMSE and NLL")
        results = compute_n_step_rmse_nll(
            transformer=transformer,
            sc_model=sc_model,
            dataset=collection.test_cf_treatment_seq,
            scaling_params=scaling_params,
            horizon=horizon,
            layout=layout,
            device=device,
            eval_taus=eval_taus,
            reliability_models=reliability_models,
            R_min=float(args.R_min),
            process_noise_sc=float(args.process_noise_sc),
            max_eval_windows=int(args.max_eval_windows) if args.max_eval_windows is not None else None,
            unscale=False,
            rmse_mode="flatten",
            prior_var_scale2=prior_var_scale2,
            sc_var_scale2=sc_var_scale2,
        )

        print("\nCreating visualizations")
        
        # ============================== Top-K RMSE ==============================
        print("\n" + "=" * 80)
        print("Top-K RMSE (worst-performing trajectories)")
        print("=" * 80)
    
        if 'per_trajectory_rmse_prior' in results and 'per_trajectory_rmse_sc' in results and 'per_trajectory_rmse_fused' in results:
            from evaluation import compute_top_k_rmse
            
            k_values = [1, 5, 10, 20, 50, 100]
            
            # PRIOR model
            top_k_prior = compute_top_k_rmse(results['per_trajectory_rmse_prior'], k_values)
            print("\n--- PRIOR (Transformer) ---")
            for k, rmse in top_k_prior['top_k'].items():
                print(f"  Top-{k:3d} RMSE: {rmse:.6f}")
            print("-" * 80)
            print(f"  Overall Mean RMSE: {top_k_prior['mean']:.6f}")
            print(f"  Overall Std RMSE:  {top_k_prior['std']:.6f}")
            print(f"  Total trajectories: {top_k_prior['count']}")
            
            # EVIDENCE (SC) model
            top_k_sc = compute_top_k_rmse(results['per_trajectory_rmse_sc'], k_values)
            print("\n--- EVIDENCE (Dynamic SC) ---")
            for k, rmse in top_k_sc['top_k'].items():
                print(f"  Top-{k:3d} RMSE: {rmse:.6f}")
            print("-" * 80)
            print(f"  Overall Mean RMSE: {top_k_sc['mean']:.6f}")
            print(f"  Overall Std RMSE:  {top_k_sc['std']:.6f}")
            print(f"  Total trajectories: {top_k_sc['count']}")
            
            # FUSED model
            top_k_fused = compute_top_k_rmse(results['per_trajectory_rmse_fused'], k_values)
            print("\n--- FUSED (CI) ---")
            for k, rmse in top_k_fused['top_k'].items():
                print(f"  Top-{k:3d} RMSE: {rmse:.6f}")
            print("-" * 80)
            print(f"  Overall Mean RMSE: {top_k_fused['mean']:.6f}")
            print(f"  Overall Std RMSE:  {top_k_fused['std']:.6f}")
            print(f"  Total trajectories: {top_k_fused['count']}")
            print("=" * 80)
    
    # Visualization 
    print("\n[5] Generating visualizations")
    
    print("\nPlot Kalman fusion examples (test_cf_treatment_seq)")
    ylim = None
    if isinstance(args.plot_ylim, str) and args.plot_ylim.strip() and "," in args.plot_ylim:
        parts = args.plot_ylim.split(",", 1)
        if len(parts) == 2:
            ylim = (float(parts[0].strip()), float(parts[1].strip()))

    vis_kalman_fusion(
        transformer=transformer,
        sc_model=sc_model,
        cf_subset=collection.test_cf_treatment_seq,
        scaling_params=scaling_params,
        horizon=horizon,
        layout=layout,
        idx_list=list(range(int(args.plot_examples))),
        device=device,
        reliability_models=reliability_models,
        R_min=float(args.R_min),
        process_noise_sc=float(args.process_noise_sc),
        n_sigma=float(args.plot_n_sigma),
        out_dim=int(args.plot_dim),
        save_dir=os.path.join(str(args.plot_dir), "fusion"),
        ylim=ylim,
        prior_var_scale2=prior_var_scale2,
        sc_var_scale2=sc_var_scale2,
    )
    print(f"fusion plots saved under ./{args.plot_dir}/fusion/")
    
    print("\nPlot combined patient visualization (test_cf_treatment_seq)")
    vis_patients(
        dataset=collection.test_cf_treatment_seq,
        transformer=transformer,
        sc_model=sc_model,
        layout=layout,
        scaling_params=scaling_params,
        horizon=horizon,
        device=device,
        patient_ids=[6234, 8282, 14561, 14685, 20598, 21634],
        cf_ids=[1, 9, 1, 5, 1, 2],
        reliability_models=reliability_models,
        R_min=float(args.R_min),
        process_noise_sc=float(args.process_noise_sc),
        prior_var_scale2=prior_var_scale2,
        sc_var_scale2=sc_var_scale2,
        n_sigma=1.0,
        output_dir=os.path.join(str(args.plot_dir), "patients"),
        num_patients=6,
    )
    print(f"Combined patient plots saved under ./{args.plot_dir}/patients/")

if __name__ == "__main__":
    main()
