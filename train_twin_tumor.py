#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
from pathlib import Path
import math
import random
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Project imports (adjust if your repo structure differs)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cancer_dataset import SyntheticCancerDatasetCollection  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tumor_with_transformer_prior_dtsc_eval")

# External evaluation helpers (use existing repo functions)
from evaluation import (  # noqa: E402
    _prep_out_scaling,
    unscale_gaussian,
    unscale_y,
    masked_rmse_per_tau,
    masked_gaussian_nll_per_tau_avg_dims,
)


# =============================================================================
# Small wrappers for eval helpers (handles signature differences)
# =============================================================================

def _call_masked_rmse_per_tau(mu: np.ndarray, y: np.ndarray, mask: np.ndarray, H: int) -> np.ndarray:
    try:
        return masked_rmse_per_tau(mu=mu, y=y, mask=mask, H=H)
    except TypeError:
        return masked_rmse_per_tau(mu=mu, y=y, mask=mask)


def _call_masked_gaussian_nll_per_tau(mu: np.ndarray, var: np.ndarray, y: np.ndarray, mask: np.ndarray,
                                      include_const: bool, H: int) -> np.ndarray:
    try:
        return masked_gaussian_nll_per_tau_avg_dims(mu=mu, var=var, y=y, mask=mask,
                                                    include_const=include_const, H=H)
    except TypeError:
        return masked_gaussian_nll_per_tau_avg_dims(mu=mu, var=var, y=y, mask=mask,
                                                    include_const=include_const)


# =============================================================================
# Reproducibility
# =============================================================================

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Hydra dataset generation
# =============================================================================

def generate_cancer_dataset(seed: int, coeff: float = 0.0):
    seed_everything(seed)

    config_dir = os.path.abspath("./config")
    if not os.path.isdir(config_dir):
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    OmegaConf.set_struct(OmegaConf.create({}), False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        overrides = [
            "+dataset=cancer_sim",
            f"dataset.coeff={coeff}",
            f"exp.seed={seed}",
        ]
        cfg = compose(config_name="config", overrides=overrides)

    OmegaConf.set_struct(cfg, False)

    logger.info("=" * 80)
    logger.info("Generating Cancer Simulation Dataset")
    logger.info(f"Seed: {seed} | Confounding coeff: {coeff}")
    logger.info("=" * 80)

    dataset_collection = instantiate(cfg.dataset, _recursive_=True)
    return dataset_collection


def _get_subset(dataset_collection, split_name: str):
    if hasattr(dataset_collection, split_name):
        return getattr(dataset_collection, split_name)

    fallbacks = {
        "train": "train_f",
        "val": "val_f",
        "test": "test_f",
        "test_cf": "test_cf_treatment_seq",
    }
    if split_name in fallbacks and hasattr(dataset_collection, fallbacks[split_name]):
        return getattr(dataset_collection, fallbacks[split_name])

    available = [k for k in dir(dataset_collection) if k.endswith("_f") or "test" in k]
    raise AttributeError(
        f"Unknown split '{split_name}'. "
        f"Try one of: train_f, val_f, test_f, test_cf_treatment_seq (or train/val/test/test_cf). "
        f"Available similar attrs: {available[:40]}"
    )


# =============================================================================
# Tokenization for tumor data (fixed layout)
# =============================================================================

def _infer_time_varying_key(subset, user_key: str = ""):
    d = subset.data
    if user_key:
        if user_key in d:
            arr = np.asarray(d[user_key])
            if arr.ndim == 3:
                return user_key
        logger.warning(f"--tv_key '{user_key}' not found as a 3D array in subset.data; ignoring.")

    banned = {"outputs", "current_treatments"}
    for k, v in d.items():
        if k in banned:
            continue
        try:
            arr = np.asarray(v)
        except Exception:
            continue
        if arr.ndim == 3 and arr.shape[0] == np.asarray(d["outputs"]).shape[0]:
            return k
    return None


def build_token_from_parts(
    y_prev, a_onehot, x_static, tv_t,
    Dy: int, A_dim: int, D_static: int, D_tv: int
) -> np.ndarray:
    parts = []
    parts.append(np.asarray(y_prev, dtype=np.float32).reshape(-1))
    parts.append(np.asarray(a_onehot, dtype=np.float32).reshape(-1))
    if D_static > 0:
        parts.append(np.asarray(x_static, dtype=np.float32).reshape(-1))
    if D_tv > 0:
        parts.append(np.asarray(tv_t, dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0).astype(np.float32)


class CancerTokenDataset(Dataset):
    """
    Teacher-forced tokens for transformer training:
      x_t uses true y_{t-1} (zeros at t=0) + a_t + static + optional tv_t
    """
    def __init__(self, subset, tv_key: str = ""):
        super().__init__()
        d = subset.data
        self.Y = np.asarray(d["outputs"]).astype(np.float32)  # (N,T,Dy)

        if "current_treatments" not in d:
            raise KeyError("subset.data must contain 'current_treatments' for transformer tokens.")
        self.A = np.asarray(d["current_treatments"]).astype(np.float32)  # (N,T,A_dim)

        self.L = np.asarray(d["sequence_lengths"]).astype(int)  # (N,)
        self.Xs = np.asarray(d["static_features"]).astype(np.float32) if "static_features" in d else None

        self.tv_key = _infer_time_varying_key(subset, user_key=str(tv_key))
        self.TV = np.asarray(d[self.tv_key]).astype(np.float32) if (self.tv_key is not None) else None

        self.N, self.T, self.Dy = self.Y.shape
        self.A_dim = int(self.A.shape[-1])

        self.D_static = int(self.Xs.shape[-1]) if (self.Xs is not None) else 0
        self.D_tv = int(self.TV.shape[-1]) if (self.TV is not None) else 0

        self.input_dim = int(self.Dy + self.A_dim + self.D_static + self.D_tv)
        self.treat_slice = (int(self.Dy), int(self.Dy + self.A_dim))

        logger.info(
            f"[CancerTokenDataset] N={self.N} T={self.T} Dy={self.Dy} A_dim={self.A_dim} "
            f"D_static={self.D_static} D_tv={self.D_tv} tv_key={self.tv_key} input_dim={self.input_dim}"
        )

    def __len__(self):
        return int(self.N)

    def __getitem__(self, idx: int):
        idx = int(idx)
        L = int(self.L[idx])
        L = max(1, min(L, self.T))

        y = self.Y[idx]  # (T,Dy)
        a = self.A[idx]  # (T,A_dim)
        x_static = self.Xs[idx] if (self.Xs is not None) else None
        tv = self.TV[idx] if (self.TV is not None) else None  # (T,D_tv)

        x_seq = np.zeros((self.T, self.input_dim), dtype=np.float32)
        y_seq = np.zeros((self.T, self.Dy), dtype=np.float32)
        m_seq = np.zeros((self.T,), dtype=np.float32)

        y_prev = np.zeros((self.Dy,), dtype=np.float32)
        for t in range(L):
            tv_t = tv[t] if (tv is not None) else None
            tok = build_token_from_parts(
                y_prev=y_prev,
                a_onehot=a[t],
                x_static=x_static,
                tv_t=(np.zeros((0,), dtype=np.float32) if tv_t is None else tv_t),
                Dy=self.Dy, A_dim=self.A_dim, D_static=self.D_static, D_tv=self.D_tv,
            )
            x_seq[t] = tok
            y_seq[t] = y[t]
            m_seq[t] = 1.0
            y_prev = y[t].astype(np.float32).reshape(-1)

        return x_seq, y_seq, m_seq


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
        B, T, _ = x.shape
        device = x.device
        if T > self.max_len:
            raise ValueError(f"T={T} exceeds max_len={self.max_len}. Increase --max_len.")
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
# MMD utilities
# =============================================================================

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


def train_gaussian_prior(
    model: nn.Module,
    train_ds: Dataset,
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

            m = mask.float().unsqueeze(-1)
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
                logger.info(
                    f"[prior] epoch {epoch:03d}/{num_epochs} | loss={loss_sum/max(nb,1):.4f} | "
                    f"NLL={nll_sum/max(nb,1):.4f} | MMD={mmd_sum/max(nb,1):.6f} | lambda_mmd={float(lambda_mmd):.3g}"
                )
            else:
                logger.info(f"[prior] epoch {epoch:03d}/{num_epochs} | train NLL={loss_sum/max(nb,1):.4f}")


# =============================================================================
# Plotting ground truth (unchanged)
# =============================================================================

def plot_groundtruth(
    subset,
    idxs,
    save_dir: str,
    out_dim: int = 0,
    plot_treatments: bool = True,
    title_prefix: str = "",
    ylim: tuple = None,
    show: bool = False,
):
    d = subset.data
    Y = np.asarray(d["outputs"]).astype(float)
    seq_lens = np.asarray(d["sequence_lengths"]).astype(int)

    A = None
    if plot_treatments and ("current_treatments" in d):
        A = np.asarray(d["current_treatments"]).astype(float)

    N, T, Dy = Y.shape
    if out_dim < 0 or out_dim >= Dy:
        raise ValueError(f"out_dim={out_dim} out of range for Dy={Dy}")

    save_dir = str(save_dir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for k, i in enumerate(idxs):
        L = int(seq_lens[i])
        L = max(1, min(L, T))

        y = Y[i, :L, out_dim]
        t = np.arange(L)

        if A is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(t, y)
            ax.set_xlabel("time")
            ax.set_ylabel(f"y[{out_dim}]")
            ax.set_title(f"{title_prefix}idx={i} (L={L})".strip())
            if ylim is not None:
                ax.set_ylim(*ylim)
            fig.tight_layout()
        else:
            A_i = A[i, :L, :]
            A_dim = A_i.shape[-1]

            fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
            ax0, ax1 = axes

            ax0.plot(t, y)
            ax0.set_ylabel(f"y[{out_dim}]")
            ax0.set_title(f"{title_prefix}idx={i} (L={L})".strip())
            if ylim is not None:
                ax0.set_ylim(*ylim)

            for a in range(A_dim):
                ax1.step(t, A_i[:, a], where="post", label=f"a[{a}]")
            ax1.set_xlabel("time")
            ax1.set_ylabel("treatment")
            ax1.legend(loc="upper right", fontsize=9)
            fig.tight_layout()

        out_path = os.path.join(save_dir, f"gt_{i:05d}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        if (k + 1) % 25 == 0:
            logger.info(f"Plotted {k+1}/{len(idxs)} trajectories...")

    logger.info(f"Saved {len(idxs)} ground-truth plots to: {save_dir}")

    if show:
        logger.info("show=True: interactive preview is not implemented in this script.")


# =============================================================================
# NEW: Plotting factual history + counterfactual truth + predictions
# =============================================================================

def _select_indices(N: int, n: int, do_random: bool, seed: int) -> List[int]:
    N = int(N)
    n = int(max(0, min(int(n), N)))
    if n <= 0:
        return []
    if not do_random:
        return list(range(n))
    rng = np.random.default_rng(int(seed))
    return rng.choice(np.arange(N), size=n, replace=False).astype(int).tolist()


def _get_factual_and_cf_outputs(subset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tries to retrieve factual and counterfactual outputs. If dataset does not provide separate keys,
    falls back to subset.data["outputs"] (assumes shared factual prefix and CF future).
    """
    d = subset.data
    y_main = np.asarray(d["outputs"]).astype(float)

    y_fact = None
    for k in ("outputs_factual", "factual_outputs", "y_factual", "outputs_f", "y_f"):
        if k in d:
            arr = np.asarray(d[k]).astype(float)
            if arr.shape == y_main.shape:
                y_fact = arr
                break
    if y_fact is None:
        y_fact = y_main

    y_cf = None
    for k in ("outputs_cf", "counterfactual_outputs", "y_cf", "outputs_counterfactual"):
        if k in d:
            arr = np.asarray(d[k]).astype(float)
            if arr.shape == y_main.shape:
                y_cf = arr
                break
    if y_cf is None:
        y_cf = y_main

    return y_fact, y_cf


@torch.no_grad()
def _prior_rollout_one(
    transformer: nn.Module,
    y_fact_i: np.ndarray,
    a_i: np.ndarray,
    xs_i: Optional[np.ndarray],
    tv_i: Optional[np.ndarray],
    t0: int,
    Hi: int,
    input_dim: int,
    Dy: int,
    A_dim: int,
    D_static: int,
    D_tv: int,
    device: str,
    logvar_clip: Tuple[float, float] = (-10.0, 8.0),
) -> Tuple[np.ndarray, np.ndarray]:
    T = int(y_fact_i.shape[0])
    x_tokens = np.zeros((T, input_dim), dtype=np.float32)

    # teacher-forced past up to t0-1
    y_prev = np.zeros((Dy,), dtype=np.float32)
    for t in range(t0):
        tv_t = tv_i[t] if (tv_i is not None) else None
        tok = build_token_from_parts(
            y_prev=y_prev, a_onehot=a_i[t], x_static=xs_i,
            tv_t=(np.zeros((0,), dtype=np.float32) if tv_t is None else tv_t),
            Dy=Dy, A_dim=A_dim, D_static=D_static, D_tv=D_tv
        )
        x_tokens[t] = tok
        y_prev = y_fact_i[t].astype(np.float32).reshape(-1)

    mu = np.full((Hi, Dy), np.nan, dtype=np.float32)
    var = np.full((Hi, Dy), np.nan, dtype=np.float32)

    transformer.eval().to(device)

    y_prev_pred = y_fact_i[t0 - 1].astype(np.float32).reshape(-1)
    for k in range(Hi):
        t = t0 + k
        tv_t = tv_i[t] if (tv_i is not None) else None
        tok = build_token_from_parts(
            y_prev=y_prev_pred, a_onehot=a_i[t], x_static=xs_i,
            tv_t=(np.zeros((0,), dtype=np.float32) if tv_t is None else tv_t),
            Dy=Dy, A_dim=A_dim, D_static=D_static, D_tv=D_tv
        )
        x_tokens[t] = tok

        x = torch.from_numpy(x_tokens).unsqueeze(0).to(device)
        mask_bt = torch.zeros((1, T), device=device, dtype=torch.float32)
        mask_bt[0, :t + 1] = 1.0
        pad_mask = mask_to_padding(mask_bt)

        mean, logvar = transformer(x, src_key_padding_mask=pad_mask)
        mean_t = mean[0, t].detach().cpu().numpy().astype(np.float32)
        logvar_t = torch.clamp(logvar[0, t], min=logvar_clip[0], max=logvar_clip[1]).detach().cpu().numpy()
        var_t = (np.exp(logvar_t) + 1e-8).astype(np.float32)

        mu[k] = mean_t
        var[k] = var_t
        y_prev_pred = mean_t.astype(np.float32).reshape(-1)

    return mu, var


def _sc_rollout_one(
    sc,
    y_fact_i: np.ndarray,
    a_i: np.ndarray,
    t0: int,
    Hi: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    mu = np.full((Hi, sc.Dy), np.nan, dtype=np.float32)
    var = np.full((Hi, sc.Dy), np.nan, dtype=np.float32)
    diags: List[Dict[str, float]] = []

    Hh0 = min(sc.hist_len, t0)
    y_hist = y_fact_i[t0 - Hh0:t0, :].copy()

    for k in range(Hi):
        t = t0 + k
        mu_t, var_t, diag = sc.predict_step(y_hist=y_hist, a_t=a_i[t], t=t)
        diags.append(diag)

        if np.any(~np.isfinite(mu_t)) or np.any(~np.isfinite(var_t)):
            # keep history unchanged if SC step is invalid
            continue

        mu[k] = mu_t
        var[k] = var_t

        y_hist = np.concatenate([y_hist, mu_t[None, :]], axis=0)
        if y_hist.shape[0] > sc.hist_len:
            y_hist = y_hist[-sc.hist_len:, :]

    return mu, var, diags


def _fused_rollout_one(
    transformer: nn.Module,
    sc,
    y_fact_i: np.ndarray,
    a_i: np.ndarray,
    xs_i: Optional[np.ndarray],
    tv_i: Optional[np.ndarray],
    t0: int,
    Hi: int,
    input_dim: int,
    Dy: int,
    A_dim: int,
    D_static: int,
    D_tv: int,
    device: str,
    fusion: str = "ci",
    ci_grid: int = 51,
    ci_objective: str = "logdet",
    logvar_clip: Tuple[float, float] = (-10.0, 8.0),
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    T = int(y_fact_i.shape[0])
    x_tokens = np.zeros((T, input_dim), dtype=np.float32)

    # teacher-forced past up to t0-1
    y_prev = np.zeros((Dy,), dtype=np.float32)
    for t in range(t0):
        tv_t = tv_i[t] if (tv_i is not None) else None
        tok = build_token_from_parts(
            y_prev=y_prev, a_onehot=a_i[t], x_static=xs_i,
            tv_t=(np.zeros((0,), dtype=np.float32) if tv_t is None else tv_t),
            Dy=Dy, A_dim=A_dim, D_static=D_static, D_tv=D_tv
        )
        x_tokens[t] = tok
        y_prev = y_fact_i[t].astype(np.float32).reshape(-1)

    # init SC history with truth up to t0-1
    Hh0 = min(sc.hist_len, t0)
    y_hist_sc = y_fact_i[t0 - Hh0:t0, :].copy()

    mu = np.full((Hi, Dy), np.nan, dtype=np.float32)
    var = np.full((Hi, Dy), np.nan, dtype=np.float32)
    diags: List[Dict[str, float]] = []

    transformer.eval().to(device)

    y_prev_pred = y_fact_i[t0 - 1].astype(np.float32).reshape(-1)
    for k in range(Hi):
        t = t0 + k

        # prior
        tv_t = tv_i[t] if (tv_i is not None) else None
        tok = build_token_from_parts(
            y_prev=y_prev_pred, a_onehot=a_i[t], x_static=xs_i,
            tv_t=(np.zeros((0,), dtype=np.float32) if tv_t is None else tv_t),
            Dy=Dy, A_dim=A_dim, D_static=D_static, D_tv=D_tv
        )
        x_tokens[t] = tok

        x = torch.from_numpy(x_tokens).unsqueeze(0).to(device)
        mask_bt = torch.zeros((1, T), device=device, dtype=torch.float32)
        mask_bt[0, :t + 1] = 1.0
        pad_mask = mask_to_padding(mask_bt)

        mean_p, logvar_p = transformer(x, src_key_padding_mask=pad_mask)
        mu_p = mean_p[0, t].detach().cpu().numpy().astype(np.float32)
        logvar_t = torch.clamp(logvar_p[0, t], min=logvar_clip[0], max=logvar_clip[1]).detach().cpu().numpy()
        var_p = (np.exp(logvar_t) + 1e-8).astype(np.float32)

        # sc
        mu_s, var_s, diag = sc.predict_step(y_hist=y_hist_sc, a_t=a_i[t], t=t)
        diags.append(diag)

        if np.any(~np.isfinite(mu_s)) or np.any(~np.isfinite(var_s)):
            mu_post, var_post = mu_p, var_p
        else:
            if fusion == "kalman":
                mu_post, var_post = fuse_diag_kalman(mu_p, var_p, mu_s, var_s)
            else:
                mu_post, var_post, _w = fuse_diag_ci(mu_p, var_p, mu_s, var_s, grid=int(ci_grid), objective=str(ci_objective))

        mu[k] = mu_post
        var[k] = var_post

        y_prev_pred = mu_post.astype(np.float32).reshape(-1)
        y_hist_sc = np.concatenate([y_hist_sc, mu_post[None, :]], axis=0)
        if y_hist_sc.shape[0] > sc.hist_len:
            y_hist_sc = y_hist_sc[-sc.hist_len:, :]

    return mu, var, diags


def plot_cf_predictions(
    subset,
    idxs: List[int],
    horizon: int,
    out_dim: int,
    save_dir: str,
    transformer: Optional[nn.Module],
    sc,
    tv_key: str,
    device: str,
    which: str = "fused",  # "prior"|"sc"|"fused"|"all"
    fusion: str = "ci",
    ci_grid: int = 51,
    ci_objective: str = "logdet",
    logvar_clip: Tuple[float, float] = (-10.0, 8.0),
    sigma: float = 1.96,
    plot_ci_band: bool = True,
    plot_treatments: bool = True,
    title_prefix: str = "",
    ylim: Optional[Tuple[float, float]] = None,
    show: bool = False,
    unscale: bool = False,
    scaling_params=None,
):
    d = subset.data
    y_fact_all, y_cf_all = _get_factual_and_cf_outputs(subset)

    A_all = np.asarray(d["current_treatments"]).astype(float)
    Ls = np.asarray(d["sequence_lengths"]).astype(int)
    Xs_all = np.asarray(d["static_features"]).astype(float) if "static_features" in d else None

    tv_name = _infer_time_varying_key(subset, user_key=str(tv_key))
    TV_all = np.asarray(d[tv_name]).astype(float) if (tv_name is not None) else None

    N, T, Dy = y_cf_all.shape
    A_dim = int(A_all.shape[-1])
    D_static = int(Xs_all.shape[-1]) if (Xs_all is not None) else 0
    D_tv = int(TV_all.shape[-1]) if (TV_all is not None) else 0
    input_dim = int(Dy + A_dim + D_static + D_tv)

    if out_dim < 0 or out_dim >= Dy:
        raise ValueError(f"out_dim={out_dim} out of range for Dy={Dy}")

    H = int(horizon)
    t0_all = _get_future_past_split_points(subset, horizon=H)

    save_dir = str(save_dir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    run_prior = (which in ("prior", "all", "fused"))
    run_sc = (which in ("sc", "all", "fused"))
    run_fused = (which in ("fused", "all"))

    if run_prior or run_fused:
        if transformer is None:
            raise ValueError("Plotting requires transformer prior for prior/fused. Train or load checkpoint.")

    for j, i in enumerate(idxs):
        i = int(i)
        L = int(Ls[i])
        t0 = int(t0_all[i])
        if t0 <= 0 or t0 >= L:
            continue

        Hi = min(H, L - t0)
        if Hi <= 0:
            continue

        y_fact_i = y_fact_all[i].astype(float)
        y_cf_i = y_cf_all[i].astype(float)
        a_i = A_all[i].astype(float)
        xs_i = Xs_all[i].astype(float) if Xs_all is not None else None
        tv_i = TV_all[i].astype(float) if TV_all is not None else None

        # Gather series to plot (scaled space first)
        pred = {}  # name -> (mu(Hi,Dy), var(Hi,Dy))
        if run_prior and (which != "fused"):
            mu_p, var_p = _prior_rollout_one(
                transformer=transformer,
                y_fact_i=y_fact_i,
                a_i=a_i,
                xs_i=xs_i,
                tv_i=tv_i,
                t0=t0,
                Hi=Hi,
                input_dim=input_dim,
                Dy=Dy,
                A_dim=A_dim,
                D_static=D_static,
                D_tv=D_tv,
                device=device,
                logvar_clip=logvar_clip,
            )
            pred["prior"] = (mu_p, var_p)

        if run_sc and (which != "fused"):
            mu_s, var_s, _ = _sc_rollout_one(sc=sc, y_fact_i=y_fact_i, a_i=a_i, t0=t0, Hi=Hi)
            pred["sc"] = (mu_s, var_s)

        if run_fused:
            mu_f, var_f, _ = _fused_rollout_one(
                transformer=transformer,
                sc=sc,
                y_fact_i=y_fact_i,
                a_i=a_i,
                xs_i=xs_i,
                tv_i=tv_i,
                t0=t0,
                Hi=Hi,
                input_dim=input_dim,
                Dy=Dy,
                A_dim=A_dim,
                D_static=D_static,
                D_tv=D_tv,
                device=device,
                fusion=fusion,
                ci_grid=ci_grid,
                ci_objective=ci_objective,
                logvar_clip=logvar_clip,
            )
            pred["fused"] = (mu_f, var_f)

            if which == "all":
                # also optionally plot prior/sc alongside fused
                mu_p, var_p = _prior_rollout_one(
                    transformer=transformer,
                    y_fact_i=y_fact_i,
                    a_i=a_i,
                    xs_i=xs_i,
                    tv_i=tv_i,
                    t0=t0,
                    Hi=Hi,
                    input_dim=input_dim,
                    Dy=Dy,
                    A_dim=A_dim,
                    D_static=D_static,
                    D_tv=D_tv,
                    device=device,
                    logvar_clip=logvar_clip,
                )
                pred.setdefault("prior", (mu_p, var_p))
                mu_s, var_s, _ = _sc_rollout_one(sc=sc, y_fact_i=y_fact_i, a_i=a_i, t0=t0, Hi=Hi)
                pred.setdefault("sc", (mu_s, var_s))

        # Unscale (including truth) if requested
        y_hist = y_fact_i[:t0, :]              # factual history (prefix)
        y_true_future = y_cf_i[t0:t0 + Hi, :]  # counterfactual ground-truth future

        if unscale:
            if scaling_params is None:
                raise ValueError("unscale=True but scaling_params is None.")
            y_hist_u = unscale_y(y_hist[None, ...], scaling_params)[0]
            y_true_u = unscale_y(y_true_future[None, ...], scaling_params)[0]

            pred_u = {}
            for name, (mu0, var0) in pred.items():
                mu_u, var_u = unscale_gaussian(mu0[None, ...], var0[None, ...], scaling_params)
                pred_u[name] = (mu_u[0], var_u[0])
            y_hist = y_hist_u
            y_true_future = y_true_u
            pred = pred_u

        # Build timeline
        t_hist = np.arange(t0)
        t_fut = np.arange(t0, t0 + Hi)

        # Plot
        if plot_treatments:
            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
            ax0, ax1 = axes
        else:
            fig, ax0 = plt.subplots(1, 1, figsize=(10, 4.5))
            ax1 = None

        # factual history
        ax0.plot(t_hist, y_hist[:, out_dim], label="factual history")

        # cf truth
        ax0.plot(t_fut, y_true_future[:, out_dim], label="counterfactual truth")

        # predictions
        for name, (mu0, var0) in pred.items():
            yhat = mu0[:, out_dim]
            ax_line = ax0.plot(t_fut, yhat, label=f"pred: {name}")[0]
            if plot_ci_band:
                c = ax_line.get_color()
                sd = np.sqrt(np.maximum(var0[:, out_dim], 0.0))
                lo = yhat - float(sigma) * sd
                hi = yhat + float(sigma) * sd
                ax0.fill_between(t_fut, lo, hi, alpha=0.18, color=c)

        ax0.axvline(x=t0 - 0.5, linestyle="--", linewidth=1.0)
        ax0.set_ylabel(f"y[{out_dim}]")
        ax0.set_title(f"{title_prefix}idx={i}  (t0={t0}, Hi={Hi}, L={L})".strip())
        if ylim is not None:
            ax0.set_ylim(*ylim)
        ax0.legend(loc="best", fontsize=9)

        if ax1 is not None:
            A_i = a_i[:t0 + Hi, :]
            for a in range(A_dim):
                ax1.step(np.arange(t0 + Hi), A_i[:, a], where="post", label=f"a[{a}]")
            ax1.axvline(x=t0 - 0.5, linestyle="--", linewidth=1.0)
            ax1.set_xlabel("time")
            ax1.set_ylabel("treatment")
            ax1.legend(loc="upper right", fontsize=8, ncol=min(3, A_dim))

        fig.tight_layout()
        out_path = os.path.join(save_dir, f"cf_pred_{which}_idx_{i:05d}.png")
        fig.savefig(out_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)

        if (j + 1) % 25 == 0:
            logger.info(f"[plot_pred] Plotted {j+1}/{len(idxs)} examples...")

    logger.info(f"[plot_pred] Saved {len(idxs)} prediction plots to: {save_dir}")


# =============================================================================
# Evaluation parsing + split points
# =============================================================================

def parse_taus_arg(s: str, horizon: int) -> List[int]:
    s = str(s).strip().lower()
    if s in ("all", "*", ":", ""):
        return list(range(1, horizon + 1))

    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a.strip())
            b = int(b.strip())
            lo, hi = (a, b) if a <= b else (b, a)
            for t in range(lo, hi + 1):
                out.add(t)
        else:
            out.add(int(part))

    out = [t for t in sorted(out) if 1 <= t <= horizon]
    if not out:
        return list(range(1, horizon + 1))
    return out


def _get_future_past_split_points(subset, horizon: int) -> np.ndarray:
    d = subset.data
    N = int(np.asarray(d["outputs"]).shape[0])
    Ls = np.asarray(d["sequence_lengths"]).astype(int)

    for key in ("future_past_split", "future_past_splits", "t0", "split", "split_points"):
        if key in d:
            arr = np.asarray(d[key]).astype(int).reshape(-1)
            if arr.shape[0] == N:
                return arr

    t0 = np.zeros((N,), dtype=int)
    for i in range(N):
        L = int(Ls[i])
        t0[i] = max(1, min(L - 1, L - int(horizon)))
    return t0


# =============================================================================
# Dynamic Synthetic Control (DTSC-style): kernel KNN on history, predict delta under current action
# =============================================================================

class DynamicSyntheticControlKernel:
    """
    DTSC-style estimator:
      - donor retrieval by action match at current time t
      - KNN using standardized history features (last hist_len steps)
      - predict delta: y_t = y_{t-1} + sum w_j * (y_j,t - y_j,t-1)
      - variance: weighted var of donor deltas (per outcome dim) + var_floor
      - Fix B (reliability inflation):
          r = r_ess * r_w
          var_eff = var / (r + eps) + R_min
        optional hard-drop when support is extremely poor (drop_bad_support=1)
    """

    def __init__(
        self,
        donor_Y: np.ndarray,          # (Nd,T,Dy)
        donor_A: np.ndarray,          # (Nd,T,A_dim) onehot
        donor_L: np.ndarray,          # (Nd,)
        hist_len: int = 5,
        k: int = 50,
        bw: float = 1.0,
        z_std_floor: float = 1e-3,
        var_floor: float = 1e-4,
        min_donors: int = 5,
        use_delta_feat: int = 1,

        # ---- Fix B knobs ----
        use_reliability: int = 1,
        ess_ref: float = -1.0,              # if <=0, defaults to 0.5*k_eff
        rel_eps: float = 1e-6,
        R_min: float = 1e-3,                # additive floor after inflation (scaled space)
        drop_bad_support: int = 0,          # if 1: return NaN when very poor support
        drop_ess_min: float = 5.0,
        drop_wmax_max: float = 0.9,
        r_min_clip: float = 0.0,            # optional clamp r>=r_min_clip
    ):
        self.Y = np.asarray(donor_Y, dtype=np.float32)
        self.A = np.asarray(donor_A, dtype=np.float32)
        self.L = np.asarray(donor_L, dtype=int)
        self.Nd, self.T, self.Dy = self.Y.shape
        self.A_dim = int(self.A.shape[-1])

        self.hist_len = int(max(1, hist_len))
        self.k = int(max(1, k))
        self.bw = float(max(1e-6, bw))
        self.z_std_floor = float(max(1e-12, z_std_floor))
        self.var_floor = float(max(1e-12, var_floor))
        self.min_donors = int(max(1, min_donors))
        self.use_delta_feat = int(use_delta_feat)

        # Fix B
        self.use_reliability = bool(int(use_reliability))
        self.ess_ref = float(ess_ref)
        self.rel_eps = float(max(1e-12, rel_eps))
        self.R_min = float(max(0.0, R_min))
        self.drop_bad_support = bool(int(drop_bad_support))
        self.drop_ess_min = float(max(0.0, drop_ess_min))
        self.drop_wmax_max = float(min(max(0.0, drop_wmax_max), 1.0))
        self.r_min_clip = float(min(max(0.0, r_min_clip), 1.0))

        # Precompute per-action donor index lists (kept simple)
        self._donor_by_action = [np.arange(self.Nd, dtype=int) for _ in range(self.A_dim)]

    def _history_feature(self, y_hist: np.ndarray) -> np.ndarray:
        """
        y_hist: (Hh, Dy) last Hh steps (Hh<=hist_len) in chronological order.
        feature: concat(y_hist) and optionally concat(deltas).
        """
        y_hist = np.asarray(y_hist, dtype=np.float32)
        feat = [y_hist.reshape(-1)]
        if self.use_delta_feat and y_hist.shape[0] >= 2:
            dy = (y_hist[1:] - y_hist[:-1]).reshape(-1)
            feat.append(dy)
        return np.concatenate(feat, axis=0).astype(np.float32)

    def _standardize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        sd = np.maximum(sd, self.z_std_floor)
        return (X - mu) / sd, mu, sd

    @staticmethod
    def _clip01(x: float) -> float:
        return float(min(max(float(x), 0.0), 1.0))

    def _reliability_from_support(self, ess: float, wmax: float, k_eff: int) -> float:
        """
        Conservative heuristic reliability in [0,1]:
          r_ess = clip( ess / ess_ref, 0, 1 )
          r_w   = clip( (1 - wmax) / (1 - 1/k_eff), 0, 1 )
          r = r_ess * r_w
        """
        k_eff = max(int(k_eff), 1)
        if self.ess_ref > 0.0:
            ess_ref = float(self.ess_ref)
        else:
            ess_ref = 0.5 * float(k_eff)

        r_ess = self._clip01(float(ess) / (ess_ref + 1e-12))
        denom = (1.0 - 1.0 / float(k_eff)) + 1e-12
        r_w = self._clip01((1.0 - float(wmax)) / denom)

        r = float(r_ess * r_w)
        if self.r_min_clip > 0.0:
            r = max(r, float(self.r_min_clip))
        return float(min(max(r, 0.0), 1.0))

    def predict_step(
        self,
        y_hist: np.ndarray,     # (<=hist_len, Dy), includes y_{t-1} as last row
        a_t: np.ndarray,        # (A_dim,) onehot at time t
        t: int,                 # time index in donor sequences
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Returns:
          mu_t (Dy,), var_t (Dy,), diagnostics dict
        """
        t = int(t)
        if t <= 0 or t >= self.T:
            return (np.full((self.Dy,), np.nan, np.float32),
                    np.full((self.Dy,), np.nan, np.float32),
                    {"ess": 0.0, "entropy": 0.0, "w_max": 0.0, "min_dist": float("inf"),
                     "n_donors": 0.0, "r": 0.0})

        a_idx = int(np.argmax(a_t)) if a_t is not None else 0

        # Candidate donors: match action at time t
        idx_all = self._donor_by_action[a_idx]
        a_d = np.argmax(self.A[idx_all, t, :], axis=-1)
        idx_all = idx_all[a_d == a_idx]
        if idx_all.size == 0:
            return (np.full((self.Dy,), np.nan, np.float32),
                    np.full((self.Dy,), np.nan, np.float32),
                    {"ess": 0.0, "entropy": 0.0, "w_max": 0.0, "min_dist": float("inf"),
                     "n_donors": 0.0, "r": 0.0})

        # validity by length
        ok = self.L[idx_all] > t
        idx_all = idx_all[ok]
        if idx_all.size < self.min_donors:
            return (np.full((self.Dy,), np.nan, np.float32),
                    np.full((self.Dy,), np.nan, np.float32),
                    {"ess": float(idx_all.size), "entropy": 0.0, "w_max": 0.0, "min_dist": float("inf"),
                     "n_donors": float(idx_all.size), "r": 0.0})

        # Build donor history features using last hist_len steps ending at t-1
        Hh = min(self.hist_len, t)
        donor_feats = []
        valid_idx = []
        for j in idx_all.tolist():
            yj = self.Y[j]
            y_hist_j = yj[t - Hh:t, :]  # ends at t-1
            if np.any(~np.isfinite(y_hist_j)):
                continue
            donor_feats.append(self._history_feature(y_hist_j))
            valid_idx.append(j)
        if len(valid_idx) < self.min_donors:
            return (np.full((self.Dy,), np.nan, np.float32),
                    np.full((self.Dy,), np.nan, np.float32),
                    {"ess": float(len(valid_idx)), "entropy": 0.0, "w_max": 0.0, "min_dist": float("inf"),
                     "n_donors": float(len(valid_idx)), "r": 0.0})

        X = np.stack(donor_feats, axis=0)  # (M, Df)
        Xs, mu_z, sd_z = self._standardize(X)

        # Query feature from y_hist (ends at t-1)
        y_hist = np.asarray(y_hist, dtype=np.float32)
        if y_hist.ndim != 2 or y_hist.shape[1] != self.Dy:
            raise ValueError("y_hist must be (Hh, Dy)")
        if y_hist.shape[0] >= Hh:
            yq = y_hist[-Hh:, :]
        else:
            pad = np.repeat(y_hist[:1], repeats=(Hh - y_hist.shape[0]), axis=0)
            yq = np.concatenate([pad, y_hist], axis=0)

        q = self._history_feature(yq)[None, :]  # (1, Df)
        q = (q - mu_z) / sd_z

        # KNN by Euclidean distance in standardized space
        d2 = np.sum((Xs - q) ** 2, axis=1)  # (M,)
        order = np.argsort(d2)
        kk = min(self.k, order.size)
        sel = order[:kk]
        sel_d2 = d2[sel]
        sel_idx = np.asarray(valid_idx, dtype=int)[sel]

        # kernel weights
        w = np.exp(-sel_d2 / (2.0 * (self.bw ** 2) + 1e-12))
        w = w / (np.sum(w) + 1e-12)

        # donor delta at time t
        Yt = self.Y[sel_idx, t, :]     # (kk, Dy)
        Ytm1 = self.Y[sel_idx, t - 1, :]
        dY = Yt - Ytm1                 # (kk, Dy)

        # predict mean
        y_prev = yq[-1, :]
        mu_t = y_prev + (w[:, None] * dY).sum(axis=0)

        # raw var: weighted var of delta + var_floor
        dY_mean = (w[:, None] * dY).sum(axis=0, keepdims=True)
        dY_center = dY - dY_mean
        var_t = (w[:, None] * (dY_center ** 2)).sum(axis=0) + self.var_floor

        # diagnostics
        ess = float(1.0 / (np.sum(w ** 2) + 1e-12))
        ent = float(-np.sum(w * np.log(w + 1e-12)))
        w_max = float(np.max(w))
        min_dist = float(np.sqrt(np.min(sel_d2)) if sel_d2.size > 0 else float("inf"))

        # Fix B: reliability inflation / optional drop
        r = 1.0
        if self.use_reliability:
            r = self._reliability_from_support(ess=ess, wmax=w_max, k_eff=int(len(w)))

            if self.drop_bad_support and ((ess < self.drop_ess_min) or (w_max > self.drop_wmax_max)):
                # hard-ignore this SC step
                return (np.full((self.Dy,), np.nan, np.float32),
                        np.full((self.Dy,), np.nan, np.float32),
                        {"ess": ess, "entropy": ent, "w_max": w_max, "min_dist": min_dist,
                         "n_donors": float(len(valid_idx)), "r": float(r)})

            # inflate (downweight) SC by increasing variance when support is low
            var_t = var_t / (float(r) + self.rel_eps) + self.R_min

        diag = {"ess": ess, "entropy": ent, "w_max": w_max, "min_dist": min_dist,
                "n_donors": float(len(valid_idx)), "r": float(r)}

        return mu_t.astype(np.float32), var_t.astype(np.float32), diag


# =============================================================================
# Fusion: diagonal Kalman update or diagonal Covariance Intersection (CI)
# =============================================================================

def fuse_diag_kalman(mu_p: np.ndarray, var_p: np.ndarray, mu_m: np.ndarray, var_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eps = 1e-12
    var_p = np.maximum(var_p, 1e-8)
    var_m = np.maximum(var_m, 1e-8)
    prec = (1.0 / (var_p + eps)) + (1.0 / (var_m + eps))
    post_var = 1.0 / np.maximum(prec, eps)
    post_mu = post_var * ((mu_p / (var_p + eps)) + (mu_m / (var_m + eps)))
    return post_mu.astype(np.float32), post_var.astype(np.float32)


def fuse_diag_ci(mu_p: np.ndarray, var_p: np.ndarray, mu_m: np.ndarray, var_m: np.ndarray,
                 grid: int = 51, objective: str = "logdet") -> Tuple[np.ndarray, np.ndarray, float]:
    eps = 1e-12
    var_p = np.maximum(var_p, 1e-8)
    var_m = np.maximum(var_m, 1e-8)

    ws = np.linspace(0.0, 1.0, int(max(2, grid)), dtype=np.float64)
    best_w = 0.5
    best_obj = float("inf")

    inv_p = 1.0 / (var_p.astype(np.float64) + eps)
    inv_m = 1.0 / (var_m.astype(np.float64) + eps)

    for w in ws:
        prec = w * inv_p + (1.0 - w) * inv_m
        var_f = 1.0 / np.maximum(prec, eps)
        if objective == "trace":
            obj = float(np.sum(var_f))
        else:
            obj = float(np.sum(np.log(var_f + eps)))
        if obj < best_obj:
            best_obj = obj
            best_w = float(w)

    w = best_w
    prec = w * inv_p + (1.0 - w) * inv_m
    var_f = 1.0 / np.maximum(prec, eps)
    mu_f = var_f * (w * (mu_p.astype(np.float64) * inv_p) + (1.0 - w) * (mu_m.astype(np.float64) * inv_m))
    return mu_f.astype(np.float32), var_f.astype(np.float32), best_w


# =============================================================================
# PRIOR-only evaluation collection (teacher-forced or rollout)
# =============================================================================

@torch.no_grad()
def collect_prior_teacher_forced(
    transformer: nn.Module,
    subset,
    tv_key: str,
    horizon: int,
    device: str,
    logvar_clip: Tuple[float, float] = (-10.0, 8.0),
    max_eval_windows: Optional[int] = None,
    include_const_in_nll: bool = False,
    unscale: bool = False,
    scaling_params=None,
) -> Dict[str, np.ndarray]:
    ds = CancerTokenDataset(subset, tv_key=str(tv_key))
    d = subset.data
    Y_all = np.asarray(d["outputs"]).astype(float)
    Ls = np.asarray(d["sequence_lengths"]).astype(int)
    N, T, Dy = Y_all.shape
    H = int(horizon)

    idxs = np.arange(N)
    if max_eval_windows is not None and N > int(max_eval_windows):
        rng = np.random.default_rng(0)
        idxs = rng.choice(idxs, size=int(max_eval_windows), replace=False)
    idxs = np.asarray(idxs, dtype=int)
    Nw = int(len(idxs))

    mu = np.full((Nw, H, Dy), np.nan, float)
    var = np.full((Nw, H, Dy), np.nan, float)
    y = np.full((Nw, H, Dy), np.nan, float)
    m = np.zeros((Nw, H, Dy), float)

    t0_all = _get_future_past_split_points(subset, horizon=H)
    transformer.eval().to(device)

    for r, i in enumerate(idxs.tolist()):
        x_seq, _, mask_seq = ds[int(i)]
        L = int(Ls[int(i)])
        t0 = int(t0_all[int(i)])
        if t0 <= 0 or t0 >= L:
            continue
        Hi = min(H, L - t0)
        if Hi <= 0:
            continue

        x = torch.from_numpy(np.asarray(x_seq, np.float32)).unsqueeze(0).to(device)
        mask_bt = torch.from_numpy(np.asarray(mask_seq, np.float32)).unsqueeze(0).to(device)
        pad_mask = mask_to_padding(mask_bt)

        mean, logvar = transformer(x, src_key_padding_mask=pad_mask)
        logvar = torch.clamp(logvar, min=logvar_clip[0], max=logvar_clip[1])
        v = torch.exp(logvar).cpu().numpy()[0] + 1e-8
        mu_full = mean.cpu().numpy()[0]

        mu[r, :Hi, :] = mu_full[t0:t0 + Hi, :]
        var[r, :Hi, :] = v[t0:t0 + Hi, :]
        yt = Y_all[int(i), t0:t0 + Hi, :]
        y[r, :Hi, :] = yt

        mf = np.asarray(mask_seq, float).reshape(-1)[t0:t0 + Hi].reshape(-1, 1)
        mf = np.repeat(mf, Dy, axis=1) * np.isfinite(yt).astype(float)
        m[r, :Hi, :] = mf

    if unscale:
        mu_u, var_u = unscale_gaussian(mu, var, scaling_params)
        y_u = unscale_y(y, scaling_params)
        mu, var, y = mu_u, var_u, y_u

    rmse = _call_masked_rmse_per_tau(mu=mu, y=y, mask=m, H=H)
    nll = _call_masked_gaussian_nll_per_tau(mu=mu, var=var, y=y, mask=m, include_const=include_const_in_nll, H=H)
    return {"mu": mu, "var": var, "y": y, "mask": m, "rmse": rmse, "nll": nll}


@torch.no_grad()
def collect_prior_rollout(
    transformer: nn.Module,
    subset,
    tv_key: str,
    horizon: int,
    device: str,
    logvar_clip: Tuple[float, float] = (-10.0, 8.0),
    max_eval_windows: Optional[int] = None,
    include_const_in_nll: bool = False,
    unscale: bool = False,
    scaling_params=None,
) -> Dict[str, np.ndarray]:
    d = subset.data
    Y_all = np.asarray(d["outputs"]).astype(float)
    A_all = np.asarray(d["current_treatments"]).astype(float)
    Ls = np.asarray(d["sequence_lengths"]).astype(int)
    Xs_all = np.asarray(d["static_features"]).astype(float) if "static_features" in d else None

    tv_name = _infer_time_varying_key(subset, user_key=str(tv_key))
    TV_all = np.asarray(d[tv_name]).astype(float) if (tv_name is not None) else None

    N, T, Dy = Y_all.shape
    A_dim = int(A_all.shape[-1])
    D_static = int(Xs_all.shape[-1]) if (Xs_all is not None) else 0
    D_tv = int(TV_all.shape[-1]) if (TV_all is not None) else 0
    input_dim = int(Dy + A_dim + D_static + D_tv)

    H = int(horizon)
    t0_all = _get_future_past_split_points(subset, horizon=H)

    idxs = np.arange(N)
    if max_eval_windows is not None and N > int(max_eval_windows):
        rng = np.random.default_rng(0)
        idxs = rng.choice(idxs, size=int(max_eval_windows), replace=False)
    idxs = np.asarray(idxs, dtype=int)
    Nw = int(len(idxs))

    mu = np.full((Nw, H, Dy), np.nan, float)
    var = np.full((Nw, H, Dy), np.nan, float)
    y = np.full((Nw, H, Dy), np.nan, float)
    m = np.zeros((Nw, H, Dy), float)

    transformer.eval().to(device)

    for r, i in enumerate(idxs.tolist()):
        i = int(i)
        L = int(Ls[i])
        t0 = int(t0_all[i])
        if t0 <= 0 or t0 >= L:
            continue
        Hi = min(H, L - t0)
        if Hi <= 0:
            continue

        y_i = Y_all[i]
        a_i = A_all[i]
        xs_i = Xs_all[i] if Xs_all is not None else None
        tv_i = TV_all[i] if TV_all is not None else None

        x_tokens = np.zeros((T, input_dim), dtype=np.float32)

        # teacher-forced past
        y_prev = np.zeros((Dy,), dtype=np.float32)
        for t in range(t0):
            tv_t = tv_i[t] if (tv_i is not None) else None
            tok = build_token_from_parts(
                y_prev=y_prev, a_onehot=a_i[t], x_static=xs_i,
                tv_t=(np.zeros((0,), dtype=np.float32) if tv_t is None else tv_t),
                Dy=Dy, A_dim=A_dim, D_static=D_static, D_tv=D_tv
            )
            x_tokens[t] = tok
            y_prev = y_i[t].astype(np.float32).reshape(-1)

        # rollout future
        y_prev_pred = y_i[t0 - 1].astype(np.float32).reshape(-1)
        for k in range(Hi):
            t = t0 + k
            tv_t = tv_i[t] if (tv_i is not None) else None
            tok = build_token_from_parts(
                y_prev=y_prev_pred, a_onehot=a_i[t], x_static=xs_i,
                tv_t=(np.zeros((0,), dtype=np.float32) if tv_t is None else tv_t),
                Dy=Dy, A_dim=A_dim, D_static=D_static, D_tv=D_tv
            )
            x_tokens[t] = tok

            x = torch.from_numpy(x_tokens).unsqueeze(0).to(device)
            mask_bt = torch.zeros((1, T), device=device, dtype=torch.float32)
            mask_bt[0, :t + 1] = 1.0
            pad_mask = mask_to_padding(mask_bt)

            mean, logvar = transformer(x, src_key_padding_mask=pad_mask)
            mean_t = mean[0, t].detach().cpu().numpy()
            logvar_t = torch.clamp(logvar[0, t], min=logvar_clip[0], max=logvar_clip[1]).detach().cpu().numpy()
            var_t = np.exp(logvar_t) + 1e-8

            mu[r, k, :] = mean_t
            var[r, k, :] = var_t
            y[r, k, :] = y_i[t, :]

            valid = 1.0 if (t < L and np.all(np.isfinite(y_i[t, :]))) else 0.0
            m[r, k, :] = valid

            y_prev_pred = mean_t.astype(np.float32).reshape(-1)

    if unscale:
        mu_u, var_u = unscale_gaussian(mu, var, scaling_params)
        y_u = unscale_y(y, scaling_params)
        mu, var, y = mu_u, var_u, y_u

    rmse = _call_masked_rmse_per_tau(mu=mu, y=y, mask=m, H=H)
    nll = _call_masked_gaussian_nll_per_tau(mu=mu, var=var, y=y, mask=m, include_const=include_const_in_nll, H=H)
    return {"mu": mu, "var": var, "y": y, "mask": m, "rmse": rmse, "nll": nll}


# =============================================================================
# SC-only evaluation collection (teacher-forced or rollout)
# =============================================================================

def collect_sc_teacher_forced(
    sc: DynamicSyntheticControlKernel,
    subset,
    horizon: int,
    max_eval_windows: Optional[int] = None,
    include_const_in_nll: bool = False,
    unscale: bool = False,
    scaling_params=None,
) -> Dict[str, np.ndarray]:
    d = subset.data
    Y_all = np.asarray(d["outputs"]).astype(float)
    A_all = np.asarray(d["current_treatments"]).astype(float)
    Ls = np.asarray(d["sequence_lengths"]).astype(int)
    N, T, Dy = Y_all.shape
    H = int(horizon)

    t0_all = _get_future_past_split_points(subset, horizon=H)

    idxs = np.arange(N)
    if max_eval_windows is not None and N > int(max_eval_windows):
        rng = np.random.default_rng(0)
        idxs = rng.choice(idxs, size=int(max_eval_windows), replace=False)
    idxs = np.asarray(idxs, dtype=int)
    Nw = int(len(idxs))

    mu = np.full((Nw, H, Dy), np.nan, float)
    var = np.full((Nw, H, Dy), np.nan, float)
    y = np.full((Nw, H, Dy), np.nan, float)
    m = np.zeros((Nw, H, Dy), float)

    for r, i in enumerate(idxs.tolist()):
        i = int(i)
        L = int(Ls[i])
        t0 = int(t0_all[i])
        if t0 <= 0 or t0 >= L:
            continue
        Hi = min(H, L - t0)
        if Hi <= 0:
            continue

        y_i = Y_all[i]
        a_i = A_all[i]

        for k in range(Hi):
            t = t0 + k
            Hh = min(sc.hist_len, t)
            y_hist = y_i[t - Hh:t, :]
            if np.any(~np.isfinite(y_hist)):
                continue

            mu_t, var_t, _diag = sc.predict_step(y_hist=y_hist, a_t=a_i[t], t=t)
            if np.any(~np.isfinite(mu_t)) or np.any(~np.isfinite(var_t)):
                continue

            mu[r, k, :] = mu_t
            var[r, k, :] = var_t
            y[r, k, :] = y_i[t, :]
            m[r, k, :] = 1.0 if np.all(np.isfinite(y_i[t, :])) else 0.0

    if unscale:
        mu_u, var_u = unscale_gaussian(mu, var, scaling_params)
        y_u = unscale_y(y, scaling_params)
        mu, var, y = mu_u, var_u, y_u

    rmse = _call_masked_rmse_per_tau(mu=mu, y=y, mask=m, H=H)
    nll = _call_masked_gaussian_nll_per_tau(mu=mu, var=var, y=y, mask=m, include_const=include_const_in_nll, H=H)
    return {"mu": mu, "var": var, "y": y, "mask": m, "rmse": rmse, "nll": nll}


def collect_sc_rollout(
    sc: DynamicSyntheticControlKernel,
    subset,
    horizon: int,
    max_eval_windows: Optional[int] = None,
    include_const_in_nll: bool = False,
    unscale: bool = False,
    scaling_params=None,
) -> Dict[str, np.ndarray]:
    d = subset.data
    Y_all = np.asarray(d["outputs"]).astype(float)
    A_all = np.asarray(d["current_treatments"]).astype(float)
    Ls = np.asarray(d["sequence_lengths"]).astype(int)
    N, T, Dy = Y_all.shape
    H = int(horizon)

    t0_all = _get_future_past_split_points(subset, horizon=H)

    idxs = np.arange(N)
    if max_eval_windows is not None and N > int(max_eval_windows):
        rng = np.random.default_rng(0)
        idxs = rng.choice(idxs, size=int(max_eval_windows), replace=False)
    idxs = np.asarray(idxs, dtype=int)
    Nw = int(len(idxs))

    mu = np.full((Nw, H, Dy), np.nan, float)
    var = np.full((Nw, H, Dy), np.nan, float)
    y = np.full((Nw, H, Dy), np.nan, float)
    m = np.zeros((Nw, H, Dy), float)

    for r, i in enumerate(idxs.tolist()):
        i = int(i)
        L = int(Ls[i])
        t0 = int(t0_all[i])
        if t0 <= 0 or t0 >= L:
            continue
        Hi = min(H, L - t0)
        if Hi <= 0:
            continue

        y_i = Y_all[i]
        a_i = A_all[i]

        # initialize history with true up to t0-1
        t_init = t0
        Hh0 = min(sc.hist_len, t_init)
        y_hist = y_i[t_init - Hh0:t_init, :].copy()

        for k in range(Hi):
            t = t0 + k
            mu_t, var_t, _diag = sc.predict_step(y_hist=y_hist, a_t=a_i[t], t=t)
            if np.any(~np.isfinite(mu_t)) or np.any(~np.isfinite(var_t)):
                continue

            mu[r, k, :] = mu_t
            var[r, k, :] = var_t
            y[r, k, :] = y_i[t, :]
            m[r, k, :] = 1.0 if (t < L and np.all(np.isfinite(y_i[t, :]))) else 0.0

            # rollout history update
            y_hist = np.concatenate([y_hist, mu_t[None, :]], axis=0)
            if y_hist.shape[0] > sc.hist_len:
                y_hist = y_hist[-sc.hist_len:, :]

    if unscale:
        mu_u, var_u = unscale_gaussian(mu, var, scaling_params)
        y_u = unscale_y(y, scaling_params)
        mu, var, y = mu_u, var_u, y_u

    rmse = _call_masked_rmse_per_tau(mu=mu, y=y, mask=m, H=H)
    nll = _call_masked_gaussian_nll_per_tau(mu=mu, var=var, y=y, mask=m, include_const=include_const_in_nll, H=H)
    return {"mu": mu, "var": var, "y": y, "mask": m, "rmse": rmse, "nll": nll}


# =============================================================================
# FUSED evaluation collection (rollout)
# =============================================================================

@torch.no_grad()
def collect_fused_rollout(
    transformer: nn.Module,
    sc: DynamicSyntheticControlKernel,
    subset,
    tv_key: str,
    horizon: int,
    device: str,
    fusion: str = "ci",          # "ci" or "kalman"
    ci_grid: int = 51,
    ci_objective: str = "logdet",
    logvar_clip: Tuple[float, float] = (-10.0, 8.0),
    max_eval_windows: Optional[int] = None,
    include_const_in_nll: bool = False,
    unscale: bool = False,
    scaling_params=None,
) -> Dict[str, np.ndarray]:
    d = subset.data
    Y_all = np.asarray(d["outputs"]).astype(float)
    A_all = np.asarray(d["current_treatments"]).astype(float)
    Ls = np.asarray(d["sequence_lengths"]).astype(int)
    Xs_all = np.asarray(d["static_features"]).astype(float) if "static_features" in d else None

    tv_name = _infer_time_varying_key(subset, user_key=str(tv_key))
    TV_all = np.asarray(d[tv_name]).astype(float) if (tv_name is not None) else None

    N, T, Dy = Y_all.shape
    A_dim = int(A_all.shape[-1])
    D_static = int(Xs_all.shape[-1]) if (Xs_all is not None) else 0
    D_tv = int(TV_all.shape[-1]) if (TV_all is not None) else 0
    input_dim = int(Dy + A_dim + D_static + D_tv)

    H = int(horizon)
    t0_all = _get_future_past_split_points(subset, horizon=H)

    idxs = np.arange(N)
    if max_eval_windows is not None and N > int(max_eval_windows):
        rng = np.random.default_rng(0)
        idxs = rng.choice(idxs, size=int(max_eval_windows), replace=False)
    idxs = np.asarray(idxs, dtype=int)
    Nw = int(len(idxs))

    mu_f = np.full((Nw, H, Dy), np.nan, float)
    var_f = np.full((Nw, H, Dy), np.nan, float)
    y = np.full((Nw, H, Dy), np.nan, float)
    m = np.zeros((Nw, H, Dy), float)

    transformer.eval().to(device)

    for r, i in enumerate(idxs.tolist()):
        i = int(i)
        L = int(Ls[i])
        t0 = int(t0_all[i])
        if t0 <= 0 or t0 >= L:
            continue
        Hi = min(H, L - t0)
        if Hi <= 0:
            continue

        y_i = Y_all[i]
        a_i = A_all[i]
        xs_i = Xs_all[i] if Xs_all is not None else None
        tv_i = TV_all[i] if TV_all is not None else None

        x_tokens = np.zeros((T, input_dim), dtype=np.float32)

        # teacher-forced past
        y_prev = np.zeros((Dy,), dtype=np.float32)
        for t in range(t0):
            tv_t = tv_i[t] if (tv_i is not None) else None
            tok = build_token_from_parts(
                y_prev=y_prev, a_onehot=a_i[t], x_static=xs_i,
                tv_t=(np.zeros((0,), dtype=np.float32) if tv_t is None else tv_t),
                Dy=Dy, A_dim=A_dim, D_static=D_static, D_tv=D_tv
            )
            x_tokens[t] = tok
            y_prev = y_i[t].astype(np.float32).reshape(-1)

        # initialize SC history with truth up to t0-1
        Hh0 = min(sc.hist_len, t0)
        y_hist_sc = y_i[t0 - Hh0:t0, :].copy()

        # rollout future
        y_prev_pred = y_i[t0 - 1].astype(np.float32).reshape(-1)
        for k in range(Hi):
            t = t0 + k

            # prior step
            tv_t = tv_i[t] if (tv_i is not None) else None
            tok = build_token_from_parts(
                y_prev=y_prev_pred, a_onehot=a_i[t], x_static=xs_i,
                tv_t=(np.zeros((0,), dtype=np.float32) if tv_t is None else tv_t),
                Dy=Dy, A_dim=A_dim, D_static=D_static, D_tv=D_tv
            )
            x_tokens[t] = tok

            x = torch.from_numpy(x_tokens).unsqueeze(0).to(device)
            mask_bt = torch.zeros((1, T), device=device, dtype=torch.float32)
            mask_bt[0, :t + 1] = 1.0
            pad_mask = mask_to_padding(mask_bt)

            mean_p, logvar_p = transformer(x, src_key_padding_mask=pad_mask)
            mu_p = mean_p[0, t].detach().cpu().numpy().astype(np.float32)
            logvar_t = torch.clamp(logvar_p[0, t], min=logvar_clip[0], max=logvar_clip[1]).detach().cpu().numpy()
            var_p = (np.exp(logvar_t) + 1e-8).astype(np.float32)

            # SC step (Fix B is inside sc.predict_step)
            mu_s, var_s, _diag = sc.predict_step(y_hist=y_hist_sc, a_t=a_i[t], t=t)

            # if SC missing -> fall back to prior
            if np.any(~np.isfinite(mu_s)) or np.any(~np.isfinite(var_s)):
                mu_post, var_post = mu_p, var_p
            else:
                if fusion == "kalman":
                    mu_post, var_post = fuse_diag_kalman(mu_p, var_p, mu_s, var_s)
                else:
                    mu_post, var_post, _w = fuse_diag_ci(mu_p, var_p, mu_s, var_s,
                                                         grid=int(ci_grid), objective=str(ci_objective))

            mu_f[r, k, :] = mu_post
            var_f[r, k, :] = var_post
            y[r, k, :] = y_i[t, :]
            m[r, k, :] = 1.0 if (t < L and np.all(np.isfinite(y_i[t, :]))) else 0.0

            # update histories
            y_prev_pred = mu_post.astype(np.float32).reshape(-1)
            y_hist_sc = np.concatenate([y_hist_sc, mu_post[None, :]], axis=0)
            if y_hist_sc.shape[0] > sc.hist_len:
                y_hist_sc = y_hist_sc[-sc.hist_len:, :]

    if unscale:
        mu_u, var_u = unscale_gaussian(mu_f, var_f, scaling_params)
        y_u = unscale_y(y, scaling_params)
        mu_f, var_f, y = mu_u, var_u, y_u

    rmse = _call_masked_rmse_per_tau(mu=mu_f, y=y, mask=m, H=H)
    nll = _call_masked_gaussian_nll_per_tau(mu=mu_f, var=var_f, y=y, mask=m, include_const=include_const_in_nll, H=H)
    return {"mu": mu_f, "var": var_f, "y": y, "mask": m, "rmse": rmse, "nll": nll}


# =============================================================================
# Printing / saving evaluation reports
# =============================================================================

def _masked_overall_rmse(mu: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    eps = 1e-12
    se = ((mu - y) ** 2) * mask
    num = float(np.nansum(se))
    den = float(np.nansum(mask)) + eps
    return float(np.sqrt(num / den))


def _masked_overall_gaussian_nll(mu: np.ndarray, var: np.ndarray, y: np.ndarray, mask: np.ndarray,
                                 include_const: bool = False) -> float:
    eps = 1e-12
    var = np.maximum(var, 1e-8)
    nll = 0.5 * (((y - mu) ** 2) / var + np.log(var))
    if include_const:
        nll = nll + 0.5 * np.log(2.0 * np.pi)
    nll = nll * mask
    num = float(np.nansum(nll))
    den = float(np.nansum(mask)) + eps
    return float(num / den)


def print_eval_report(
    name: str,
    pack: Dict[str, np.ndarray],
    eval_taus_1based: List[int],
    include_const_in_nll: bool = False,
    max_rows: int = 80,
):
    mu = np.asarray(pack["mu"])
    var = np.asarray(pack["var"])
    y = np.asarray(pack["y"])
    mask = np.asarray(pack["mask"])
    rmse = np.asarray(pack["rmse"]).reshape(-1)
    nll = np.asarray(pack["nll"]).reshape(-1)

    Nw, H, Dy = mu.shape

    valid_pts = float(np.nansum(mask))
    total_pts = float(Nw * H * Dy)
    valid_ratio = valid_pts / max(total_pts, 1.0)

    taus = list(eval_taus_1based)
    rmse_sel = np.array([rmse[t - 1] for t in taus], dtype=float)
    nll_sel = np.array([nll[t - 1] for t in taus], dtype=float)

    overall_rmse = _masked_overall_rmse(mu, y, mask)
    overall_nll = _masked_overall_gaussian_nll(mu, var, y, mask, include_const=include_const_in_nll)

    print("\n" + "=" * 80)
    print(f"[{name}] Evaluation Report")
    print("-" * 80)
    print(f"Windows: N={Nw} | Horizon H={H} | Dy={Dy}")
    print(f"Valid masked points: {valid_pts:.0f} / {total_pts:.0f} ({100.0 * valid_ratio:.2f}%)")
    print("-" * 80)

    print(f"Per-step metrics (showing up to {max_rows} taus):")
    print(f"{'tau':>6}  {'RMSE':>12}  {'NLL':>12}")
    print("-" * 34)
    for j, t in enumerate(taus[:max_rows]):
        print(f"{t:6d}  {rmse_sel[j]:12.6f}  {nll_sel[j]:12.6f}")
    if len(taus) > max_rows:
        print(f"... ({len(taus) - max_rows} more taus omitted)")
    print("-" * 34)

    print("Aggregates over selected taus:")
    print(f"  RMSE mean/median/last: {rmse_sel.mean():.6f} / {np.median(rmse_sel):.6f} / {rmse_sel[-1]:.6f}")
    print(f"  NLL  mean/median/last: {nll_sel.mean():.6f} / {np.median(nll_sel):.6f} / {nll_sel[-1]:.6f}")

    print("Overall (flattened over all windows × taus × dims in pack):")
    print(f"  RMSE overall: {overall_rmse:.6f}")
    print(f"  NLL  overall: {overall_nll:.6f}")
    print("=" * 80 + "\n")

    logger.info(f"[{name}] avg(RMSE)={rmse_sel.mean():.6f} avg(NLL)={nll_sel.mean():.6f} overall_RMSE={overall_rmse:.6f}")


def save_eval_json(path: str, reports: Dict[str, Dict], indent: int = 2):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(reports, f, indent=indent)
    logger.info(f"[eval] saved JSON report to: {path}")


# =============================================================================
# Checkpoint I/O
# =============================================================================

def save_prior_checkpoint(path: str, model: nn.Module, meta: Dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, path)
    logger.info(f"[save] transformer prior saved to: {path}")


def load_prior_checkpoint(path: str, device: str = "cpu") -> Tuple[nn.Module, Dict]:
    ckpt = torch.load(path, map_location=device)
    if "arch" not in ckpt:
        raise ValueError(f"Checkpoint at {path} missing 'arch' field.")
    if "input_dim" not in ckpt or "Dy" not in ckpt:
        raise ValueError(f"Checkpoint at {path} missing 'input_dim' or 'Dy' fields.")

    arch = ckpt["arch"]
    model = GaussianCausalTransformer(
        input_dim=int(ckpt["input_dim"]),
        dim_outcomes=int(ckpt["Dy"]),
        d_model=int(arch.get("d_model", 128)),
        n_heads=int(arch.get("n_heads", 4)),
        num_layers=int(arch.get("n_layers", 3)),
        dim_feedforward=int(arch.get("ff_dim", arch.get("dim_feedforward", 256))),
        dropout=float(arch.get("dropout", 0.1)),
        max_len=int(arch.get("max_len", 256)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k not in ("state_dict",)}
    logger.info(f"[load] transformer prior loaded from: {path}")
    return model, meta


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CancerSim: Transformer prior (Gaussian+MMD) + Dynamic SC + evaluation (RMSE/NLL)."
    )

    # Seeds / dataset generation
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--data_seed", type=int, default=180)
    parser.add_argument("--coeff", type=float, default=0.0)


    parser.add_argument("--split", type=str, default="test_cf_treatment_seq")
    parser.add_argument("--do_plot_gt", type=int, default=0)
    parser.add_argument("--n_examples", type=int, default=100)
    parser.add_argument("--random", type=int, default=0)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--out_dim", type=int, default=0)
    parser.add_argument("--plot_treatments", type=int, default=1)
    parser.add_argument("--plot_dir", type=str, default="plots_cancer_gt")
    parser.add_argument("--title_prefix", type=str, default="")
    parser.add_argument("--ylim", type=str, default="")
    parser.add_argument("--show", type=int, default=0)


    parser.add_argument("--do_plot_pred", type=int, default=0)
    parser.add_argument("--plot_pred_split", type=str, default="",
                        help="Split to plot predictions on (defaults to --eval_split if empty).")
    parser.add_argument("--plot_pred_dir", type=str, default="plots_cancer_pred")
    parser.add_argument("--plot_pred_n", type=int, default=50)
    parser.add_argument("--plot_pred_random", type=int, default=1)
    parser.add_argument("--plot_pred_seed", type=int, default=0)
    parser.add_argument("--plot_pred_out_dim", type=int, default=0)
    parser.add_argument("--plot_pred_plot_treatments", type=int, default=1)
    parser.add_argument("--plot_pred_sigma", type=float, default=1.96)
    parser.add_argument("--plot_pred_ci_band", type=int, default=1)
    parser.add_argument("--plot_pred_which", type=str, default="fused",
                        choices=["prior", "sc", "fused", "all"])
    parser.add_argument("--plot_pred_unscale", type=int, default=-1,
                        help="If 1 unscale plots; if 0 plot scaled; if -1 follow --unscale_eval.")

    # Transformer prior training / loading
    parser.add_argument("--train_prior", type=int, default=1)
    parser.add_argument("--train_split", type=str, default="train_f")
    parser.add_argument("--tv_key", type=str, default="")
    parser.add_argument("--prior_epochs", type=int, default=100)
    parser.add_argument("--prior_batch_size", type=int, default=64)
    parser.add_argument("--prior_lr", type=float, default=1e-3)

    # MMD
    parser.add_argument("--lambda_mmd", type=float, default=0.01)
    parser.add_argument("--mmd_min_per_group", type=int, default=8)
    parser.add_argument("--mmd_max_per_group", type=int, default=256)
    parser.add_argument("--mmd_bw_mode", type=str, default="median", choices=["median", "fixed"])
    parser.add_argument("--mmd_bw", type=float, default=None)
    parser.add_argument("--mmd_bw_scale", type=float, default=1.0)

    # Transformer arch
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=256)

    # Checkpoint I/O
    parser.add_argument("--save_prior", type=str, default="")
    parser.add_argument("--load_prior", type=str, default="")

    # Evaluation toggles
    parser.add_argument("--eval_prior", type=int, default=1)
    parser.add_argument("--eval_sc", type=int, default=1)
    parser.add_argument("--eval_fused", type=int, default=1)

    parser.add_argument("--eval_split", type=str, default="test_cf_treatment_seq")
    parser.add_argument("--eval_mode", type=str, default="rollout", choices=["rollout", "teacher_forced"])
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--eval_taus", type=str, default="all")
    parser.add_argument("--max_eval_windows", type=int, default=None)
    parser.add_argument("--include_const_nll", type=int, default=0)
    parser.add_argument("--unscale_eval", type=int, default=0)

    # Donor pool for SC
    parser.add_argument("--donor_split", type=str, default="train_f",
                        help="Donor pool split for SC (typically train_f).")

    # Dynamic SC params
    parser.add_argument("--sc_hist_len", type=int, default=5)
    parser.add_argument("--sc_k", type=int, default=50)
    parser.add_argument("--sc_bw", type=float, default=1.0)
    parser.add_argument("--sc_z_std_floor", type=float, default=1e-3)
    parser.add_argument("--sc_var_floor", type=float, default=1e-4)
    parser.add_argument("--sc_min_donors", type=int, default=5)
    parser.add_argument("--sc_use_delta_feat", type=int, default=1)

    # ---- Fix B args (reliability inflation) ----
    parser.add_argument("--sc_use_reliability", type=int, default=1,
                        help="If 1, inflate SC variance using r(ESS,w_max).")
    parser.add_argument("--sc_ess_ref", type=float, default=-1.0,
                        help="Reference ESS. If <=0, uses 0.5*k_eff per step.")
    parser.add_argument("--sc_rel_eps", type=float, default=1e-6,
                        help="epsilon in var/(r+eps).")
    parser.add_argument("--sc_R_min", type=float, default=1e-3,
                        help="Additive variance floor after inflation.")
    parser.add_argument("--sc_drop_bad_support", type=int, default=0,
                        help="If 1, ignore SC step (return NaN) when ESS low or w_max high.")
    parser.add_argument("--sc_drop_ess_min", type=float, default=5.0,
                        help="Drop SC when ESS < this (only if sc_drop_bad_support=1).")
    parser.add_argument("--sc_drop_wmax_max", type=float, default=0.9,
                        help="Drop SC when w_max > this (only if sc_drop_bad_support=1).")
    parser.add_argument("--sc_r_min_clip", type=float, default=0.0,
                        help="Optional clamp r >= this (0 disables).")

    # Fusion params
    parser.add_argument("--fusion", type=str, default="ci", choices=["ci", "kalman"])
    parser.add_argument("--ci_grid", type=int, default=51)
    parser.add_argument("--ci_objective", type=str, default="logdet", choices=["logdet", "trace"])

    # Persist evaluation
    parser.add_argument("--save_eval_json", type=str, default="",
                        help="If set, save metrics to this JSON file.")

    args = parser.parse_args()

    data_seed = int(args.seed) if args.seed is not None else int(args.data_seed)
    seed_everything(data_seed)

    # -------------------------------------------------------------------------
    # Load / generate dataset
    # -------------------------------------------------------------------------
    dataset_collection = generate_cancer_dataset(seed=data_seed, coeff=float(args.coeff))
    if hasattr(dataset_collection, "process_data_multi"):
        dataset_collection.process_data_multi()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[device] {device}")

    # Horizon default
    if args.horizon is not None:
        horizon = int(args.horizon)
    else:
        horizon = int(getattr(dataset_collection, "projection_horizon", 1))
    horizon = max(1, int(horizon))

    # -------------------------------------------------------------------------
    # Build / load transformer
    # -------------------------------------------------------------------------
    transformer = None
    ckpt_meta = None

    if int(args.train_prior) == 1:
        train_subset = _get_subset(dataset_collection, args.train_split)
        train_ds = CancerTokenDataset(train_subset, tv_key=str(args.tv_key))

        T_max = int(np.asarray(train_subset.data["outputs"]).shape[1])
        max_len = int(max(int(args.max_len), T_max))
        logger.info(f"[max_len] inferred T_max={T_max} => max_len={max_len}")

        transformer = GaussianCausalTransformer(
            input_dim=int(train_ds.input_dim),
            dim_outcomes=int(train_ds.Dy),
            d_model=int(args.d_model),
            n_heads=int(args.n_heads),
            num_layers=int(args.n_layers),
            dim_feedforward=int(args.ff_dim),
            dropout=float(args.dropout),
            max_len=max_len,
        ).to(device)

        train_gaussian_prior(
            model=transformer,
            train_ds=train_ds,
            num_epochs=int(args.prior_epochs),
            batch_size=int(args.prior_batch_size),
            lr=float(args.prior_lr),
            device=device,
            lambda_mmd=float(args.lambda_mmd),
            treat_slice=train_ds.treat_slice,
            A_dim=int(train_ds.A_dim),
            mmd_min_per_group=int(args.mmd_min_per_group),
            mmd_max_per_group=int(args.mmd_max_per_group),
            mmd_bw_mode=str(args.mmd_bw_mode),
            mmd_bw=args.mmd_bw,
            mmd_bw_scale=float(args.mmd_bw_scale),
        )

        if isinstance(args.save_prior, str) and args.save_prior.strip():
            out_path = str(args.save_prior)
            meta = {
                "input_dim": int(train_ds.input_dim),
                "Dy": int(train_ds.Dy),
                "A_dim": int(train_ds.A_dim),
                "D_static": int(train_ds.D_static),
                "D_tv": int(train_ds.D_tv),
                "tv_key": train_ds.tv_key,
                "treat_slice": tuple(train_ds.treat_slice),
                "arch": {
                    "d_model": int(args.d_model),
                    "n_heads": int(args.n_heads),
                    "n_layers": int(args.n_layers),
                    "ff_dim": int(args.ff_dim),
                    "dropout": float(args.dropout),
                    "max_len": int(max_len),
                },
            }
            save_prior_checkpoint(out_path, transformer, meta)
    else:
        if isinstance(args.load_prior, str) and args.load_prior.strip():
            transformer, ckpt_meta = load_prior_checkpoint(str(args.load_prior), device=device)
            if (not str(args.tv_key).strip()) and ckpt_meta is not None and ("tv_key" in ckpt_meta):
                args.tv_key = str(ckpt_meta["tv_key"] or "")
        else:
            transformer = None

    # -------------------------------------------------------------------------
    # Build Dynamic SC donor pool
    # -------------------------------------------------------------------------
    donor_subset = _get_subset(dataset_collection, args.donor_split)
    d_donor = donor_subset.data
    donor_Y = np.asarray(d_donor["outputs"]).astype(np.float32)
    donor_A = np.asarray(d_donor["current_treatments"]).astype(np.float32)
    donor_L = np.asarray(d_donor["sequence_lengths"]).astype(int)

    sc = DynamicSyntheticControlKernel(
        donor_Y=donor_Y,
        donor_A=donor_A,
        donor_L=donor_L,
        hist_len=int(args.sc_hist_len),
        k=int(args.sc_k),
        bw=float(args.sc_bw),
        z_std_floor=float(args.sc_z_std_floor),
        var_floor=float(args.sc_var_floor),
        min_donors=int(args.sc_min_donors),
        use_delta_feat=int(args.sc_use_delta_feat),

        # Fix B
        use_reliability=int(args.sc_use_reliability),
        ess_ref=float(args.sc_ess_ref),
        rel_eps=float(args.sc_rel_eps),
        R_min=float(args.sc_R_min),
        drop_bad_support=int(args.sc_drop_bad_support),
        drop_ess_min=float(args.sc_drop_ess_min),
        drop_wmax_max=float(args.sc_drop_wmax_max),
        r_min_clip=float(args.sc_r_min_clip),
    )

    logger.info(
        f"[SC] donor_split={args.donor_split} Nd={donor_Y.shape[0]} hist_len={sc.hist_len} k={sc.k} bw={sc.bw} "
        f"use_reliability={int(sc.use_reliability)} R_min={sc.R_min:g} drop_bad_support={int(sc.drop_bad_support)}"
    )

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    reports_for_json = {}
    if int(args.eval_prior) == 1 or int(args.eval_sc) == 1 or int(args.eval_fused) == 1:
        eval_subset = _get_subset(dataset_collection, args.eval_split)
        eval_taus = parse_taus_arg(args.eval_taus, horizon=horizon)
        include_const = bool(int(args.include_const_nll))
        unscale = bool(int(args.unscale_eval))

        scaling_params = None
        if unscale:
            if hasattr(dataset_collection, "train_f") and hasattr(dataset_collection.train_f, "scaling_params"):
                scaling_params = dataset_collection.train_f.scaling_params
            elif hasattr(dataset_collection, "train_scaling_params"):
                scaling_params = dataset_collection.train_scaling_params
            else:
                raise ValueError("unscale_eval=1 but scaling_params not found on dataset_collection.train_f.")

        logger.info("=" * 80)
        logger.info(f"[EVAL] split={args.eval_split} | mode={args.eval_mode} | H={horizon} | taus={eval_taus}")
        logger.info(f"[EVAL] donor_split={args.donor_split} | fusion={args.fusion} ci_grid={args.ci_grid} obj={args.ci_objective}")
        logger.info("=" * 80)

        # PRIOR
        if int(args.eval_prior) == 1:
            if transformer is None:
                raise ValueError("eval_prior=1 requires a trained model (--train_prior 1) or a checkpoint (--load_prior).")

            if args.eval_mode == "teacher_forced":
                pack_prior = collect_prior_teacher_forced(
                    transformer=transformer, subset=eval_subset, tv_key=str(args.tv_key),
                    horizon=horizon, device=device, include_const_in_nll=include_const,
                    max_eval_windows=args.max_eval_windows, unscale=unscale, scaling_params=scaling_params
                )
            else:
                pack_prior = collect_prior_rollout(
                    transformer=transformer, subset=eval_subset, tv_key=str(args.tv_key),
                    horizon=horizon, device=device, include_const_in_nll=include_const,
                    max_eval_windows=args.max_eval_windows, unscale=unscale, scaling_params=scaling_params
                )

            print_eval_report(
                name=f"PRIOR ({args.eval_mode}) split={args.eval_split}",
                pack=pack_prior,
                eval_taus_1based=eval_taus,
                include_const_in_nll=include_const,
                max_rows=80,
            )
            reports_for_json["prior"] = {
                "rmse_per_tau": list(np.asarray(pack_prior["rmse"]).reshape(-1)),
                "nll_per_tau": list(np.asarray(pack_prior["nll"]).reshape(-1)),
            }

        # SC
        if int(args.eval_sc) == 1:
            if args.eval_mode == "teacher_forced":
                pack_sc = collect_sc_teacher_forced(
                    sc=sc, subset=eval_subset, horizon=horizon,
                    max_eval_windows=args.max_eval_windows, include_const_in_nll=include_const,
                    unscale=unscale, scaling_params=scaling_params
                )
            else:
                pack_sc = collect_sc_rollout(
                    sc=sc, subset=eval_subset, horizon=horizon,
                    max_eval_windows=args.max_eval_windows, include_const_in_nll=include_const,
                    unscale=unscale, scaling_params=scaling_params
                )

            print_eval_report(
                name=f"SC ({args.eval_mode}) split={args.eval_split}",
                pack=pack_sc,
                eval_taus_1based=eval_taus,
                include_const_in_nll=include_const,
                max_rows=80,
            )
            reports_for_json["sc"] = {
                "rmse_per_tau": list(np.asarray(pack_sc["rmse"]).reshape(-1)),
                "nll_per_tau": list(np.asarray(pack_sc["nll"]).reshape(-1)),
            }

        # FUSED
        if int(args.eval_fused) == 1:
            if transformer is None:
                raise ValueError("eval_fused=1 requires transformer prior (train or load checkpoint).")

            pack_fused = collect_fused_rollout(
                transformer=transformer,
                sc=sc,
                subset=eval_subset,
                tv_key=str(args.tv_key),
                horizon=horizon,
                device=device,
                fusion=str(args.fusion),
                ci_grid=int(args.ci_grid),
                ci_objective=str(args.ci_objective),
                include_const_in_nll=include_const,
                max_eval_windows=args.max_eval_windows,
                unscale=unscale,
                scaling_params=scaling_params,
            )

            print_eval_report(
                name=f"FUSED ({args.fusion}) rollout split={args.eval_split}",
                pack=pack_fused,
                eval_taus_1based=eval_taus,
                include_const_in_nll=include_const,
                max_rows=80,
            )
            reports_for_json["fused"] = {
                "rmse_per_tau": list(np.asarray(pack_fused["rmse"]).reshape(-1)),
                "nll_per_tau": list(np.asarray(pack_fused["nll"]).reshape(-1)),
            }

        if isinstance(args.save_eval_json, str) and args.save_eval_json.strip():
            save_eval_json(str(args.save_eval_json), reports_for_json)

    # -------------------------------------------------------------------------
    # Plot ground truth (optional)
    # -------------------------------------------------------------------------
    if int(args.do_plot_gt) == 1:
        subset = _get_subset(dataset_collection, args.split)
        Y = np.asarray(subset.data["outputs"])
        N = Y.shape[0]
        idxs = _select_indices(N=N, n=int(args.n_examples), do_random=bool(int(args.random)), seed=int(args.sample_seed))

        ylim = None
        if isinstance(args.ylim, str) and args.ylim.strip():
            try:
                a, b = args.ylim.split(",", 1)
                ylim = (float(a.strip()), float(b.strip()))
            except Exception:
                logger.warning(f"Could not parse --ylim='{args.ylim}'. Ignoring.")
                ylim = None

        save_dir = os.path.join(str(args.plot_dir), str(args.split))
        plot_groundtruth(
            subset=subset,
            idxs=idxs,
            save_dir=save_dir,
            out_dim=int(args.out_dim),
            plot_treatments=bool(int(args.plot_treatments)),
            title_prefix=str(args.title_prefix),
            ylim=ylim,
            show=bool(int(args.show)),
        )

    # -------------------------------------------------------------------------
    # Plot factual history + counterfactual truth + predictions (optional)
    # -------------------------------------------------------------------------
    if int(args.do_plot_pred) == 1:
        plot_split = str(args.plot_pred_split).strip() or str(args.eval_split)
        subset = _get_subset(dataset_collection, plot_split)

        Y = np.asarray(subset.data["outputs"])
        N = int(Y.shape[0])
        idxs = _select_indices(N=N, n=int(args.plot_pred_n), do_random=bool(int(args.plot_pred_random)), seed=int(args.plot_pred_seed))

        # ylim reuse (optional)
        ylim = None
        if isinstance(args.ylim, str) and args.ylim.strip():
            try:
                a, b = args.ylim.split(",", 1)
                ylim = (float(a.strip()), float(b.strip()))
            except Exception:
                logger.warning(f"Could not parse --ylim='{args.ylim}'. Ignoring.")
                ylim = None

        # unscale handling
        if int(args.plot_pred_unscale) == -1:
            unscale_plot = bool(int(args.unscale_eval))
        else:
            unscale_plot = bool(int(args.plot_pred_unscale))

        scaling_params = None
        if unscale_plot:
            if hasattr(dataset_collection, "train_f") and hasattr(dataset_collection.train_f, "scaling_params"):
                scaling_params = dataset_collection.train_f.scaling_params
            elif hasattr(dataset_collection, "train_scaling_params"):
                scaling_params = dataset_collection.train_scaling_params
            else:
                raise ValueError("plot_pred_unscale=1 but scaling_params not found on dataset_collection.train_f.")

        save_dir = os.path.join(str(args.plot_pred_dir), str(plot_split), f"which_{args.plot_pred_which}")
        plot_cf_predictions(
            subset=subset,
            idxs=idxs,
            horizon=horizon,
            out_dim=int(args.plot_pred_out_dim),
            save_dir=save_dir,
            transformer=transformer,
            sc=sc,
            tv_key=str(args.tv_key),
            device=device,
            which=str(args.plot_pred_which),
            fusion=str(args.fusion),
            ci_grid=int(args.ci_grid),
            ci_objective=str(args.ci_objective),
            sigma=float(args.plot_pred_sigma),
            plot_ci_band=bool(int(args.plot_pred_ci_band)),
            plot_treatments=bool(int(args.plot_pred_plot_treatments)),
            title_prefix=str(args.title_prefix),
            ylim=ylim,
            show=bool(int(args.show)),
            unscale=bool(unscale_plot),
            scaling_params=scaling_params,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
