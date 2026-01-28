#!/usr/bin/env python3
"""
Evaluation utilities for trained models on semi-synthetic MIMIC-III dataset.
Evaluates all 4 models: PRIOR (Transformer), EVIDENCE (SC), EVIDENCE-TF (SC teacher-forced), FUSED (Kalman)
"""

import os
import sys
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def _prep_out_scaling(scaling_params, Dy: int):
    out_mean = np.asarray(scaling_params["output_means"]).astype(float).reshape(-1)
    out_std = np.asarray(scaling_params["output_stds"]).astype(float).reshape(-1)
    if out_mean.size == 1:
        out_mean = np.repeat(out_mean, Dy)
    if out_std.size == 1:
        out_std = np.repeat(out_std, Dy)
    assert out_mean.size == Dy and out_std.size == Dy
    return out_mean, out_std


def unscale_gaussian(mu_s, var_s, out_mean, out_std):
    """
    mu = mu_s * std + mean
    var = var_s * std^2
    """
    mu_s = np.asarray(mu_s, float)
    var_s = np.asarray(var_s, float)
    std = np.asarray(out_std, float).reshape(1, 1, -1)
    mean = np.asarray(out_mean, float).reshape(1, 1, -1)
    mu = mu_s * std + mean
    var = np.maximum(var_s * (std ** 2), 1e-8)
    return mu, var


def unscale_y(y_s, out_mean, out_std):
    y_s = np.asarray(y_s, float)
    std = np.asarray(out_std, float).reshape(1, 1, -1)
    mean = np.asarray(out_mean, float).reshape(1, 1, -1)
    return y_s * std + mean


def masked_rmse_per_tau(mu, y, mask, H: int, norm_const: float = 1.0):
    """
    Step-wise RMSE: MSE across all dims, then sqrt.
    This matches get_normalised_masked_rmse in time_varying_model.py.
    
    mu, y: (N,H,Dy) in UNITS (unscaled or scaled)
    mask: (N,H,Dy) in {0,1}
    norm_const: normalization constant (1.0 for MIMIC-III)
    
    Returns: list of RMSE per time step, averaged over (samples, dims)
    """
    mu = np.asarray(mu, float)
    y = np.asarray(y, float)
    mask = np.asarray(mask, float)

    # Filter out NaN/inf entries
    finite = np.isfinite(mu) & np.isfinite(y) & np.isfinite(mask)
    m = (mask > 0.0) & finite

    _, H2, Dy = mu.shape
    assert H2 == H

    out = [float("nan")] * H
    for t in range(H):
        # MSE across all samples and dims 
        mt = m[:, t, :]  # (N, Dy)
        n_valid = int(mt.sum())
        if n_valid < 1:
            continue
        # Only compute on valid entries
        mu_valid = mu[:, t, :][mt]
        y_valid = y[:, t, :][mt]
        e = y_valid - mu_valid
        mse_t = float(np.mean(e * e))
        rmse_t = float(np.sqrt(mse_t)) / norm_const
        out[t] = rmse_t
    return out


def masked_rmse_per_traj(mu, y, mask, norm_const: float = 1.0) -> np.ndarray:
    """
    Per-trajectory RMSE: MSE across all time steps and dims, then sqrt.

    mu, y: (N, H, Dy) in UNITS (unscaled or scaled)
    mask: (N, H, Dy) in {0,1}
    norm_const: normalization constant (1.0 for MIMIC-III)

    Returns: (N,) RMSE per trajectory
    """
    mu = np.asarray(mu, float)
    y = np.asarray(y, float)
    mask = np.asarray(mask, float)

    finite = np.isfinite(mu) & np.isfinite(y) & np.isfinite(mask)
    m = (mask > 0.0) & finite

    N = mu.shape[0]
    out = np.full((N,), np.nan, dtype=float)
    for i in range(N):
        mi = m[i]
        n_valid = int(mi.sum())
        if n_valid < 1:
            continue
        mu_valid = mu[i][mi]
        y_valid = y[i][mi]
        e = y_valid - mu_valid
        mse_i = float(np.mean(e * e))
        out[i] = float(np.sqrt(mse_i)) / norm_const

    return out


def compute_top_k_rmse(per_trajectory_rmse: np.ndarray, k_values: list = None) -> Dict[str, Any]:
    """
    Compute top-K RMSE statistics (average RMSE of worst-performing trajectories).
    
    Args:
        per_trajectory_rmse: (N,) array of per-trajectory RMSEs
        k_values: List of K values to compute top-K for (default: [1, 5, 10, 20, 50, 100])
        
    Returns:
        Dictionary with top-K RMSE values and overall statistics
    """
    if k_values is None:
        k_values = [1, 5, 10, 20, 50, 100]
    
    # Remove NaN values
    valid_rmses = per_trajectory_rmse[np.isfinite(per_trajectory_rmse)]
    
    if len(valid_rmses) == 0:
        logger.warning('No valid RMSEs to compute top-K')
        return {'top_k': {}, 'mean': float('nan'), 'std': float('nan'), 'count': 0}
    
    # Sort RMSEs in descending order (highest errors first)
    sorted_rmses = np.sort(valid_rmses)[::-1]
    
    # Compute top-K averages
    top_k_results = {}
    for k in k_values:
        if k <= len(sorted_rmses):
            top_k_results[k] = float(np.mean(sorted_rmses[:k]))
    
    return {
        'top_k': top_k_results,
        'mean': float(np.mean(valid_rmses)),
        'std': float(np.std(valid_rmses)),
        'count': int(len(valid_rmses))
    }


def masked_rmse_per_tau_avg_dims(mu, y, mask, H: int):
    """
    Step-wise RMSE, averaged over outcome dims (each dim RMSE computed separately).
    LEGACY approach: RMSE per dim, then average RMSEs.
    
    mu, y: (N,H,Dy) in UNITS (unscaled)
    mask: (N,H,Dy) in {0,1}
    
    Note: This gives different values than train_multi because:
          sqrt(mean(mse)) != mean(sqrt(mse))
    """
    mu = np.asarray(mu, float)
    y = np.asarray(y, float)
    mask = np.asarray(mask, float)

    # Filter out NaN/inf entries
    finite = np.isfinite(mu) & np.isfinite(y) & np.isfinite(mask)
    m = (mask > 0.0) & finite

    _, H2, Dy = mu.shape
    assert H2 == H

    out = [float("nan")] * H
    for t in range(H):
        vals = []
        for j in range(Dy):
            mj = m[:, t, j]
            n_valid = int(mj.sum())
            if n_valid < 1:
                continue
            e = y[mj, t, j] - mu[mj, t, j]
            rmse = float(np.sqrt(np.mean(e * e)))
            vals.append(rmse)
        if vals:
            out[t] = float(np.mean(vals))
    return out


def masked_gaussian_nll_per_tau(mu, var, y, mask, H: int, include_const: bool = True):
    """
    Step-wise Gaussian NLL: flatten across all dims (consistent with masked_rmse_per_tau).
    mu,var,y: (N,H,Dy) in UNITS (unscaled)
    mask: (N,H,Dy)
    
    Returns: list of NLL per time step, averaged over (samples, dims)
    """
    mu = np.asarray(mu, float)
    var = np.maximum(np.asarray(var, float), 1e-8)
    y = np.asarray(y, float)
    mask = np.asarray(mask, float)

    # Filter out NaN/inf entries
    finite = np.isfinite(mu) & np.isfinite(var) & np.isfinite(y) & np.isfinite(mask)
    m = (mask > 0.0) & finite

    _, H2, Dy = mu.shape
    assert H2 == H

    const = 0.5 * np.log(2.0 * np.pi) if include_const else 0.0

    out = [float("nan")] * H
    for t in range(H):
        # Flatten across all samples and dims
        mt = m[:, t, :]  # (N, Dy)
        n_valid = int(mt.sum())
        if n_valid < 1:
            continue
        # Only compute on valid entries
        mu_valid = mu[:, t, :][mt]
        var_valid = var[:, t, :][mt]
        y_valid = y[:, t, :][mt]
        e = y_valid - mu_valid
        nll = const + 0.5 * (np.log(var_valid) + (e * e) / var_valid)
        out[t] = float(np.mean(nll))
    return out


def masked_gaussian_nll_per_tau_avg_dims(mu, var, y, mask, H: int, include_const: bool = True):
    """
    Step-wise Gaussian NLL, averaged over outcome dims.
    mu,var,y: (N,H,Dy) in UNITS (unscaled)
    mask: (N,H,Dy)
    """
    mu = np.asarray(mu, float)
    var = np.maximum(np.asarray(var, float), 1e-8)
    y = np.asarray(y, float)
    mask = np.asarray(mask, float)

    # Filter out NaN/inf entries
    finite = np.isfinite(mu) & np.isfinite(var) & np.isfinite(y) & np.isfinite(mask)
    m = (mask > 0.0) & finite

    _, H2, Dy = mu.shape
    assert H2 == H

    const = 0.5 * np.log(2.0 * np.pi) if include_const else 0.0

    out = [float("nan")] * H
    for t in range(H):
        vals = []
        for j in range(Dy):
            mj = m[:, t, j]
            n_valid = int(mj.sum())
            if n_valid < 1:
                continue
            e = y[mj, t, j] - mu[mj, t, j]
            v = var[mj, t, j]
            nll = const + 0.5 * (np.log(v) + (e * e) / v)
            vals.append(float(np.mean(nll)))
        if vals:
            out[t] = float(np.mean(vals))
    return out


def get_stepwise_metric(title: str, metric_tau: list, taus: list, metric_name: str):
    print(f"\n=== {title} (per step {metric_name}) ===")
    for t in taus:
        v = metric_tau[t]
        if np.isfinite(v):
            print(f"  step {t+1:02d}: {metric_name}={v:.6f} ")
        else:
            print(f"  step {t+1:02d}: {metric_name}=nan ")


def summarize_over_taus(metric_tau: list, taus: list):
    vals = [metric_tau[t] for t in taus if np.isfinite(metric_tau[t])]
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), int(len(vals))



def compute_n_step_rmse_nll(
    transformer,
    sc_model,
    dataset,
    scaling_params: Dict[str, Any],
    horizon: int,
    layout: dict,
    device: str,
    eval_taus: list = None,
    reliability_models=None,
    R_min: float = 1e-4,
    process_noise_sc: float = 1e-6,
    max_eval_windows: Optional[int] = None,
    unscale: bool = True,
    rmse_mode: str = "flatten",  # "flatten" or "per_dim"
    prior_var_scale2: np.ndarray = None,
    sc_var_scale2: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Evaluation function for multi-step counterfactual predictions.
    
    Evaluates 3 models: PRIOR (Transformer), EVIDENCE (SC), FUSED (CI)
    
    Args:
        transformer: Trained Transformer model
        sc_model: Trained SC model (DTSC or AttentionSC)
        dataset: Test dataset (test_cf_treatment_seq)
        scaling_params: Scaling parameters
        horizon: Prediction horizon
        layout: Token layout dict
        device: Device for computation
        eval_taus: List of 0-based tau indices to evaluate (None = all steps)
        reliability_models: Reliability calibration models
        R_min: Minimum reliability value
        process_noise_sc: Process noise for SC
        max_eval_windows: Maximum windows to evaluate (None = all)
        unscale: If True, compute RMSE in unscaled (original) space
                 If False, compute RMSE in scaled space
        rmse_mode: "flatten" for MSE-then-sqrt,
                   "per_dim" for RMSE-per-dim-then-average
        prior_var_scale2: Variance calibration for prior (H, Dy)
        sc_var_scale2: Variance calibration for SC (H, Dy)
        
    Returns:
        Dictionary with RMSE, NLL, and diagnostics for 3 models
    """
    from train_twin_mimic import collect_gaussian_forecasts, get_future_past_split
    
    logger.info("=" * 80)
    logger.info(f"Multi-Step Counterfactual Evaluation")
    logger.info("=" * 80)
    
    H = int(horizon)
    if eval_taus is None:
        eval_taus = list(range(H))
    taus = sorted(set(int(t) for t in eval_taus if 0 <= int(t) < H))
    
    # Collect forecasts for 3 models
    pack = collect_gaussian_forecasts(
        transformer=transformer,
        sc_model=sc_model,
        subset=dataset,
        horizon=H,
        layout=layout,
        max_windows=max_eval_windows,
        device=device,
        reliability_models=reliability_models,
        R_min=R_min,
        process_noise_sc=process_noise_sc,
        prior_var_scale2=prior_var_scale2,
        sc_var_scale2=sc_var_scale2,
    )
    
    Dy = pack["y"].shape[-1]
    
    nan_idx = np.unique(np.where(np.isnan(pack["y"]))[0])
    not_nan = np.array([i for i in range(pack["y"].shape[0]) if i not in nan_idx])

    if unscale:
        out_mean, out_std = _prep_out_scaling(scaling_params, Dy)
        y = unscale_y(pack["y"], out_mean, out_std)
        mu_T, var_T = unscale_gaussian(pack["mu_T"], pack["var_T"], out_mean, out_std)
        mu_SC, var_SC = unscale_gaussian(pack["mu_SC"], pack["var_SC"], out_mean, out_std)
        mu_F, var_F = unscale_gaussian(pack["mu_F"], pack["var_F"], out_mean, out_std)
    else:
        # Keep in scaled space 
        y = pack["y"]
        mu_T, var_T = pack["mu_T"], pack["var_T"]
        mu_SC, var_SC = pack["mu_SC"], pack["var_SC"]
        mu_F, var_F = pack["mu_F"], pack["var_F"]
    
    mask = pack["mask"]
    y_filtered = y[not_nan]
    mu_T_filtered = mu_T[not_nan]
    mu_SC_filtered = mu_SC[not_nan]
    mu_F_filtered = mu_F[not_nan]
    mask_filtered = mask[not_nan]
    norm_const = 1.0

    if rmse_mode == "flatten":
        rmse_T = masked_rmse_per_tau(mu_T_filtered, y_filtered, mask_filtered, H, norm_const)
        rmse_SC = masked_rmse_per_tau(mu_SC_filtered, y_filtered, mask_filtered, H, norm_const)
        rmse_F = masked_rmse_per_tau(mu_F_filtered, y_filtered, mask_filtered, H, norm_const)
    else:
        rmse_T = masked_rmse_per_tau_avg_dims(mu_T_filtered, y_filtered, mask_filtered, H)
        rmse_SC = masked_rmse_per_tau_avg_dims(mu_SC_filtered, y_filtered, mask_filtered, H)
        rmse_F = masked_rmse_per_tau_avg_dims(mu_F_filtered, y_filtered, mask_filtered, H)
        
    # Compute NLL
    nll_T = masked_gaussian_nll_per_tau_avg_dims(mu_T, var_T, y, mask, H, include_const=True)
    nll_SC = masked_gaussian_nll_per_tau_avg_dims(mu_SC, var_SC, y, mask, H, include_const=True)
    nll_F = masked_gaussian_nll_per_tau_avg_dims(mu_F, var_F, y, mask, H, include_const=True)
    
    # Print per-step metrics
    get_stepwise_metric("PRIOR (Transformer)", rmse_T, taus, "RMSE")
    get_stepwise_metric("EVIDENCE (Dynamic SC)", rmse_SC, taus, "RMSE")
    get_stepwise_metric("FUSED (CI)", rmse_F, taus, "RMSE")
    
    get_stepwise_metric("PRIOR (Transformer)", nll_T, taus, "NLL")
    get_stepwise_metric("EVIDENCE (Dynamic SC)", nll_SC, taus, "NLL")
    get_stepwise_metric("FUSED (CI)", nll_F, taus, "NLL")
    
    # Summarize over steps
    rmse_T_avg, nT = summarize_over_taus(rmse_T, taus)
    rmse_SC_avg, nSC = summarize_over_taus(rmse_SC, taus)
    rmse_F_avg, nF = summarize_over_taus(rmse_F, taus)
    
    nll_T_avg, _ = summarize_over_taus(nll_T, taus)
    nll_SC_avg, _ = summarize_over_taus(nll_SC, taus)
    nll_F_avg, _ = summarize_over_taus(nll_F, taus)
    
    logger.info("\n=== Summary over selected steps ===")
    logger.info(f"  PRIOR   : mean RMSE={rmse_T_avg:.6f} | mean NLL={nll_T_avg:.6f} | steps_count={nT}")
    logger.info(f"  EVIDENCE: mean RMSE={rmse_SC_avg:.6f} | mean NLL={nll_SC_avg:.6f} | steps_count={nSC}")
    logger.info(f"  FUSED   : mean RMSE={rmse_F_avg:.6f} | mean NLL={nll_F_avg:.6f} | steps_count={nF}")

    per_trajectory_rmse_prior = masked_rmse_per_traj(mu_T_filtered, y_filtered, mask_filtered, norm_const)
    per_trajectory_rmse_sc = masked_rmse_per_traj(mu_SC_filtered, y_filtered, mask_filtered, norm_const)
    per_trajectory_rmse_fused = masked_rmse_per_traj(mu_F_filtered, y_filtered, mask_filtered, norm_const)
    split_times = get_future_past_split(dataset, horizon=H)

    
    results = {
        'rmse_prior_per_step': rmse_T,
        'rmse_sc_per_step': rmse_SC,
        'rmse_fused_per_step': rmse_F,
        'nll_prior_per_step': nll_T,
        'nll_sc_per_step': nll_SC,
        'nll_fused_per_step': nll_F,
        'rmse_prior_avg': rmse_T_avg,
        'rmse_sc_avg': rmse_SC_avg,
        'rmse_fused_avg': rmse_F_avg,
        'nll_prior_avg': nll_T_avg,
        'nll_sc_avg': nll_SC_avg,
        'nll_fused_avg': nll_F_avg,
        'taus': taus,
        'predictions': {
            'prior_mean': mu_T,
            'sc_mean': mu_SC,
            'fused_mean': mu_F,
        },
        'ground_truth': y,
        'mask': mask,
        'per_trajectory_rmse_prior': per_trajectory_rmse_prior,
        'per_trajectory_rmse_sc': per_trajectory_rmse_sc,
        'per_trajectory_rmse_fused': per_trajectory_rmse_fused,
        'split_times': split_times,
    }
    
    return results


def compute_n_step_rmse_nll_cancer(
    transformer,
    sc_model,
    dataset,
    scaling_params: Dict[str, Any],
    horizon: int,
    layout: dict,
    device: str,
    eval_taus: list = None,
    reliability_models=None,
    R_min: float = 1e-4,
    process_noise_sc: float = 1e-6,
    max_eval_windows: Optional[int] = None,
    unscale: bool = True,
    norm_const: float = 1.0,
    percentage: bool = False,
    rmse_mode: str = "flatten",
    prior_var_scale2: np.ndarray = None,
    sc_var_scale2: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Cancer simulation evaluation function for multi-step counterfactual predictions.
    
    Evaluates 3 models: PRIOR (Transformer), EVIDENCE (SC), FUSED (CI)
    
    Args:
        transformer: Trained Transformer model
        sc_model: Trained SC model (DTSC or AttentionSC)
        dataset: Test dataset (test_cf_treatment_seq)
        scaling_params: Scaling parameters
        horizon: Prediction horizon
        layout: Token layout dict
        device: Device for computation
        eval_taus: List of 0-based tau indices to evaluate (None = all steps)
        reliability_models: Reliability calibration models
        R_min: Minimum reliability value
        process_noise_sc: Process noise for SC
        max_eval_windows: Maximum windows to evaluate (None = all)
        unscale: If True, compute RMSE in unscaled (original) space
        norm_const: Normalization constant (TUMOUR_DEATH_THRESHOLD)
        percentage: If True, multiply RMSE by 100 
        rmse_mode: "flatten" for MSE-then-sqrt,
                   "per_dim" for RMSE-per-dim-then-average
        prior_var_scale2: Variance calibration for prior (H, Dy)
        sc_var_scale2: Variance calibration for SC (H, Dy)
        
    Returns:
        Dictionary with RMSE, NLL, and diagnostics for 3 models
    """
    # Import cancer-specific collection function
    from train_twin_cancer import collect_gaussian_forecasts
    from train_twin_mimic import get_future_past_split
    
    logger.info("=" * 80)
    logger.info(f"Multi-Step Counterfactual Evaluation")
    logger.info("=" * 80)
    
    H = int(horizon)
    if eval_taus is None:
        eval_taus = list(range(H))
    taus = sorted(set(int(t) for t in eval_taus if 0 <= int(t) < H))
    
    # Collect forecasts using cancer-specific function
    pack = collect_gaussian_forecasts(
        transformer=transformer,
        sc_model=sc_model,
        subset=dataset,
        horizon=H,
        layout=layout,
        max_windows=max_eval_windows,
        device=device,
        reliability_models=reliability_models,
        R_min=R_min,
        process_noise_sc=process_noise_sc,
        prior_var_scale2=prior_var_scale2,
        sc_var_scale2=sc_var_scale2,
    )
    
    Dy = pack["y"].shape[-1]
    
    nan_idx = np.unique(np.where(np.isnan(pack["y"]))[0])
    not_nan = np.array([i for i in range(pack["y"].shape[0]) if i not in nan_idx])

    if unscale:
        out_mean, out_std = _prep_out_scaling(scaling_params, Dy)
        y = unscale_y(pack["y"], out_mean, out_std)
        mu_T, var_T = unscale_gaussian(pack["mu_T"], pack["var_T"], out_mean, out_std)
        mu_SC, var_SC = unscale_gaussian(pack["mu_SC"], pack["var_SC"], out_mean, out_std)
        mu_F, var_F = unscale_gaussian(pack["mu_F"], pack["var_F"], out_mean, out_std)
    else:
        # Keep in scaled space
        y = pack["y"]
        mu_T, var_T = pack["mu_T"], pack["var_T"]
        mu_SC, var_SC = pack["mu_SC"], pack["var_SC"]
        mu_F, var_F = pack["mu_F"], pack["var_F"]
    
    mask = pack["mask"]
    
    # Filter out NaN trajectories (consistent with baseline)
    y_filtered = y[not_nan]
    mu_T_filtered = mu_T[not_nan]
    mu_SC_filtered = mu_SC[not_nan]
    mu_F_filtered = mu_F[not_nan]
    mask_filtered = mask[not_nan]

    # Compute RMSE (normalize by norm_const, then apply percentage if needed)
    if rmse_mode == "flatten":
        rmse_T = masked_rmse_per_tau(mu_T_filtered, y_filtered, mask_filtered, H, norm_const)
        rmse_SC = masked_rmse_per_tau(mu_SC_filtered, y_filtered, mask_filtered, H, norm_const)
        rmse_F = masked_rmse_per_tau(mu_F_filtered, y_filtered, mask_filtered, H, norm_const)
    else:
        rmse_T = masked_rmse_per_tau_avg_dims(mu_T_filtered, y_filtered, mask_filtered, H)
        rmse_SC = masked_rmse_per_tau_avg_dims(mu_SC_filtered, y_filtered, mask_filtered, H)
        rmse_F = masked_rmse_per_tau_avg_dims(mu_F_filtered, y_filtered, mask_filtered, H)
    
    # Apply percentage scaling if requested (matches baseline behavior)
    if percentage:
        rmse_T = [v * 100.0 if np.isfinite(v) else v for v in rmse_T]
        rmse_SC = [v * 100.0 if np.isfinite(v) else v for v in rmse_SC]
        rmse_F = [v * 100.0 if np.isfinite(v) else v for v in rmse_F]
        
    # Compute NLL (always in unscaled space for proper calibration)
    nll_T = masked_gaussian_nll_per_tau_avg_dims(mu_T, var_T, y, mask, H, include_const=True)
    nll_SC = masked_gaussian_nll_per_tau_avg_dims(mu_SC, var_SC, y, mask, H, include_const=True)
    nll_F = masked_gaussian_nll_per_tau_avg_dims(mu_F, var_F, y, mask, H, include_const=True)
    
    # Print per-step metrics
    get_stepwise_metric("PRIOR (Transformer)", rmse_T, taus, "RMSE")
    get_stepwise_metric("EVIDENCE (Dynamic SC)", rmse_SC, taus, "RMSE")
    get_stepwise_metric("FUSED (CI)", rmse_F, taus, "RMSE")
    
    get_stepwise_metric("PRIOR (Transformer)", nll_T, taus, "NLL")
    get_stepwise_metric("EVIDENCE (Dynamic SC)", nll_SC, taus, "NLL")
    get_stepwise_metric("FUSED (CI)", nll_F, taus, "NLL")
    
    # Summarize over steps
    rmse_T_avg, nT = summarize_over_taus(rmse_T, taus)
    rmse_SC_avg, nSC = summarize_over_taus(rmse_SC, taus)
    rmse_F_avg, nF = summarize_over_taus(rmse_F, taus)
    
    nll_T_avg, _ = summarize_over_taus(nll_T, taus)
    nll_SC_avg, _ = summarize_over_taus(nll_SC, taus)
    nll_F_avg, _ = summarize_over_taus(nll_F, taus)
    
    logger.info("\n=== Summary over selected steps ===")
    logger.info(f"  PRIOR   : mean RMSE={rmse_T_avg:.6f} | mean NLL={nll_T_avg:.6f} | steps_count={nT}")
    logger.info(f"  EVIDENCE: mean RMSE={rmse_SC_avg:.6f} | mean NLL={nll_SC_avg:.6f} | steps_count={nSC}")
    logger.info(f"  FUSED   : mean RMSE={rmse_F_avg:.6f} | mean NLL={nll_F_avg:.6f} | steps_count={nF}")
    logger.info(f"\nEvaluated {not_nan.size} windows (skipped {nan_idx.size} with NaN)")
    logger.info(f"Horizon: {H}, Taus: {taus}")
    logger.info(f"Unscale: {unscale}, Norm_const: {norm_const:.2f}, Percentage: {percentage}, RMSE mode: {rmse_mode}")

    # Per-trajectory RMSE
    per_trajectory_rmse_prior = masked_rmse_per_traj(mu_T_filtered, y_filtered, mask_filtered, norm_const)
    per_trajectory_rmse_sc = masked_rmse_per_traj(mu_SC_filtered, y_filtered, mask_filtered, norm_const)
    per_trajectory_rmse_fused = masked_rmse_per_traj(mu_F_filtered, y_filtered, mask_filtered, norm_const)
    split_times = get_future_past_split(dataset, horizon=H)
    
    results = {
        'rmse_prior_per_step': rmse_T,
        'rmse_sc_per_step': rmse_SC,
        'rmse_fused_per_step': rmse_F,
        'nll_prior_per_step': nll_T,
        'nll_sc_per_step': nll_SC,
        'nll_fused_per_step': nll_F,
        'rmse_prior_avg': rmse_T_avg,
        'rmse_sc_avg': rmse_SC_avg,
        'rmse_fused_avg': rmse_F_avg,
        'nll_prior_avg': nll_T_avg,
        'nll_sc_avg': nll_SC_avg,
        'nll_fused_avg': nll_F_avg,
        'taus': taus,
        'predictions': {
            'prior_mean': mu_T,
            'sc_mean': mu_SC,
            'fused_mean': mu_F,
        },
        'ground_truth': y,
        'mask': mask,
        'per_trajectory_rmse_prior': per_trajectory_rmse_prior,
        'per_trajectory_rmse_sc': per_trajectory_rmse_sc,
        'per_trajectory_rmse_fused': per_trajectory_rmse_fused,
        'split_times': split_times,
    }
    
    return results
