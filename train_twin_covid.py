#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_covid.train_twin_covid

Training script for Twin Network on COVID-19 simulation data.

    - outcome: I
    - treatment: intervention indicator
    - no static features
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from pytorch_lightning.utilities.seed import seed_everything  # type: ignore
except Exception:
    from lightning_fabric.utilities.seed import seed_everything  # type: ignore
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import COVID-19 specific modules
from torch.utils.data import Dataset

# Import exp_mimic SC+fusion code
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'exp_mimic'))
from train_twin_mimic import (
    AttentionSC_MultiOutcome,
    GaussianCausalTransformer,
    compute_token_slices,
    train_gaussian_prior,
    fit_reliability_models,
    collect_gaussian_forecasts,
    fit_var_scale2_per_tau,
    parse_taus_arg,
)

from generate_covid_data import (
    POP_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
    WEEKS,
    DAYS_PER_WEEK,
    WEEKLY_ACTION_MODE,
    WEEKLY_ACTION_P,
    WEEKLY_STATE_MODE,
    generate_covid_data_weekly,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# Utilities
# =============================================================================


def build_xy_from_states_actions(states: np.ndarray, actions: np.ndarray):
    """Create model-ready x,y,mask from weekly states/actions.

    x = [prev_I, action_onehot(2)] = 3D
    y = [I] (infected proportion)
    
    COVID data is already in proportions [0, 1].
    """
    outcomes = states[:, :, 2:3]  # I only

    prev_outcomes = np.zeros_like(outcomes)
    prev_outcomes[:, 1:, :] = outcomes[:, :-1, :]

    action_onehot = np.concatenate([1.0 - actions, actions], axis=2).astype(np.float32)

    x = np.concatenate([prev_outcomes, action_onehot], axis=-1).astype(np.float32)
    y = outcomes.astype(np.float32)
    mask = np.ones((states.shape[0], states.shape[1]), dtype=np.float32)

    return x, y, mask


class NumpyTimeSeriesDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray, unscaled_y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        self.mask = torch.from_numpy(mask).float()
        self.unscaled_y = torch.from_numpy(unscaled_y).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask[idx]


def build_causal_mask(T: int, device: torch.device):
    """Build causal mask for transformer (upper triangular)."""
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


def mask_to_padding(mask_bt: torch.Tensor) -> torch.Tensor:
    """Convert validity mask to padding mask for transformer."""
    if mask_bt.dim() == 3:
        mask_bt = mask_bt[..., 0]
    mask_bt = mask_bt.float()
    B, T = mask_bt.shape
    lengths = mask_bt.sum(dim=1).long().clamp(min=0, max=T)
    ar = torch.arange(T, device=mask_bt.device).unsqueeze(0).expand(B, T)
    pad = ar >= lengths.unsqueeze(1)
    return pad


def gaussian_nll_loss(pred_mean, pred_logvar, target, mask):
    """Gaussian negative log-likelihood loss."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    
    mse = (target - pred_mean) ** 2
    nll = 0.5 * (np.log(2 * np.pi) + pred_logvar + mse / torch.exp(pred_logvar))
    
    nll_masked = nll * mask
    loss = nll_masked.sum() / mask.sum()
    
    return loss


def train_epoch(model, loader, optimizer, device, clip_grad=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for x, y, mask in loader:
        x = x.to(device)      # (B, T, input_dim)
        y = y.to(device)      # (B, T, dim_outcomes)
        mask = mask.to(device)  # (B, T)
        
        # Forward pass
        pred_mean, pred_logvar = model(x, mask)
        
        # Compute loss
        loss = gaussian_nll_loss(pred_mean, pred_logvar, y, mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate_epoch(model, loader, device):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            
            pred_mean, pred_logvar = model(x, mask)
            loss = gaussian_nll_loss(pred_mean, pred_logvar, y, mask)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


@torch.no_grad()
def factual_rollout(model: nn.Module,
                    x0: torch.Tensor,
                    y_true: torch.Tensor,
                    mask: torch.Tensor,
                    device: str):
    """Factual rollout using true actions but predicted previous outputs.

    Args:
        x0: (T,3) with [prev_I(1), action_onehot(2)] - NO VITALS
        y_true: (T,1) ground truth normalized outputs
        mask: (T,)

    Returns:
        y_pred: (T,1) predicted normalized outputs
    """
    model.eval()

    x_roll = x0.clone()
    T = x_roll.shape[0]
    y_pred = torch.zeros((T, y_true.shape[-1]), dtype=torch.float32)

    x_roll = x_roll.to(device)
    y_true = y_true.to(device)
    mask = mask.to(device)

    for t in range(T):
        xb = x_roll.unsqueeze(0)  # (1,T,3)
        mb = mask.unsqueeze(0)    # (1,T)
        mean, _logvar = model(xb, mb)
        y_pred[t] = mean[0, t]

        if t + 1 < T:
            x_roll[t + 1, 0] = y_pred[t, 0]

    return y_pred.detach().cpu()


@torch.no_grad()
def rollout_with_actions(model: nn.Module,
                         x0: torch.Tensor,
                         actions: torch.Tensor,
                         y_dim: int,
                         device: str):
    """Autoregressive rollout under provided action sequence.

    Uses vitals from x0 (teacher-forced), replaces action channel with `actions`,
    and feeds back predicted I as prev_I.

    Args:
        x0: (T,5) baseline features (normalized)
        actions: (T,) or (T,1)
        y_dim: outcome dim (1)
    """
    model.eval()
    x_roll = x0.clone()
    T = x_roll.shape[0]

    a = actions
    if a.dim() == 1:
        a = a.unsqueeze(-1)
    a = a.to(x_roll.dtype)

    x_roll[:, 1:2] = a

    y_pred = torch.zeros((T, y_dim), dtype=torch.float32)
    x_roll = x_roll.to(device)
    for t in range(T):
        xb = x_roll.unsqueeze(0)
        mb = torch.ones((1, T), device=device, dtype=torch.float32)
        mean, _logvar = model(xb, mb)
        y_pred[t] = mean[0, t]
        if t + 1 < T:
            x_roll[t + 1, 0] = y_pred[t, 0]
    return y_pred.detach().cpu()


def save_rollout_plots(output_dir: Path,
                       traj_idx: int,
                       y_true: np.ndarray,
                       pred_dict: dict,
                       actions_true: np.ndarray,
                       title_prefix: str = "test"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comp_names = ["I"]
    colors = ["#d62728"]
    T = y_true.shape[0]
    x_axis = np.arange(1, T + 1)

    def _shade(ax, a, alpha=0.12, color="0.6"):
        active = (np.asarray(a).reshape(-1) > 0).astype(bool)
        for t, on in enumerate(active, start=1):
            if on:
                ax.axvspan(t - 0.5, t + 0.5, color=color, alpha=alpha, lw=0)

    fig, ax = plt.subplots(1, 1, figsize=(11, 4.2), sharex=True)

    ax.plot(x_axis, y_true[:, 0], color=colors[0], lw=2.5, label="GT I")
    for name, y_pred in pred_dict.items():
        ax.plot(x_axis, y_pred[:, 0], lw=2.2, ls="--", label=name)
    _shade(ax, actions_true)
    ax.set_xlabel("Time step")
    ax.set_ylabel("I (proportion)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.suptitle(f"{title_prefix} rollout traj {traj_idx} (shaded=factual intervention)")
    fig.tight_layout()
    fig.savefig(output_dir / f"{title_prefix}_rollout_traj_{traj_idx}.png", dpi=150)
    plt.close(fig)


def save_fusion_window_plots(
    output_dir: Path,
    idx_list: list,
    pack: dict,
    scaling_params: dict,
    horizon: int,
    n_sigma: float = 1.0,
    title_prefix: str = "fusion",
    subset=None,
):
    """Plot Prior vs SC vs Fused predictions with uncertainties."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mu_T, var_T = pack["mu_T"].copy(), pack["var_T"].copy()
    mu_SC, var_SC = pack["mu_SC"].copy(), pack["var_SC"].copy()
    mu_F, var_F = pack["mu_F"].copy(), pack["var_F"].copy()
    y_u = pack["y"].copy()
    mask = np.asarray(pack["mask"], dtype=float)
    
    history_data = None
    sequence_lengths = None
    intervention_data = None
    if subset is not None and hasattr(subset, 'data_original') and subset.data_original is not None:
        history_data = subset.data_original.get('history_data', None)
        sequence_lengths = subset.data_original.get('sequence_lengths', None)
        intervention_data = subset.data_original.get('intervention_data', None)

    for k, widx in enumerate(idx_list):
        widx = int(widx)
        if widx < 0 or widx >= y_u.shape[0]:
            logger.warning(f"Window index {widx} out of bounds, skipping")
            continue

        y_w = y_u[widx, :horizon, 0]
        m_w = mask[widx, :horizon, 0] if mask.ndim == 3 else mask[widx, :horizon]
        valid_T = int(np.sum(m_w > 0))
        
        if valid_T <= 1:
            logger.warning(f"Window {widx}: insufficient valid timesteps, skipping")
            continue

        def _band(mu, var):
            mu_w = mu[widx, :valid_T, 0]
            sd_w = np.sqrt(np.maximum(var[widx, :valid_T, 0], 1e-8))
            lo = mu_w - float(n_sigma) * sd_w
            hi = mu_w + float(n_sigma) * sd_w
            return mu_w, lo, hi

        muT, loT, hiT = _band(mu_T, var_T)
        muS, loS, hiS = _band(mu_SC, var_SC)
        muF, loF, hiF = _band(mu_F, var_F)

        # Get history and interventions from exploded window if available
        history = None
        history_len = 0
        interventions = None
        if history_data is not None and sequence_lengths is not None:
            seq_len = int(sequence_lengths[widx])
            history_len = max(0, seq_len - horizon)
            if history_len > 0:
                # Extend history to include t0 to fill gap
                history = history_data[widx, :history_len + 1, 0]  # Include t0
            if intervention_data is not None:
                interventions = intervention_data[widx, :seq_len]
        
        # Colors matching vis_patients in vis_func.py
        color_history = "black"
        color_gt_future = "#525354"
        color_fused = "#E52716"
        color_intervention = "#D1D3D4"  # Same grey as vis_patients treatment heatmap

        fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
        
        # Remove top and right borders, only show x and y axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # X-axis: Absolute week numbers starting from 0
        # History: weeks 0 to history_len-1
        # Forecast: weeks history_len to history_len+valid_T-1
        
        # Shade intervention periods (using vis_patients grey color #D1D3D4)
        if interventions is not None:
            for t_idx in range(history_len, min(len(interventions), history_len + valid_T)):
                if interventions[t_idx] > 0.5:
                    # Absolute week number
                    week_num = t_idx
                    ax.axvspan(week_num - 0.5, week_num + 0.5, color=color_intervention, alpha=0.5, lw=0, zorder=0)
        
        # Plot history: black line with dots (absolute weeks 0 to history_len-1)
        if history is not None and history_len > 0:
            x_history = np.arange(0, history_len)  # Absolute weeks 0 to history_len-1
            # History array has history_len+1 points, use the first history_len points
            history_vals = history[:history_len]
            ax.plot(x_history, history_vals, 'o-', color=color_history, linewidth=1.5, 
                   markersize=4, alpha=0.7, label="History" if k == 0 else "", zorder=5)
        
        # Plot split line: light green dotted line at the boundary between history and forecast
        if history_len > 0:
            ax.axvline(x=history_len - 0.5, color='lightgreen', linestyle=':', linewidth=1.5, zorder=4)
        
        # Plot future ground truth: dashed dark grey line with dots
        x_forecast = np.arange(history_len, history_len + valid_T)  # Absolute weeks
        ax.plot(x_forecast, y_w[:valid_T], 'o--', color=color_gt_future, linewidth=2.0,
               markersize=5, label="Ground truth (CF)" if k == 0 else "", zorder=5)

        # Plot only Fused predictions (removed Transformer and SC)
        ax.plot(x_forecast, muF, color=color_fused, linewidth=1.8, label="Fused" if k == 0 else "", zorder=3)
        ax.fill_between(x_forecast, loF, hiF, color='#F7BBB6', alpha=0.8, lw=0, zorder=2)

        # Set x-axis labels: Absolute week numbers starting from 0
        total_weeks = history_len + valid_T
        x_positions = list(range(0, total_weeks))
        week_labels = [str(w) for w in x_positions]
        
        # Show ticks, but limit if too many
        if len(x_positions) > 20:
            step = max(1, len(x_positions) // 15)
            x_ticks = x_positions[::step]
            week_labels_ticks = [week_labels[i] for i in range(0, len(week_labels), step)]
        else:
            x_ticks = x_positions
            week_labels_ticks = week_labels
        
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(week_labels_ticks)

        ax.set_xlabel("Week")
        ax.set_ylabel("I proportion")
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='best', framealpha=0.95)
        
        fig.suptitle(f"Window {widx} (history={history_len}, forecast={valid_T})")
        fig.tight_layout()
        
        plot_path = output_dir / f"{title_prefix}_window_{widx}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {plot_path.name}")


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Twin Network on COVID-19 data')
    # =============================================================================
    # Data generation
    # ============================================================================= happens on-the-fly (no pre-generated pickles needed)
    parser.add_argument('--weeks', type=int, default=WEEKS, help='Number of weeks per trajectory')
    parser.add_argument('--days_per_week', type=int, default=DAYS_PER_WEEK, help='Days per week (internal sim)')
    parser.add_argument('--weekly_state_mode', type=str, default=WEEKLY_STATE_MODE, choices=['eow', 'mean'])
    parser.add_argument('--weekly_action_mode', type=str, default=WEEKLY_ACTION_MODE, choices=['mean', 'max'])
    parser.add_argument('--weekly_action_p', type=float, default=WEEKLY_ACTION_P, help='P(action=1) per week')
    parser.add_argument('--beta_change', type=float, default=0.25, help='Beta multiplier when action=1 (0.5 = reduce to 50%)')
    parser.add_argument('--pop_size', type=int, default=POP_SIZE)
    parser.add_argument('--train_size', type=int, default=TRAIN_SIZE)
    parser.add_argument('--val_size', type=int, default=VAL_SIZE)
    parser.add_argument('--test_size', type=int, default=TEST_SIZE)
    parser.add_argument('--no_multiprocess', action='store_true', help='Disable multiprocessing for Covasim sims')

    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Transformer model dimension')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping threshold')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize outputs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')

    parser.add_argument('--plot_n', type=int, default=3,
                        help='Number of rollout plots (factual+CF) to save after training')

    # exp_mimic-style SC + fusion (keep same components, only adapt data schema)
    parser.add_argument('--horizon', type=int, default=12,
                        help='Prediction horizon for SC/fusion windows (default: weeks//2 for balanced history/forecast)')
    parser.add_argument('--max_calib_windows', type=int, default=800)
    parser.add_argument('--max_eval_windows', type=int, default=None)
    parser.add_argument('--calib_taus', type=str, default='all')
    parser.add_argument('--eval_taus', type=str, default='all')
    parser.add_argument('--plot_fusion_examples', type=int, default=5,
                        help='Number of SC/fusion windows to plot (randomly sampled from available windows, 0 disables)')
    parser.add_argument('--plot_n_sigma', type=float, default=1.0)
    parser.add_argument('--R_min', type=float, default=1e-4)
    parser.add_argument('--process_noise_sc', type=float, default=1e-6)

    # Attention SC hyperparams (use mimic defaults)
    parser.add_argument('--attn_epochs', type=int, default=10)
    parser.add_argument('--attn_batch_size', type=int, default=256)
    parser.add_argument('--attn_lr', type=float, default=2e-3)
    parser.add_argument('--knn', type=int, default=50)
    parser.add_argument('--tau_dist', type=float, default=0.25)
    parser.add_argument('--zstd_floor', type=float, default=5e-2)
    parser.add_argument('--attn_emb_dim', type=int, default=64)
    parser.add_argument('--attn_hidden', type=int, default=256)
    parser.add_argument('--attn_dropout', type=float, default=0.1)
    parser.add_argument('--attn_temp', type=float, default=0.2)
    parser.add_argument('--attn_candidates', type=int, default=128)
    parser.add_argument('--attn_steps_per_epoch', type=int, default=1500)
    parser.add_argument('--attn_entropy_reg', type=float, default=0.0)

    # Reliability calibration
    parser.add_argument('--calib_epochs', type=int, default=250)
    parser.add_argument('--calib_lr', type=float, default=5e-2)
    
    args = parser.parse_args()
    
    # Set seed
    seed_everything(args.seed)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("COVID-19 TWIN NETWORK TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data: generated on-the-fly")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Attention heads: {args.n_heads}")
    print(f"  Transformer layers: {args.num_layers}")
    print(f"  Device: {args.device}")
    print()
    
    # =============================================================================
    # Generate Data
    # =============================================================================
    logger.info("Generating COVID-19 data on-the-fly...")

    train_states, train_actions = generate_covid_data_weekly(
        num_trajectories=int(args.train_size),
        weeks=int(args.weeks),
        pop_size=int(args.pop_size),
        multi_process=not bool(args.no_multiprocess),
        seed=int(args.seed),
        weekly_actions=None,
        days_per_week=int(args.days_per_week),
        state_mode=str(args.weekly_state_mode),
        action_mode=str(args.weekly_action_mode),
        action_p=float(args.weekly_action_p),
        beta_change=float(args.beta_change),
    )
    val_states, val_actions = generate_covid_data_weekly(
        num_trajectories=int(args.val_size),
        weeks=int(args.weeks),
        pop_size=int(args.pop_size),
        multi_process=not bool(args.no_multiprocess),
        seed=int(args.seed) + 1,
        weekly_actions=None,
        days_per_week=int(args.days_per_week),
        state_mode=str(args.weekly_state_mode),
        action_mode=str(args.weekly_action_mode),
        action_p=float(args.weekly_action_p),
        beta_change=float(args.beta_change),
    )
    test_states, test_actions = generate_covid_data_weekly(
        num_trajectories=int(args.test_size),
        weeks=int(args.weeks),
        pop_size=int(args.pop_size),
        multi_process=not bool(args.no_multiprocess),
        seed=int(args.seed) + 2,
        weekly_actions=None,
        days_per_week=int(args.days_per_week),
        state_mode=str(args.weekly_state_mode),
        action_mode=str(args.weekly_action_mode),
        action_p=float(args.weekly_action_p),
        beta_change=float(args.beta_change),
    )

    # No normalization needed for COVID data (already proportions [0, 1])
    scaling_params = {}  # Empty dict for API compatibility
    logger.info(f"COVID data requires no normalization (already in [0,1] range)")

    x_train, y_train, m_train = build_xy_from_states_actions(train_states, train_actions)
    x_val, y_val, m_val = build_xy_from_states_actions(val_states, val_actions)
    x_test, y_test, m_test = build_xy_from_states_actions(test_states, test_actions)

    train_dataset = NumpyTimeSeriesDataset(x_train, y_train, m_train, unscaled_y=train_states[:, :, 2:3])
    val_dataset = NumpyTimeSeriesDataset(x_val, y_val, m_val, unscaled_y=val_states[:, :, 2:3])
    test_dataset = NumpyTimeSeriesDataset(x_test, y_test, m_test, unscaled_y=test_states[:, :, 2:3])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # Get dimensions
    sample_x, sample_y, _ = train_dataset[0]
    input_dim = sample_x.shape[-1]      # 3: [prev_I (1), action_onehot (2)] - NO VITALS
    output_dim = sample_y.shape[-1]     # 1: I outcome
    seq_len = sample_x.shape[0]
    
    logger.info(f"Dataset ready:")
    logger.info(f"  Train: {len(train_dataset)} trajectories")
    logger.info(f"  Val: {len(val_dataset)} trajectories")
    logger.info(f"  Test: {len(test_dataset)} trajectories")
    logger.info(f"  Input dim: {input_dim}")
    logger.info(f"  Output dim: {output_dim}")
    logger.info(f"  Sequence length: {seq_len}")
    
    from covid_dataset import CovidDatasetCollection
    
    all_states = np.concatenate([train_states, val_states, test_states], axis=0)
    all_actions = np.concatenate([train_actions, val_actions, test_actions], axis=0)
    
    total_size = len(all_states)
    train_f = len(train_states) / total_size
    val_f = len(val_states) / total_size
    
    temp_collection = CovidDatasetCollection(
        states_weekly=all_states,
        actions_weekly=all_actions,
        train_f=train_f,
        val_f=val_f
    )
    
    train_subset = temp_collection.get_dataset('train')
    val_subset = temp_collection.get_dataset('val')
    
    horizon = int(args.horizon) if args.horizon is not None else max(1, int(args.weeks) // 2)
    
    test_f = temp_collection.get_dataset('test', explode_horizon=horizon)
    
    logger.info(f"Test subset created (exploded):")
    logger.info(f"  test_f: {test_f.num_trajectories} windows from {test_f.num_base_trajectories} base trajectories")

    Dy = 1
    A_dim = 2
    D_static = 0
    D_vitals = 0
    
    test_subset_full = temp_collection.get_dataset('test')
    layout = {
        'name': 'prev_outputs+treat+static+vitals',
        'order': ('prev_outputs', 'treatments', 'static', 'vitals'),
        'D_vitals': D_vitals,
        'dim_outputs': Dy,
        'dim_treatments': A_dim,
        'dim_static': D_static,
        'dim_vitals': D_vitals,
        'input_dim': Dy + A_dim + D_static + D_vitals,
        'output_start': 0,
        'treatment_start': Dy,
        'static_start': Dy + A_dim,
        'vitals_start': Dy + A_dim + D_static,
    }

    token_slices = compute_token_slices(
        layout=layout,
        Dy=Dy,
        A_dim=A_dim,
        D_static=D_static,
        D_vitals=D_vitals,
        input_dim=layout['input_dim'],
    )

    device = str(args.device)

    transformer = GaussianCausalTransformer(
        input_dim=int(layout['input_dim']),
        dim_outcomes=int(Dy),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        num_layers=int(args.num_layers),
        dim_feedforward=int(args.d_model) * 4,
        dropout=float(args.dropout),
        max_len=int(seq_len) + 4,
    ).to(device)

    logger.info("Training transformer prior (exp_mimic)...")
    train_gaussian_prior(
        model=transformer,
        train_ds=train_subset,
        num_epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=device,
        lambda_mmd=0.0,
        treat_slice=token_slices.get('treatments'),
        A_dim=A_dim,
    )

    # =============================================================================
    # Synthetic Control (AttentionSC_MultiOutcome)
    # =============================================================================
    logger.info("Fitting Attention SC model (exp_mimic)...")

    sc_model = AttentionSC_MultiOutcome(
        Dy=Dy,
        static_pca_dim=0,
        vitals_pca_dim=0,
        lambda_vitals=0.0,
        zstd_floor=float(args.zstd_floor),
        knn=int(args.knn),
        tau_dist=float(args.tau_dist),
        attn_device=device,
        attn_epochs=int(args.attn_epochs),
        attn_lr=float(args.attn_lr),
        attn_batch_size=int(args.attn_batch_size),
        attn_candidates=int(args.attn_candidates),
        attn_steps_per_epoch=int(args.attn_steps_per_epoch),
        attn_temp=float(args.attn_temp),
        attn_entropy_reg=float(args.attn_entropy_reg),
        attn_emb_dim=int(args.attn_emb_dim),
        attn_hidden=int(args.attn_hidden),
        attn_dropout=float(args.attn_dropout),
    )
    sc_model.fit(train_subset=train_subset, device_for_training=device)

    # =============================================================================
    # Reliability calibration + variance calibration
    # =============================================================================

    logger.info(f"\nCalibrating reliability models on test_f (last {horizon} timesteps from exploded windows)...")

    reliability_models = fit_reliability_models(
        sc_model=sc_model,
        subset=test_f,
        horizon=horizon,
        taus=[t for t in parse_taus_arg(args.calib_taus, horizon=horizon)],
        max_windows=int(args.max_calib_windows),
        device=device,
        lr=float(args.calib_lr),
        epochs=int(args.calib_epochs),
        batch_size=512,
        R_min=float(args.R_min),
        eps=1e-3,
        process_noise=float(args.process_noise_sc),
        seed=123,
        sc_var_scale2=None,
    )

    pack_cal = collect_gaussian_forecasts(
        transformer=transformer,
        sc_model=sc_model,
        subset=test_f,
        horizon=horizon,
        layout=layout,
        max_windows=int(args.max_calib_windows),
        device=device,
        reliability_models=reliability_models,
        R_min=float(args.R_min),
        process_noise_sc=float(args.process_noise_sc),
        prior_var_scale2=None,
        sc_var_scale2=None,
    )

    prior_var_scale2 = fit_var_scale2_per_tau(
        mu=pack_cal['mu_T'],
        var=pack_cal['var_T'],
        y=pack_cal['y'],
        mask=pack_cal['mask'],
        H=horizon,
        clip=(0.001, 1000.0),
        min_points=10,
    )
    sc_var_scale2 = fit_var_scale2_per_tau(
        mu=pack_cal['mu_SC'],
        var=pack_cal['var_SC'],
        y=pack_cal['y'],
        mask=pack_cal['mask'],
        H=horizon,
        clip=(0.001, 1000.0),
        min_points=10,
    )

    logger.info("Finished SC+fusion calibration.")
    
    # =============================================================================
    # Visualization
    # =============================================================================
    script_dir = Path(__file__).resolve().parent
    plots_dir = script_dir / 'plots' / f"train_seed{int(args.seed)}"
    plots_dir.mkdir(parents=True, exist_ok=True)

    n_fusion = max(0, int(args.plot_fusion_examples))
    if n_fusion > 0:
        logger.info("\nCollecting forecasts for fusion plotting (test_f)...")
        pack_eval = collect_gaussian_forecasts(
            transformer=transformer,
            sc_model=sc_model,
            subset=test_f,
            horizon=horizon,
            layout=layout,
            max_windows=int(args.max_eval_windows) if args.max_eval_windows is not None else None,
            device=device,
            reliability_models=reliability_models,
            R_min=float(args.R_min),
            process_noise_sc=float(args.process_noise_sc),
            prior_var_scale2=prior_var_scale2,
            sc_var_scale2=sc_var_scale2,
        )
        # Select windows to plot (respect plot_fusion_examples parameter)
        num_available = int(pack_eval["y"].shape[0])
        if n_fusion > 0 and num_available > 0:
            # Randomly sample n_fusion windows, or plot all if n_fusion >= num_available
            if n_fusion >= num_available:
                idx_list = list(range(num_available))
                logger.info(f"Plotting ALL {num_available} available windows (requested {n_fusion})...")
            else:
                rng = np.random.default_rng(42)  # Fixed seed for reproducibility
                idx_list = sorted(rng.choice(num_available, size=n_fusion, replace=False).tolist())
                logger.info(f"Randomly sampling {n_fusion} windows from {num_available} available windows...")
            
            save_fusion_window_plots(
                output_dir=plots_dir / "fusion",
                idx_list=idx_list,
                pack=pack_eval,
                scaling_params=scaling_params,
                horizon=horizon,
                n_sigma=float(args.plot_n_sigma),
                title_prefix="fusion",
                subset=test_f,
            )
            logger.info(f"Saved {len(idx_list)} fusion plots to {plots_dir / 'fusion'}")
        else:
            logger.info(f"Skipping fusion plots (n_fusion={n_fusion}, num_available={num_available})")

    # =============================================================================
    # Save Checkpoint
    # =============================================================================
    logger.info("\nSaving checkpoint...")
    checkpoint_path = output_path / f"ckpt_seed{int(args.seed)}.pt"
    checkpoint = {
        'transformer_state_dict': transformer.state_dict(),
        'sc_model': sc_model,  # Save full SC model (has internal state)
        'reliability_models': reliability_models,
        'prior_var_scale2': prior_var_scale2,
        'sc_var_scale2': sc_var_scale2,
        'scaling_params': scaling_params,
        'layout': layout,
        'token_slices': token_slices,
        'config': {
            'weeks': args.weeks,
            'horizon': horizon,
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'seq_len': seq_len,
            'input_dim': layout['input_dim'],
            'output_dim': Dy,
            'seed': args.seed,
        },
        'args': vars(args),
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

    logger.info("\nTraining complete (prior + SC + fusion).")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Plots: {plots_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
