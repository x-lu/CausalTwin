#!/usr/bin/env python3
"""
Visualization functions for counterfactual trajectories.
"""

import os
import logging
from typing import Iterable, Optional, Sequence, Dict, Any, Tuple
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
#  Kalman fusion visualization: SC/Prior/Fused comparison
# ============================================================

def vis_kalman_fusion(
    transformer,
    sc_model,
    cf_subset,
    scaling_params,
    horizon: int,
    layout: dict,
    idx_list,
    device: str,
    reliability_models=None,
    R_min: float = 1e-4,
    process_noise_sc: float = 1e-6,
    n_sigma: float = 1.0,
    out_dim: int = 0,
    save_dir: str = "plots/kalman_fusion",
    ylim: tuple = None,
    prior_var_scale2: np.ndarray = None,
    sc_var_scale2: np.ndarray = None,
):
    """
    Plot Kalman fusion examples showing Prior, SC, and Fused predictions.
    Visualizes uncertainty bands and reliability for counterfactual predictions.
    If Dy >= 2, creates side-by-side subplots for dim 0 and dim 1.
    Uses rollout_cf_future and get_future_past_split from train_twin_att_mmd.
    
    Args:
        transformer: Trained Transformer model
        sc_model: Trained SC model
        cf_subset: Counterfactual test dataset
        scaling_params: Scaling parameters for unscaling
        horizon: Prediction horizon
        layout: Token layout dict
        idx_list: List of window indices to plot
        device: Device for computation
        reliability_models: Reliability calibration models
        R_min: Minimum reliability value
        process_noise_sc: Process noise for SC
        n_sigma: Number of standard deviations for uncertainty bands
        out_dim: Output dimension to plot (ignored if Dy >= 2, will plot both)
        save_dir: Directory to save plots
        ylim: Y-axis limits (optional)
        prior_var_scale2: Variance calibration for prior (H, Dy)
        sc_var_scale2: Variance calibration for SC (H, Dy)
    """
    # Lazy import to avoid circular dependency
    from train_twin_mimic import rollout_cf_future, get_future_past_split
    
    os.makedirs(save_dir, exist_ok=True)

    d = cf_subset.data
    Y = np.asarray(d["outputs"]).astype(float)
    seq_lens = np.asarray(d["sequence_lengths"]).astype(int)
    
    # Get patient IDs if available
    if hasattr(cf_subset, 'base_ids_original'):
        base_ids = np.array(cf_subset.base_ids_original).astype(int)
    elif 'base_ids_original' in d:
        base_ids = np.array(d['base_ids_original']).astype(int)
    else:
        base_ids = None
    
    # Get split times using imported function
    split = get_future_past_split(cf_subset, horizon=int(horizon))
    
    # Get future_past_split array if available for CF ID calculation
    if 'future_past_split' in d:
        cf_split_times = np.array(d['future_past_split']).astype(int)
    else:
        cf_split_times = None

    Dy = Y.shape[-1]
    
    # Prepare scaling parameters
    if 'output_means' in scaling_params:
        out_mean = np.asarray(scaling_params['output_means']).reshape(-1)
        out_std = np.asarray(scaling_params['output_stds']).reshape(-1)
    else:
        out_mean = np.zeros(Dy)
        out_std = np.ones(Dy)

    # Determine number of subplots (vertical stacking)
    n_subplots = min(2, Dy) if Dy >= 2 else 1
    plot_dims = list(range(n_subplots))

    for i in idx_list:
        if i < 0 or i >= Y.shape[0]:
            continue
        L = int(seq_lens[i])
        t0 = int(split[i])
        if t0 <= 0 or t0 >= L:
            continue

        H = min(int(horizon), L - t0)
        if H <= 0:
            continue
            
        fused = rollout_cf_future(
            transformer=transformer,
            sc_model=sc_model,
            subset=cf_subset,
            i=i,
            t0=t0,
            H=H,
            layout=layout,
            device=device,
            reliability_models=reliability_models,
            R_min=R_min,
            process_noise_sc=process_noise_sc,
            prior_var_scale2=prior_var_scale2,
            sc_var_scale2=sc_var_scale2,
        )
        if fused is None:
            continue

        mu_T_s = np.asarray(fused["prior_mean"], float)
        var_T_s = np.asarray(fused["prior_var"], float)
        mu_SC_s = np.asarray(fused["sc_mean"], float)
        var_SC_s = np.asarray(fused["sc_var"], float)
        mu_F_s = np.asarray(fused["fused_mean"], float)
        var_F_s = np.asarray(fused["fused_var"], float)
        r_s = np.asarray(fused["r"], float)

        y_true_s = Y[i, t0:t0 + H, :]

        if base_ids is not None:
            patient_id = int(base_ids[i])
            if cf_split_times is not None:
                patient_mask = (base_ids == patient_id)
                patient_indices = np.where(patient_mask)[0]
                patient_split_times = cf_split_times[patient_indices]
                split_mask = (patient_split_times == t0)
                trajectories_at_split = patient_indices[split_mask]
                cf_scenario_id = np.where(trajectories_at_split == i)[0][0] if i in trajectories_at_split else 0
                title_base = f"Patient {patient_id} | CF #{cf_scenario_id} | split={t0}"
            else:
                title_base = f"Patient {patient_id} | split={t0}"
        else:
            title_base = f"CF #{i} | split={t0}"

        fig, axes = plt.subplots(n_subplots, 1, figsize=(9, 4 * n_subplots))
        if n_subplots == 1:
            axes = [axes]
        
        for subplot_idx, dim_idx in enumerate(plot_dims):
            ax = axes[subplot_idx]
            
            y_true = y_true_s[:, dim_idx] * out_std[dim_idx] + out_mean[dim_idx]
            y_T = mu_T_s[:, dim_idx] * out_std[dim_idx] + out_mean[dim_idx]
            y_SC = mu_SC_s[:, dim_idx] * out_std[dim_idx] + out_mean[dim_idx]
            y_F = mu_F_s[:, dim_idx] * out_std[dim_idx] + out_mean[dim_idx]

            s_T = np.sqrt(np.maximum(var_T_s[:, dim_idx], 0.0)) * out_std[dim_idx]
            s_SC = np.sqrt(np.maximum(var_SC_s[:, dim_idx], 0.0)) * out_std[dim_idx]
            s_F = np.sqrt(np.maximum(var_F_s[:, dim_idx], 0.0)) * out_std[dim_idx]

            t = np.arange(H)
            y_anchor = float(Y[i, t0 - 1, dim_idx] * out_std[dim_idx] + out_mean[dim_idx])
            t_plot = np.concatenate([np.array([-1]), t])
            
            ax.plot(t_plot, np.concatenate([[y_anchor], y_true]), label="Ground truth (CF)", linewidth=2)
            ax.plot(t_plot, np.concatenate([[y_anchor], y_T]), label="Transformer prior", linestyle="--", linewidth=2)
            ax.fill_between(
                t_plot,
                np.concatenate([[y_anchor], y_T]) - n_sigma * np.concatenate([[0.0], s_T]),
                np.concatenate([[y_anchor], y_T]) + n_sigma * np.concatenate([[0.0], s_T]),
                alpha=0.15,
                label=f"±{n_sigma:g} std (prior)",
            )

            ax.plot(t_plot, np.concatenate([[y_anchor], y_SC]), label="SC evidence", linestyle=":", linewidth=2)
            ax.fill_between(
                t_plot,
                np.concatenate([[y_anchor], y_SC]) - n_sigma * np.concatenate([[0.0], s_SC]),
                np.concatenate([[y_anchor], y_SC]) + n_sigma * np.concatenate([[0.0], s_SC]),
                alpha=0.10,
                label=f"±{n_sigma:g} std (SC)",
            )

            ax.plot(t_plot, np.concatenate([[y_anchor], y_F]), label="Kalman fused posterior", linewidth=2)
            ax.fill_between(
                t_plot,
                np.concatenate([[y_anchor], y_F]) - n_sigma * np.concatenate([[0.0], s_F]),
                np.concatenate([[y_anchor], y_F]) + n_sigma * np.concatenate([[0.0], s_F]),
                alpha=0.20,
                label=f"±{n_sigma:g} std (fused)",
            )

            ax.set_xlabel("CF step (anchor at -1)", fontsize=13)
            ax.set_ylabel(f"Outcome dim {dim_idx} (unscaled)", fontsize=13)
            ax.set_title(f"{title_base} | dim={dim_idx}")
            ax.tick_params(axis='both', labelsize=13)
            ax.grid(alpha=0.3)
            if subplot_idx == 0:  # Only show legend on first subplot
                ax.legend(fontsize=13)
            if ylim is not None and len(ylim) == 2:
                ax.set_ylim(float(ylim[0]), float(ylim[1]))

        plt.tight_layout()

        out_path = os.path.join(save_dir, f"kalman_fusion_mimic_window_{i}.png")
        plt.savefig(out_path)
        plt.close()

        # Log reliability for all dimensions
        for dim_idx in plot_dims:
            logger.info(f"[plot window {i}] r per tau (dim{dim_idx}): {np.round(r_s[:, dim_idx], 3).tolist()}")
        logger.info(f"[plot saved] {out_path}")


# ============================================================
#  Combined patient visualization: history + CF + uncertainty
# ============================================================

def vis_patients(
    dataset,
    transformer,
    sc_model,
    layout: dict,
    scaling_params: dict,
    horizon: int,
    device: str,
    patient_ids: Optional[Sequence[int]] = None,
    cf_ids: Optional[Sequence[int]] = None,
    reliability_models=None,
    R_min: float = 1e-4,
    process_noise_sc: float = 1e-6,
    prior_var_scale2: np.ndarray = None,
    sc_var_scale2: np.ndarray = None,
    n_sigma: float = 1.0,
    output_dir: str = 'plots/patients/',
    outcome_names: Optional[Iterable[str]] = None,
    treatment_names: Optional[Iterable[str]] = None,
    num_patients: int = 6,
):
    """
    Combined visualization: historical trajectory + CF predictions + uncertainty bands + treatments.
    
    Creates a grid where each row shows one patient's one CF scenario with:
    - Column 1: Outcome dim 0 (history in black, CF ground truth + pred with uncertainty)
    - Column 2: Outcome dim 1 (same format)
    - Column 3: Treatment heatmap
    
    Args:
        dataset: Counterfactual dataset with all trajectories
        transformer: Trained Transformer model
        sc_model: Trained SC model
        layout: Token layout dict
        scaling_params: Scaling parameters for unscaling
        horizon: Prediction horizon
        device: Device for computation
        patient_ids: List of patient IDs to plot (optional)
        cf_ids: List of CF trajectory IDs, one per patient (optional, must match patient_ids length)
        reliability_models: Reliability calibration models
        R_min: Minimum reliability value
        process_noise_sc: Process noise for SC
        prior_var_scale2: Variance calibration for prior (H, Dy)
        sc_var_scale2: Variance calibration for SC (H, Dy)
        n_sigma: Number of standard deviations for uncertainty bands
        output_dir: Directory to save plots
        outcome_names: Outcome labels
        treatment_names: Treatment labels
        num_patients: Number of patients to randomly sample if patient_ids not specified
    """
    from train_twin_mimic import rollout_cf_future, get_future_past_split
    
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    if not hasattr(dataset, 'data') or not isinstance(dataset.data, dict):
        raise ValueError('Dataset must have a data dict attribute')
    data = dataset.data
    
    if hasattr(dataset, 'base_ids_original'):
        base_ids = np.array(dataset.base_ids_original).astype(int)
    elif 'base_ids_original' in data:
        base_ids = np.array(data['base_ids_original']).astype(int)
    else:
        logger.warning('No base_ids_original found - cannot filter by patient')
        return []
    
    if 'future_past_split' not in data:
        logger.warning('No future_past_split found in dataset - skipping visualization')
        return []
    
    cf_split_times = np.array(data['future_past_split']).astype(int)
    
    # Get scaling parameters
    if 'output_means' in scaling_params:
        out_mean = np.asarray(scaling_params['output_means']).reshape(-1)
        out_std = np.asarray(scaling_params['output_stds']).reshape(-1)
    else:
        out_mean = np.zeros(data['outputs'].shape[-1])
        out_std = np.ones(data['outputs'].shape[-1])
    
    Dy = data['outputs'].shape[-1]
    num_treatments = data['current_treatments'].shape[-1] if 'current_treatments' in data else 0
    outcome_labels = list(outcome_names) if outcome_names else [f'y{i}' for i in range(Dy)]
    treatment_labels = list(treatment_names) if treatment_names else [f't{i}' for i in range(num_treatments)]
    
    # Sample patients and CF trajectories if not specified
    if patient_ids is None:
        unique_patient_ids = np.unique(base_ids)
        rng = np.random.default_rng(42)
        if len(unique_patient_ids) > num_patients:
            patient_ids = rng.choice(unique_patient_ids, size=num_patients, replace=False)
        else:
            patient_ids = unique_patient_ids
        logger.info(f'Randomly sampled {len(patient_ids)} patients: {patient_ids.tolist()}')
    
    patient_ids = list(patient_ids)
    
    # For each patient, select one CF trajectory 
    selected_data_indices = []
    selected_patient_ids = []
    selected_cf_ids = []  # Track CF scenario IDs (0-9)
    
    for idx, patient_id in enumerate(patient_ids):
        # Find all trajectories for this patient
        patient_mask = (base_ids == patient_id)
        patient_indices = np.where(patient_mask)[0]
        
        if len(patient_indices) == 0:
            logger.warning(f'Patient {patient_id} not found in dataset')
            continue
        
        # Get unique split times for this patient
        patient_split_times = cf_split_times[patient_indices]
        unique_splits = np.unique(patient_split_times)
        
        # Pick one split time (first one or random)
        rng = np.random.default_rng(42 + idx)
        selected_split = unique_splits[0] if len(unique_splits) == 1 else rng.choice(unique_splits)
        
        # Get trajectories at this split time
        split_mask = (patient_split_times == selected_split)
        trajectories_at_split = patient_indices[split_mask]
        
        # Select CF trajectory: use cf_ids if specified, otherwise pick randomly
        if cf_ids is not None and idx < len(cf_ids):
            cf_id = cf_ids[idx]
            # cf_id is 0-based index into the CF scenarios at this split 
            if 0 <= cf_id < len(trajectories_at_split):
                selected_idx = trajectories_at_split[cf_id]
                logger.info(f'Patient {patient_id}: using CF scenario {cf_id} at split {selected_split}')
            else:
                logger.warning(f'Patient {patient_id}: cf_id {cf_id} out of range (0-{len(trajectories_at_split)-1}), using random selection')
                selected_idx = rng.choice(trajectories_at_split)
                cf_id = np.where(trajectories_at_split == selected_idx)[0][0]
        else:
            # Random selection
            selected_idx = rng.choice(trajectories_at_split)
            cf_id = np.where(trajectories_at_split == selected_idx)[0][0]
        
        selected_data_indices.append(selected_idx)
        selected_patient_ids.append(patient_id)
        selected_cf_ids.append(cf_id)
        logger.info(f'Patient {patient_id}: selected CF scenario {cf_id}/{len(trajectories_at_split)-1}, data_idx={selected_idx}, split_time={selected_split}')
    
    if len(selected_data_indices) == 0:
        logger.warning('No valid patient trajectories selected')
        return []
    
    # Create figure: 3 columns (outcome dim 0, outcome dim 1, treatments)
    n_patients = len(selected_data_indices)
    fig, axes = plt.subplots(n_patients, 3, figsize=(15, 3.5 * n_patients))
    if n_patients == 1:
        axes = axes.reshape(1, -1)
    
    # Get split times
    t0_arr = get_future_past_split(dataset, horizon)
    
    saved_files = []
    
    for row_idx, (data_idx, patient_id, cf_scenario_id) in enumerate(zip(selected_data_indices, selected_patient_ids, selected_cf_ids)):
        t0 = int(t0_arr[data_idx])
        H = min(horizon, data['sequence_lengths'][data_idx] - t0)
        
        if H <= 0:
            logger.warning(f'Patient {patient_id}: no future horizon available')
            continue
        
        # Get rollout with uncertainty
        results = rollout_cf_future(
            transformer=transformer,
            sc_model=sc_model,
            subset=dataset,
            i=data_idx,
            t0=t0,
            H=H,
            layout=layout,
            device=device,
            reliability_models=reliability_models,
            R_min=R_min,
            process_noise_sc=process_noise_sc,
            prior_var_scale2=prior_var_scale2,
            sc_var_scale2=sc_var_scale2,
        )
        
        if results is None:
            logger.warning(f'Patient {patient_id}: rollout failed')
            continue
        
        # Extract predictions (scaled space)
        mu_T_s = np.asarray(results["prior_mean"], float)
        var_T_s = np.asarray(results["prior_var"], float)
        mu_SC_s = np.asarray(results["sc_mean"], float)
        var_SC_s = np.asarray(results["sc_var"], float)
        mu_F_s = np.asarray(results["fused_mean"], float)
        var_F_s = np.asarray(results["fused_var"], float)
        
        # Unscale all models
        mu_T = mu_T_s * out_std + out_mean
        std_T = np.sqrt(np.maximum(var_T_s, 0.0)) * out_std
        mu_SC = mu_SC_s * out_std + out_mean
        std_SC = np.sqrt(np.maximum(var_SC_s, 0.0)) * out_std
        mu_F = mu_F_s * out_std + out_mean
        std_F = np.sqrt(np.maximum(var_F_s, 0.0)) * out_std
        
        # Get ground truth (unscaled)
        gt_seq_unscaled = data['unscaled_outputs'][data_idx] if 'unscaled_outputs' in data else (data['outputs'][data_idx] * out_std + out_mean)
        
        # Get mask for valid timesteps
        mask = data.get('active_entries')
        mask_arr = np.array(mask[data_idx]).reshape(-1) if mask is not None else None
        
        # Plot treatments
        if num_treatments > 0:
            ax = axes[row_idx, 0]
            treatments = data['current_treatments'][data_idx]
            max_time = min(t0 + H, treatments.shape[0])
            full_treatments = treatments[:max_time]
            time_range = np.arange(0, max_time)
            
            from matplotlib.colors import LinearSegmentedColormap
            colors = ['white', '#D1D3D4']
            n_bins = 100
            cmap_custom = LinearSegmentedColormap.from_list('custom_grey', colors, N=n_bins)
            
            im = ax.imshow(full_treatments.T, aspect='auto', cmap=cmap_custom, interpolation='nearest', extent=[0, max_time, -0.5, num_treatments - 0.5], vmin=0, vmax=1)
            ax.set_yticks(range(num_treatments))
            ax.set_yticklabels(treatment_labels, fontsize=13)
            ax.tick_params(axis='x', labelsize=13)
            ax.set_xlabel('Time', fontsize=13)
            ax.set_title(f'Patient {patient_id} | CF #{cf_scenario_id} - treatments', fontsize=10, fontweight='bold')
            ax.axvline(t0, color='lightgreen', linestyle=':', linewidth=2.0)
        
        # Plot outcomes
        for dim_idx in range(min(2, Dy)):
            ax = axes[row_idx, dim_idx + 1]
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Plot factual history 
            if t0 > 0:
                time_history = np.arange(0, t0 + 1)  # Include t0
                gt_history = gt_seq_unscaled[:t0 + 1, dim_idx]  
                ax.plot(time_history, gt_history, color='black', linewidth=1.5, alpha=0.7, label='factual' if (row_idx == 0 and dim_idx == 0) else None)
            
            # Plot CF future
            time_future = np.arange(t0, t0 + H)
            gt_future = gt_seq_unscaled[t0:t0 + H, dim_idx]
            
            # Ground truth CF 
            ax.plot(time_future, gt_future, color='#525354', linewidth=2.0, linestyle='--', label='CF ground truth' if (row_idx == 0 and dim_idx == 0) else None, zorder=5)
            
            # Set title with CF scenario ID
            ax.set_title(f'Patient {patient_id} | CF #{cf_scenario_id} - {outcome_labels[dim_idx]}', fontsize=10, fontweight='bold')
            
            # Transformer prediction 
            pred_T = mu_T[:H, dim_idx]
            std_T_plot = std_T[:H, dim_idx]
            line_T = ax.plot(time_future, pred_T, color='#F28F00', linewidth=1.8, zorder=3)[0]
            fill_T = ax.fill_between(
                time_future,
                pred_T - n_sigma * std_T_plot,
                pred_T + n_sigma * std_T_plot,
                color='#FBDEB3',
                alpha=0.8,
                zorder=2
            )
            if row_idx == 0 and dim_idx == 0:
                ax.plot([], [], color='#F28F00', linewidth=1.8, label='transformer')
                ax.fill_between([], [], [], color='#FBDEB3', alpha=0.8)
            
            # SC prediction 
            pred_SC = mu_SC[:H, dim_idx]
            std_SC_plot = std_SC[:H, dim_idx]
            line_SC = ax.plot(time_future, pred_SC, color='#047EFF', linewidth=1.8, zorder=3)[0]
            fill_SC = ax.fill_between(
                time_future,
                pred_SC - n_sigma * std_SC_plot,
                pred_SC + n_sigma * std_SC_plot,
                color='#B2D9FF',
                alpha=0.8,
                zorder=2
            )
            if row_idx == 0 and dim_idx == 0:
                ax.plot([], [], color='#047EFF', linewidth=1.8, label='SC')
                ax.fill_between([], [], [], color='#B2D9FF', alpha=0.8)
            
            # Fused prediction with uncertainty 
            pred_F = mu_F[:H, dim_idx]
            std_F_plot = std_F[:H, dim_idx]
            line_F = ax.plot(time_future, pred_F, color='#E52716', linewidth=1.8, zorder=3)[0]
            fill_F = ax.fill_between(
                time_future,
                pred_F - n_sigma * std_F_plot,
                pred_F + n_sigma * std_F_plot,
                color='#F7BBB6',
                alpha=0.8,
                zorder=2
            )
            if row_idx == 0 and dim_idx == 0:
                ax.plot([], [], color='#E52716', linewidth=1.8, label='fused')
                ax.fill_between([], [], [], color='#F7BBB6', alpha=0.8)
            
            # Mark padded timesteps
            if mask_arr is not None:
                future_mask = mask_arr[t0:t0 + H].astype(bool)
                ax.scatter(time_future[~future_mask], gt_future[~future_mask],
                          color='red', s=15, alpha=0.6, zorder=5)
            
            # Mark split point
            ax.axvline(t0, color='lightgreen', linestyle=':', linewidth=1.5, label='split' if (row_idx == 0 and dim_idx == 0) else None, zorder=4)
            
            ax.tick_params(axis='both', labelsize=13)
            ax.set_xlabel('Time', fontsize=13)
            ax.set_ylabel(outcome_labels[dim_idx], fontsize=13)
            ax.grid(True, alpha=0.3)
    
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=13, bbox_to_anchor=(0.5, -0.01))
    
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig_path = os.path.join(output_dir, f'patients_combined_{len(selected_patient_ids)}.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(fig_path)
    logger.info(f'Saved combined patient plot: {fig_path}')
    
    return saved_files

