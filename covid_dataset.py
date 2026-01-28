"""exp_covid.covid_dataset

PyTorch dataset adapter for COVID-19 weekly SEIR trajectories.
"""

import os
import sys
import numpy as np
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class CovidDataset(Dataset):
    """
    PyTorch dataset for COVID-19 trajectories.
    
    Compatible with exp_mimic pipeline: returns (x, y, mask) where:
    - x: [prev_I(1), treatment_onehot(2)] = 3D input
    - y: [I(1)] = 1D output  
    - mask: (T,) validity mask
    
    Supports rolling origin evaluation (exploded trajectories) and sequential test processing.
    """
    
    def __init__(self, states: np.ndarray, actions: np.ndarray, 
                 scaling_params: Optional[dict] = None):
        """
        Args:
            states: (N, T, 4) SEIR states [S, E, I, R]
            actions: (N, T, 1) intervention indicator [0 or 1]
            scaling_params: optional dict with normalization parameters
        """
        N, T, D = states.shape
        assert D == 4, f"Expected SEIR (4 states), got {D}"
        assert actions.shape == (N, T, 1), f"actions shape mismatch: {actions.shape} != ({N},{T},1)"
        
        self.states = states  # (N, T, 4) - S, E, I, R
        self.actions = actions  # (N, T, 1)
        self.scaling_params = scaling_params or {}
        
        self.num_trajectories = N
        self.sequence_length = T
        self.exploded = False
        
        # Build x, y for exp_mimic format
        self._build_xy()
    
    def _build_xy(self):
        """Build exp_mimic format: x=[prev_I, treatment_onehot], y=[I]."""
        I = self.states[:, :, 2:3]  # (N, T, 1) infected proportion
        
        prev_I = np.concatenate([
            np.zeros((self.num_trajectories, 1, 1), dtype=np.float32),
            I[:, :-1, :]
        ], axis=1)
        
        treatment_onehot = np.concatenate([
            1.0 - self.actions,
            self.actions
        ], axis=2)
        
        self.x = np.concatenate([prev_I, treatment_onehot], axis=2).astype(np.float32)
        self.y = I.astype(np.float32)
        self.mask = np.ones((self.num_trajectories, self.sequence_length), dtype=np.float32)
        
        self.data = {
            'prev_outputs': prev_I,
            'current_treatments': treatment_onehot,
            'prev_treatments': np.concatenate([
                np.zeros((self.num_trajectories, 1, 2), dtype=np.float32),
                treatment_onehot[:, :-1, :]
            ], axis=1),
            'vitals': np.zeros((self.num_trajectories, self.sequence_length, 0), dtype=np.float32),
            'next_vitals': np.zeros((self.num_trajectories, self.sequence_length, 0), dtype=np.float32),
            'static_features': np.zeros((self.num_trajectories, 0), dtype=np.float32),
            'outputs': I,
            'unscaled_outputs': I,
            'active_entries': self.mask[:, :, None],
            'sequence_lengths': np.full(self.num_trajectories, self.sequence_length, dtype=np.int64),
        }
        
    def explode_trajectories(self, projection_horizon: int) -> 'CovidDataset':
        """
        Create rolling origin evaluation dataset.
        
        From N base trajectories of length T, creates N×(T-projection_horizon) 
        exploded trajectories, each using first (t+1) points as history.
        
        Args:
            projection_horizon: forecast horizon
            
        Returns:
            New CovidDataset with exploded trajectories
        """
        if self.exploded:
            raise ValueError("Cannot explode already-exploded dataset")
        
        N = self.num_trajectories
        T = self.sequence_length
        
        if projection_horizon >= T:
            raise ValueError(f"projection_horizon {projection_horizon} must be < sequence_length {T}")
        
        # Number of exploded trajectories per base trajectory
        num_splits = T - projection_horizon
        total_exploded = N * num_splits
        
        # Allocate arrays - store FULL trajectories for visualization
        exploded_states = np.zeros((total_exploded, T, 4), dtype=np.float32)
        exploded_actions = np.zeros((total_exploded, T, 1), dtype=np.float32)
        exploded_sequence_lengths = np.zeros(total_exploded, dtype=np.int64)
        base_trajectory_indices = np.zeros(total_exploded, dtype=np.int64)  # Track which base trajectory
        
        idx = 0
        for i in range(N):
            for split_idx in range(num_splits):
                history_len = split_idx + 1
                
                exploded_states[idx] = self.states[i]
                exploded_actions[idx] = self.actions[i]
                exploded_sequence_lengths[idx] = history_len + projection_horizon
                base_trajectory_indices[idx] = i
                
                idx += 1
        
        # Create new dataset with exploded data
        new_dataset = CovidDataset(exploded_states, exploded_actions, 
                                   scaling_params=self.scaling_params)
        
        # Mark valid timesteps for each exploded trajectory
        for idx in range(total_exploded):
            seq_len = int(exploded_sequence_lengths[idx])
            new_dataset.mask[idx, :seq_len] = 1.0
            new_dataset.mask[idx, seq_len:] = 0.0
        
        # Update data dict with correct active_entries for full window
        new_dataset.data['active_entries'] = new_dataset.mask[:, :, None]
        new_dataset.data['sequence_lengths'] = exploded_sequence_lengths
        new_dataset.data['base_trajectory_indices'] = base_trajectory_indices
        
        new_dataset.exploded = True
        new_dataset.projection_horizon = projection_horizon
        new_dataset.num_base_trajectories = N
        new_dataset.num_splits_per_trajectory = num_splits
        
        new_dataset.base_states = self.states.copy()
        new_dataset.base_actions = self.actions.copy()
        
        new_dataset.data_original = {
            'history_data': new_dataset.y.copy(),
            'intervention_data': new_dataset.actions.copy()[:, :, 0],
            'sequence_lengths': exploded_sequence_lengths.copy(),
            'base_trajectory_indices': base_trajectory_indices.copy(),
            'base_states': self.states.copy(),
            'base_actions': self.actions.copy(),
        }
        
        return new_dataset
    
    def process_sequential_test(self, projection_horizon: int):
        """
        Extract forecast windows for sequential evaluation.
        
        Returns tuple of (x, y, active_entries) where active_entries masks valid forecast points.
        
        Args:
            projection_horizon: forecast horizon
            
        Returns:
            tuple: (x, y, active_entries) all as numpy arrays
        """
        if not self.exploded:
            raise ValueError("Must explode dataset before sequential test processing")
        
        N_exploded = len(self)
        T = self.sequence_length
        
        active_entries = np.zeros((N_exploded, T), dtype=np.float32)
        
        # For each exploded trajectory, mark forecast window as active
        for idx in range(N_exploded):
            split_idx = idx % self.num_splits_per_trajectory
            history_len = split_idx + 1
            
            # Mark forecast window as active
            forecast_start = history_len
            forecast_end = min(history_len + projection_horizon, T)
            active_entries[idx, forecast_start:forecast_end] = 1.0
        
        return self.x, self.y, active_entries
    
    def __len__(self) -> int:
        return self.num_trajectories
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (x, y, mask) for exp_mimic compatibility.
        
        - x: (T, 3) input [prev_I, treatment_onehot(2)]
        - y: (T, 1) output [I]
        - mask: (T,) validity mask
        """
        x = torch.from_numpy(self.x[idx])
        y = torch.from_numpy(self.y[idx])
        mask = torch.from_numpy(self.mask[idx])
        
        return x, y, mask


class CovidDatasetCollection:
    """
    Collection of train/val/test COVID-19 datasets with exp_mimic compatibility.
    """
    
    def __init__(self, states_weekly: np.ndarray, actions_weekly: np.ndarray,
                 train_f: float = 0.7, val_f: float = 0.1):
        """
        Args:
            states_weekly: (N, T, 4) SEIR states
            actions_weekly: (N, T, 1) intervention indicator
            train_f: fraction for training
            val_f: fraction for validation
        """
        N, T, D = states_weekly.shape
        assert D == 4, f"Expected SEIR (4 states), got {D}"
        assert actions_weekly.shape == (N, T, 1), f"actions_weekly shape mismatch"
        
        self.states_weekly = states_weekly
        self.actions_weekly = actions_weekly
        
        # Split indices
        n_train = int(N * train_f)
        n_val = int(N * val_f)
        
        self.train_indices = list(range(0, n_train))
        self.val_indices = list(range(n_train, n_train + n_val))
        self.test_indices = list(range(n_train + n_val, N))
        
        print(f"Dataset split: train={len(self.train_indices)}, "
              f"val={len(self.val_indices)}, test={len(self.test_indices)}")
    
    def get_dataset(self, split: str, explode_horizon: Optional[int] = None) -> CovidDataset:
        """
        Get dataset for specified split.
        
        Args:
            split: 'train', 'val', or 'test'
            explode_horizon: if provided, explode trajectories for rolling origin evaluation
            
        Returns:
            CovidDataset with optional explosion
        """
        if split == 'train':
            indices = self.train_indices
        elif split == 'val':
            indices = self.val_indices
        elif split == 'test':
            indices = self.test_indices
        else:
            raise ValueError(f"Unknown split: {split}")
        
        states = self.states_weekly[indices]
        actions = self.actions_weekly[indices]
        
        dataset = CovidDataset(states, actions)
        
        if explode_horizon is not None:
            dataset = dataset.explode_trajectories(projection_horizon=explode_horizon)
        
        return dataset
