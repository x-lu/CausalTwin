import pandas as pd
import numpy as np
import logging
import torch
from torch.utils.data import Dataset
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
from typing import List
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
from hydra.utils import instantiate
from copy import deepcopy
import os

from load_data import load_mimic3_data_raw
from utils import sigmoid, SplineTrendsMixture
from real_dataset import MIMIC3RealDataset
from dataset_collection import SyntheticDatasetCollection

logger = logging.getLogger(__name__)

# Get root path
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SyntheticOutcomeGenerator:
    """
    Generator of synthetic outcome
    """

    def __init__(self,
                 exogeneous_vars: List[str],
                 exog_dependency: callable,
                 exog_weight: float,
                 endo_dependency: callable,
                 endo_rand_weight: float,
                 endo_spline_weight: float,
                 outcome_name: str):
        """
        Args:
            exogeneous_vars: List of time-varying covariates
            exog_dependency: Callable function of exogeneous_vars (f_Z)
            exog_weight: alpha_f
            endo_dependency: Callable function of endogenous variables (g)
            endo_rand_weight: alpha_g
            endo_spline_weight: alpha_S
            outcome_name: Name of the outcome variable j
        """
        self.exogeneous_vars = exogeneous_vars
        self.exog_dependency = exog_dependency
        self.exog_weight = exog_weight
        self.endo_rand_weight = endo_rand_weight
        self.endo_spline_weight = endo_spline_weight
        self.endo_dependency = endo_dependency
        self.outcome_name = outcome_name

    def simulate_untreated(self, all_vitals: pd.DataFrame, static_features: pd.DataFrame):
        """
        Simulate untreated outcomes (Z)
        Args:
            all_vitals: Time-varying covariates (as exogeneous vars)
            static_features: Static covariates (as exogeneous vars)
        """
        logger.info(f'Simulating untreated outcome {self.outcome_name}')
        user_sizes = all_vitals.groupby(level='subject_id', sort=False).size()

        # Exogeneous dependency
        all_vitals[f'{self.outcome_name}_exog'] = self.exog_weight * self.exog_dependency(all_vitals[self.exogeneous_vars].values)

        # Endogeneous dependency + B-spline trend
        time_range = np.arange(0, user_sizes.max())
        y_endo_rand = self.endo_dependency(time_range, len(user_sizes))
        y_endo_splines = SplineTrendsMixture(n_patients=len(user_sizes), max_time=user_sizes.max())(time_range)
        y_endo_full = self.endo_rand_weight * y_endo_rand + self.endo_spline_weight * y_endo_splines

        all_vitals[f'{self.outcome_name}_endo'] = \
            np.array([value for (i, l) in enumerate(user_sizes) for value in y_endo_full[i, :l]]).reshape(-1, 1)

        # Untreated outcome
        all_vitals[f'{self.outcome_name}_untreated'] = \
            all_vitals[f'{self.outcome_name}_exog'] + all_vitals[f'{self.outcome_name}_endo']

        # Placeholder for treated outcome
        all_vitals[f'{self.outcome_name}'] = all_vitals[f'{self.outcome_name}_untreated'].copy()


class SyntheticTreatment:
    """
    Generator of synthetic treatment
    """

    def __init__(self,
                 confounding_vars: List[str],
                 confounder_outcomes: List[str],
                 confounding_dependency: callable,
                 window: float,
                 conf_outcome_weight: float,
                 conf_vars_weight: float,
                 bias: float,
                 full_effect: float,
                 effect_window: float,
                 treatment_name: str,
                 post_nonlinearity: callable = None):
        """
        Args:
            confounding_vars: Confounding time-varying covariates (from all_vitals)
            confounder_outcomes: Confounding previous outcomes
            confounding_dependency: Callable function of confounding_vars (f_Y)
            window: Window of averaging of confounding previous outcomes (T_l)
            conf_outcome_weight: gamma_Y
            conf_vars_weight: gamma_X
            bias: constant bias
            full_effect: beta
            effect_window: w_l
            treatment_name: Name of treatment l
            post_nonlinearity: Post non-linearity after sigmoid
        """
        self.confounding_vars = confounding_vars
        self.confounder_outcomes = confounder_outcomes
        self.confounding_dependency = confounding_dependency
        self.treatment_name = treatment_name
        self.post_nonlinearity = post_nonlinearity

        # Parameters
        self.window = window
        self.conf_outcome_weight = conf_outcome_weight
        self.conf_vars_weight = conf_vars_weight
        self.bias = bias

        self.full_effect = full_effect
        self.effect_window = effect_window

    def treatment_proba(self, patient_df, t):
        """
        Calculates propensity score for patient_df and time-step t
        Args:
            patient_df: DataFrame of patient
            t: Time-step

        Returns: propensity score
        """
        t_start = max(0, t - self.window)

        agr_range = np.arange(t_start, t + 1)
        avg_y = patient_df.loc[agr_range, self.confounder_outcomes].values.mean()
        x = patient_df.loc[t, self.confounding_vars].values.reshape(1, -1)
        f_x = self.confounding_dependency(x)
        treat_proba = sigmoid(self.bias + self.conf_outcome_weight * avg_y + self.conf_vars_weight * f_x).flatten()
        if self.post_nonlinearity is not None:
            treat_proba = self.post_nonlinearity(treat_proba)
        return treat_proba

    def get_treated_outcome(self, patient_df, t, outcome_name, treat_proba=1.0, treat=True):
        """
        Calculate future outcome under treatment, applied at the time-step t
        Args:
            patient_df: DataFrame of patient
            t: Time-step
            outcome_name: Name of the outcome variable j
            treat_proba: Propensity scores of treatment
            treat: Treatment application flag

        Returns: Effect window, treated outcome
        """
        scaled_effect = self.full_effect * treat_proba

        t_stop = min(max(patient_df.index.get_level_values('hours_in')), t + self.effect_window)
        treatment_range = np.arange(t + 1, t_stop + 1)
        treatment_range_rel = treatment_range - t

        future_outcome = patient_df.loc[treatment_range, outcome_name]
        if treat:
            future_outcome += scaled_effect / treatment_range_rel ** 0.5
        return treatment_range, future_outcome

    @staticmethod
    def combine_treatments(treatment_ranges, treated_future_outcomes, treat_flags):
        """
        Min combining of different treatment effects
        Args:
            treatment_ranges: List of effect windows w_l
            treated_future_outcomes: Future outcomes under each individual treatment
            treat_flags: Treatment application flags

        Returns: Combined effect window, combined treated outcome
        """
        treated_future_outcomes = pd.concat(treated_future_outcomes, axis=1)
        if treat_flags.any():  # Min combining all the treatments
            common_treatment_range = [set(treatment_range) for i, treatment_range in enumerate(treatment_ranges) if
                                      treat_flags[i]]
            common_treatment_range = set.union(*common_treatment_range)
            common_treatment_range = sorted(list(common_treatment_range))
            treated_future_outcomes = treated_future_outcomes.loc[common_treatment_range]
            treated_future_outcomes['agg'] = np.nanmin(treated_future_outcomes.iloc[:, treat_flags].values, axis=1)
        else:  # No treatment is applied
            common_treatment_range = treatment_ranges[0]
            treated_future_outcomes['agg'] = treated_future_outcomes.iloc[:, 0]  # Taking untreated outcomes
        return common_treatment_range, treated_future_outcomes['agg']


class MIMIC3SyntheticDataset(MIMIC3RealDataset):
    """
    Pytorch-style semi-synthetic MIMIC-III dataset
    """
    def __init__(self,
                 all_vitals: pd.DataFrame,
                 static_features: pd.DataFrame,
                 synthetic_outcomes: List[SyntheticOutcomeGenerator],
                 synthetic_treatments: List[SyntheticTreatment],
                 treatment_outcomes_influence: dict,
                 subset_name: str,
                 mode='factual',
                 projection_horizon: int = None,
                 treatments_seq: np.array = None,
                 n_treatments_seq: int = None,
                 n_splits_per_patient: int = 1):
        """
        Args:
            all_vitals: DataFrame with vitals (time-varying covariates); multiindex by (patient_id, timestep)
            static_features: DataFrame with static features
            synthetic_outcomes: List of SyntheticOutcomeGenerator
            synthetic_treatments: List of SyntheticTreatment
            treatment_outcomes_influence: dict with treatment-outcomes influences
            subset_name: train / val / test
            mode: factual / counterfactual_one_step / counterfactual_treatment_seq
            projection_horizon: Range of tau-step-ahead prediction (tau = projection_horizon + 1)
            treatments_seq: Fixed (non-random) treatment sequecne for multiple-step-ahead prediction
            n_treatments_seq: Number of random trajectories after rolling origin in test subset
            n_splits_per_patient: Number of random split times per patient (counterfactual modes).
                                  Use 0 or None to keep all valid split times.
        """

        self.subset_name = subset_name
        self.all_vitals = all_vitals.copy()
        vital_cols = all_vitals.columns
        self.synthetic_outcomes = synthetic_outcomes
        self.synthetic_treatments = synthetic_treatments
        self.treatment_outcomes_influence = treatment_outcomes_influence

        prev_treatment_cols = [f'{treatment.treatment_name}_prev' for treatment in self.synthetic_treatments]
        outcome_cols = [outcome.outcome_name for outcome in self.synthetic_outcomes]

        # Sampling untreated outcomes
        for outcome in self.synthetic_outcomes:
            outcome.simulate_untreated(self.all_vitals, static_features)
        # Placeholders
        for treatment in self.synthetic_treatments:
            self.all_vitals[f'{treatment.treatment_name}_prev'] = 0.0
        self.all_vitals['fact'] = np.nan
        self.all_vitals.loc[(slice(None), 0), 'fact'] = 1.0  # First observation is always factual
        user_sizes = self.all_vitals.groupby(level='subject_id', sort=False).size()

        # Treatment application
        seeds = np.random.randint(np.iinfo(np.int32).max, size=len(static_features))
        par = Parallel(n_jobs=multiprocessing.cpu_count() - 10, backend='loky')
        # par = Parallel(n_jobs=4, backend='loky')
        logger.info(f'Simulating {mode} treatments and applying them to outcomes.')
        if mode == 'factual':
            self.all_vitals = par(delayed(self.treat_patient_factually)(patient_ix, seed)
                                  for patient_ix, seed in tqdm(zip(static_features.index, seeds), total=len(static_features)))
        elif mode == 'counterfactual_treatment_seq' or mode == 'counterfactual_one_step':
            self.treatments_seq = treatments_seq
            if mode == 'counterfactual_one_step':
                treatment_options = [0.0, 1.0]  # TODO not only binary treatments
                self.treatments_seq = np.array([x for x in itertools.product(*([treatment_options] * len(prev_treatment_cols)))])
                self.treatments_seq = self.treatments_seq[:, None, :]
                self.cf_start = 0
            else:
                self.cf_start = 1   # First time-step needed for encoder

            assert (projection_horizon is not None and n_treatments_seq is not None) or self.treatments_seq is not None

            if self.treatments_seq is not None:
                self.projection_horizon = self.treatments_seq.shape[1]
                self.n_treatments_seq = self.treatments_seq.shape[0]
            else:
                self.projection_horizon = projection_horizon
                self.n_treatments_seq = n_treatments_seq
            self.n_splits_per_patient = int(n_splits_per_patient) if n_splits_per_patient is not None else None

            patient_ids = static_features.index.to_numpy()
            self.all_vitals = par(delayed(self.treat_patient_counterfactually)(patient_ix, seed)
                                  for patient_ix, seed in tqdm(zip(patient_ids, seeds), total=len(static_features)))

            logger.info('Concatenating all the trajectories together.')
            valid_mask = np.array([len(cf_patient_df) > 0 for cf_patient_df in self.all_vitals], dtype=bool)
            valid_patient_ids = patient_ids[valid_mask]
            self.all_vitals = [cf_patient_df for cf_patient_df in self.all_vitals if len(cf_patient_df) > 0]
            self.all_vitals = [pd.concat(cf_patient_df, keys=range(len(cf_patient_df)), names=['traj'])
                               for cf_patient_df in self.all_vitals]
        else:
            raise NotImplementedError()

        if mode == 'counterfactual_treatment_seq' or mode == 'counterfactual_one_step':
            self.all_vitals = pd.concat(self.all_vitals, keys=valid_patient_ids, names=['subject_id'])
        else:
            self.all_vitals = pd.concat(self.all_vitals, keys=static_features.index, names=['subject_id'])

        if mode == 'factual':
            # Padding with nans
            self.all_vitals = self.all_vitals.unstack(fill_value=np.nan, level=0).stack(dropna=False).swaplevel(0, 1).sort_index()
            static_features = static_features.sort_index()
            # Preserve original patient IDs before converting to numpy
            base_ids_original = np.array(static_features.index).astype(int)
            static_features = static_features.values
            seq_len = int(max(user_sizes))

        elif mode == 'counterfactual_one_step' or mode == 'counterfactual_treatment_seq':
            self.all_vitals = self.all_vitals.unstack(fill_value=np.nan, level=2).stack(dropna=False).sort_index()

            # Extract base patient IDs and CF split times: all_vitals has multiindex (subject_id, traj, hours_in)
            # Get one subject_id and cf_split_time per (subject_id, traj) pair by taking the first timestep of each trajectory
            trajectory_heads = self.all_vitals.groupby(['subject_id', 'traj']).head(1)
            base_ids_original = np.array(trajectory_heads.index.get_level_values('subject_id')).astype(int)
            cf_split_times = np.array(trajectory_heads['cf_split_time']).astype(int)

            subject_ids = trajectory_heads.index.get_level_values('subject_id')
            static_features = static_features.loc[subject_ids].values
            seq_len = int(self.all_vitals.index.get_level_values('hours_in').max()) + 1

        # Conversion to np arrays
        treatments = self.all_vitals[prev_treatment_cols].fillna(0.0).values.reshape((-1, seq_len,
                                                                                      len(prev_treatment_cols)))
        vitals = self.all_vitals[vital_cols].fillna(0.0).values.reshape((-1, seq_len, len(vital_cols)))
        outcomes_unscaled = self.all_vitals[outcome_cols].fillna(0.0).values.reshape((-1, seq_len, len(outcome_cols)))
        active_entries = (~self.all_vitals.isna().all(1)).astype(float)
        active_entries = active_entries.values.reshape((-1, seq_len, 1))
        user_sizes = np.squeeze(active_entries.sum(1))

        logger.info(f'Shape of exploded vitals: {vitals.shape}.')

        self.data = {
            'sequence_lengths': user_sizes - 1,
            'prev_treatments': treatments[:, :-1, :],
            'vitals': vitals[:, 1:, :],
            'next_vitals': vitals[:, 2:, :],
            'current_treatments': treatments[:, 1:, :],
            'static_features': static_features,
            'active_entries': active_entries[:, 1:, :],
            'unscaled_outputs': outcomes_unscaled[:, 1:, :],
            'prev_unscaled_outputs': outcomes_unscaled[:, :-1, :],
            'base_ids_original': base_ids_original,
        }
        
        # Add future_past_split for counterfactual modes
        if mode == 'counterfactual_one_step' or mode == 'counterfactual_treatment_seq':
            self.data['future_past_split'] = cf_split_times

        self.base_ids_original = base_ids_original

        self.processed = False  # Need for normalisation of newly generated outcomes
        self.processed_sequential = False
        self.processed_autoregressive = False

        self.norm_const = 1.0

    def plot_timeseries(self, n_patients=5, mode='factual', save_path=None):
        """
        Plotting patient trajectories
        Args:
            n_patients: Number of trajectories
            mode: factual / counterfactual
            save_path: Optional path to save the figure
        """
        # plot outcomes (y1, y2), not intermediate components
        fig, ax = plt.subplots(nrows=len(self.synthetic_outcomes), ncols=1, figsize=(15, 4 * len(self.synthetic_outcomes)))
        if len(self.synthetic_outcomes) == 1:
            ax = [ax]
        
        for i, patient_ix in enumerate(self.all_vitals.index.levels[0][:n_patients]):
            factuals = self.all_vitals.fillna(0.0).fact.astype(bool)
            
            for ax_ind, outcome in enumerate(self.synthetic_outcomes):
                outcome_name = outcome.outcome_name
                
                if mode == 'factual':
                    # For factual: just plot the treated outcome as a line
                    ax[ax_ind].plot(self.all_vitals.loc[patient_ix, outcome_name].values, 
                                   label=f'Patient {patient_ix}', alpha=0.7)
                    
                elif mode == 'counterfactual':
                    # For counterfactual: plot factual trajectory as line, all CFs as thin lines
                    # Plot factual trajectory (bold)
                    factual_times = self.all_vitals[factuals].loc[patient_ix, outcome_name].groupby('hours_in').head(1).index.get_level_values(1)
                    factual_values = self.all_vitals[factuals].loc[patient_ix, outcome_name].groupby('hours_in').head(1).values
                    line = ax[ax_ind].plot(factual_times, factual_values, 
                                          label=f'Patient {patient_ix} (factual)', linewidth=2, alpha=0.9)
                    color = line[0].get_color()
                    
                    # Plot all counterfactual trajectories (thin lines)
                    for traj_ix in self.all_vitals.loc[patient_ix].index.get_level_values(0).unique():
                        cf_times = self.all_vitals.loc[(patient_ix, traj_ix), outcome_name].index.get_level_values(0)
                        cf_values = self.all_vitals.loc[(patient_ix, traj_ix), outcome_name].values
                        ax[ax_ind].plot(cf_times, cf_values, color=color, linewidth=0.3, alpha=0.3)
                
                ax[ax_ind].set_ylabel(f'{outcome_name}')
                ax[ax_ind].set_xlabel('Time (hours)')
                if i == 0:  
                    ax[ax_ind].legend(loc='best', fontsize=8)
                ax[ax_ind].grid(True, alpha=0.3)

        fig.suptitle(f'Patient Trajectories - {self.subset_name} ({mode})', fontsize=16)
        plt.tight_layout()
        
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f'Saved plot to {save_path}')
            plt.close(fig)
        else:
            plt.show()
        plt.close()

    def _sample_treatments_from_factuals(self, patient_df, t, rng=np.random.RandomState(None)):
        """
        Sample treatment for patient_df and time-step t
        Args:
            patient_df: DataFrame of patient
            t: Time-step
            rng: Random numbers generator (for parallelizing)

        Returns: Propensity scores, sampled treatments
        """
        factual_patient_df = patient_df[patient_df.fact.astype(bool)]
        treat_probas = {treatment.treatment_name: treatment.treatment_proba(factual_patient_df, t) for treatment in
                        self.synthetic_treatments}
        treatment_sample = {treatment_name: rng.binomial(1, treat_proba)[0] for treatment_name, treat_proba in
                            treat_probas.items()}
        return treat_probas, treatment_sample

    def _combined_treating(self, patient_df, t, outcome: SyntheticOutcomeGenerator, treat_probas: dict, treat_flags: dict):
        """
        Combing application of treatments
        Args:
            patient_df: DataFrame of patient
            t: Time-step
            outcome: Outcome to treat
            treat_probas: Propensity scores
            treat_flags: Treatment application flags

        Returns: Combined effect window, combined treated outcome
        """
        treatment_ranges, treated_future_outcomes = [], []
        influencing_treatments = self.treatment_outcomes_influence[outcome.outcome_name]
        influencing_treatments = \
            [treatment for treatment in self.synthetic_treatments if treatment.treatment_name in influencing_treatments]

        for treatment in influencing_treatments:
            treatment_range, treated_future_outcome = \
                treatment.get_treated_outcome(patient_df, t, outcome.outcome_name, treat_probas[treatment.treatment_name],
                                              bool(treat_flags[treatment.treatment_name]))

            treatment_ranges.append(treatment_range)
            treated_future_outcomes.append(treated_future_outcome)

        common_treatment_range, future_outcomes = SyntheticTreatment.combine_treatments(
            treatment_ranges,
            treated_future_outcomes,
            np.array([bool(treat_flags[treatment.treatment_name]) for treatment in influencing_treatments])
        )
        return common_treatment_range, future_outcomes

    def treat_patient_factually(self, patient_ix: int, seed: int = None):
        """
        Generate factually treated outcomes
        Args:
            patient_ix: Index of patient
            seed: Random seed

        Returns: DataFrame of patient
        """
        patient_df = self.all_vitals.loc[patient_ix].copy()
        rng = np.random.RandomState(seed)
        prev_treatment_cols = [f'{treatment.treatment_name}_prev' for treatment in self.synthetic_treatments]

        for t in range(len(patient_df)):

            # Sampling treatments, based on previous factual outcomes
            treat_probas, treat_flags = self._sample_treatments_from_factuals(patient_df, t, rng)

            if t < max(patient_df.index.get_level_values('hours_in')):
                # Setting factuality flags
                patient_df.loc[t + 1, 'fact'] = 1.0

                # Setting factual sampled treatments
                patient_df.loc[t + 1, prev_treatment_cols] = {f'{t}_prev': v for t, v in treat_flags.items()}

                # Treatments applications
                if sum(treat_flags.values()) > 0:

                    # Treating each outcome separately
                    for outcome in self.synthetic_outcomes:
                        common_treatment_range, future_outcomes = self._combined_treating(patient_df, t, outcome, treat_probas,
                                                                                          treat_flags)
                        patient_df.loc[common_treatment_range, f'{outcome.outcome_name}'] = future_outcomes

        return patient_df

    def treat_patient_counterfactually(self, patient_ix: int, seed: int = None):
        """
        Generate counterfactually treated outcomes
        Args:
            patient_ix: Index of patient
            seed: Random seed

        Returns: DataFrame of patient
        """
        patient_df = self.all_vitals.loc[patient_ix].copy()
        rng = np.random.RandomState(seed)
        prev_treatment_cols = [f'{treatment.treatment_name}_prev' for treatment in self.synthetic_treatments]
        treatment_options = [0.0, 1.0]  #

        cf_patient_dfs = []
        max_hour = max(patient_df.index.get_level_values('hours_in'))
        valid_split_times = [
            t for t in range(len(patient_df))
            if t >= self.cf_start and t + self.projection_horizon <= max_hour
        ]
        split_times = []
        if len(valid_split_times) > 0:
            if self.n_splits_per_patient is None or int(self.n_splits_per_patient) <= 0:
                split_times = list(valid_split_times)
            else:
                k = min(int(self.n_splits_per_patient), len(valid_split_times))
                split_times = rng.choice(valid_split_times, size=k, replace=False).tolist()
        split_times_set = set(split_times)
        if len(split_times_set) == 0:
            return []

        for t in range(len(patient_df)):

            if self.treatments_seq is None:  # sampling random treatment trajectories
                possible_treatments = rng.choice(treatment_options,
                                                 (self.n_treatments_seq, self.projection_horizon, len(self.synthetic_treatments)))
            else:  # using pre-defined trajectories
                possible_treatments = self.treatments_seq

            if t + self.projection_horizon <= max(patient_df.index.get_level_values('hours_in')):

                # --------------- Counterfactual treatment treatment trajectories ---------------
                if t >= self.cf_start and len(split_times_set) > 0 and t in split_times_set:
                    possible_patient_dfs = \
                        [patient_df.copy().loc[:t + self.projection_horizon] for _ in range(possible_treatments.shape[0])]
                    
                    # Mark the split time for each CF trajectory
                    for possible_patient_df in possible_patient_dfs:
                        possible_patient_df['cf_split_time'] = t

                    for time_ind in range(self.projection_horizon):
                        for traj_ind, possible_treatment in enumerate(possible_treatments[:, time_ind, :]):

                            future_treat_probas, _ = \
                                self._sample_treatments_from_factuals(possible_patient_dfs[traj_ind], t + time_ind, rng)
                            future_treat_flags = {treatment.treatment_name: possible_treatment[j]
                                                  for j, treatment in enumerate(self.synthetic_treatments)}

                            # Setting treatment trajectories and factualities
                            possible_patient_dfs[traj_ind].loc[t + 1 + time_ind, prev_treatment_cols] = \
                                {f'{t}_prev': v for t, v in future_treat_flags.items()}

                            # Setting pseudo-factuality to ones (needed for proper future_treat_probas)
                            possible_patient_dfs[traj_ind].at[t + 1 + time_ind, 'fact'] = 1.0

                            # Treating each outcome separately
                            for outcome in self.synthetic_outcomes:
                                common_treatment_range, future_outcomes = \
                                    self._combined_treating(possible_patient_dfs[traj_ind], t + time_ind, outcome,
                                                            future_treat_probas, future_treat_flags)
                                possible_patient_dfs[traj_ind].loc[common_treatment_range, outcome.outcome_name] = future_outcomes

                    # Setting pseudo-factuality to zero
                    for possible_patient_df in possible_patient_dfs:
                        possible_patient_df.loc[t + 1:, 'fact'] = 0.0
                    cf_patient_dfs.extend(possible_patient_dfs)

                # --------------- Factual treatment sampling & Application ---------------
                treat_probas, treat_flags = self._sample_treatments_from_factuals(patient_df, t, rng)

                # Setting factuality
                patient_df.loc[t + 1, 'fact'] = 1.0

                # Setting factual sampled treatments
                patient_df.loc[t + 1, prev_treatment_cols] = {f'{t}_prev': v for t, v in treat_flags.items()}

                # Treating each outcome separately
                if sum(treat_flags.values()) > 0:

                    # Treating each outcome separately
                    for outcome in self.synthetic_outcomes:
                        common_treatment_range, future_outcomes = \
                            self._combined_treating(patient_df, t, outcome, treat_probas, treat_flags)
                        patient_df.loc[common_treatment_range, outcome.outcome_name] = future_outcomes

        return cf_patient_dfs

    def get_scaling_params(self):
        outcome_cols = [outcome.outcome_name for outcome in self.synthetic_outcomes]
        logger.info('Performing normalisation.')
        scaling_params = {
            'output_means': self.all_vitals[outcome_cols].mean(0).to_numpy(),
            'output_stds': self.all_vitals[outcome_cols].std(0).to_numpy(),
        }
        return scaling_params

    def process_data(self, scaling_params):
        """
        Pre-process dataset for one-step-ahead prediction
        Args:
            scaling_params: dict of standard normalization parameters (calculated with train subset)
        """
        if not self.processed:
            logger.info(f'Processing {self.subset_name} dataset before training')

            self.data['outputs'] = (self.data['unscaled_outputs'] - scaling_params['output_means']) / \
                scaling_params['output_stds']
            self.data['prev_outputs'] = (self.data['prev_unscaled_outputs'] - scaling_params['output_means']) / \
                scaling_params['output_stds']

            #if self.autoregressive:
            #    self.data['vitals'] = np.concatenate([self.data['vitals'], self.data['prev_outputs']], axis=2)

            data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

            self.scaling_params = scaling_params
            self.processed = True
        else:
            logger.info(f'{self.subset_name} Dataset already processed')

        return self.data


class MIMIC3SyntheticDatasetCollection(SyntheticDatasetCollection):
    """
    Dataset collection (train_f, val_f, test_cf_one_step, test_cf_treatment_seq)
    """
    def __init__(self,
                 path,
                 synth_outcomes_list: list,
                 synth_treatments_list: list,
                 treatment_outcomes_influence: dict,
                 min_seq_length: int = None,
                 max_seq_length: int = None,
                 max_number: int = None,
                 seed: int = 100,
                 data_seed: int = 100,
                 split: dict = {'val': 0.2, 'test': 0.2},
                 projection_horizon: int = 4,
                 autoregressive=True,
                 n_treatments_seq: int = None,
                 n_splits_per_patient: int = 1,
                 **kwargs):
        """
        Args:
            path: Path with MIMIC-3 dataset (HDFStore)
            synth_outcomes_list: List of SyntheticOutcomeGenerator
            synth_treatments_list: List of SyntheticTreatment
            treatment_outcomes_influence: dict with treatment-outcomes influences
            min_seq_length: Min sequence lenght in cohort
            max_seq_length: Max sequence lenght in cohort
            max_number: Maximum number of patients in cohort
            seed: Seed for sampling random functions
            data_seed: Seed for random cohort patient selection
            split: Ratio of train / val / test split
            projection_horizon: Range of tau-step-ahead prediction (tau = projection_horizon + 1)
            n_treatments_seq: Number of random trajectories after rolling origin in test subset
            n_splits_per_patient: Number of random split times per patient (counterfactual modes)
        """
        super(MIMIC3SyntheticDatasetCollection, self).__init__()
        self.seed = seed
        np.random.seed(seed)
        all_vitals, static_features = load_mimic3_data_raw(path, min_seq_length=min_seq_length,
                                                           max_seq_length=max_seq_length, max_number=max_number,
                                                           data_seed=data_seed, **kwargs)

        # Train/val/test random_split
        static_features, static_features_test = train_test_split(static_features, test_size=split['test'], random_state=seed)
        all_vitals, all_vitals_test = all_vitals.loc[static_features.index], all_vitals.loc[static_features_test.index]

        if split['val'] > 0.0:
            static_features_train, static_features_val = train_test_split(static_features,
                                                                          test_size=split['val'] / (1 - split['test']),
                                                                          random_state=2 * seed)
            all_vitals_train, all_vitals_val = all_vitals.loc[static_features_train.index], \
                all_vitals.loc[static_features_val.index]
        else:
            static_features_train = static_features
            all_vitals_train = all_vitals

        self.train_f = \
            MIMIC3SyntheticDataset(all_vitals_train, static_features_train, synth_outcomes_list, synth_treatments_list,
                                   treatment_outcomes_influence, 'train')
        # self.train_f.plot_timeseries()

        if split['val'] > 0.0:
            self.val_f = MIMIC3SyntheticDataset(all_vitals_val, static_features_val, synth_outcomes_list, synth_treatments_list,
                                                treatment_outcomes_influence, 'val')
        self.test_cf_one_step = \
            MIMIC3SyntheticDataset(all_vitals_test, static_features_test, synth_outcomes_list, synth_treatments_list,
                                   treatment_outcomes_influence, 'test', 'counterfactual_one_step',
                                   n_splits_per_patient=n_splits_per_patient)
        # self.test_cf_one_step.plot_timeseries(mode='counterfactual')

        self.test_cf_treatment_seq = \
            MIMIC3SyntheticDataset(all_vitals_test, static_features_test, synth_outcomes_list, synth_treatments_list,
                                   treatment_outcomes_influence, 'test', 'counterfactual_treatment_seq',
                                   projection_horizon, n_treatments_seq=n_treatments_seq, n_splits_per_patient=n_splits_per_patient)
        # self.test_cf_treatment_seq.plot_timeseries(mode='counterfactual')

        self.projection_horizon = projection_horizon
        self.autoregressive = autoregressive
        self.has_vitals = True
        self.train_scaling_params = self.train_f.get_scaling_params()


# -------------------- PyTorch Dataset -------------------- #
class MIMIC3SyntheticTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset wrapper for MIMIC-3 semi-synthetic data.
    
    Constructs input tensor x from:
      - 'vitals'            : (N, T, Dv)  time-varying vital signs
      - 'prev_outputs'      : (N, T, Do)  previous outcome values (normalized)
      - 'current_treatments': (N, T, Dt)  time-varying treatment 
      - 'static_features'   : (N, Ds)     patient demographics 
    
    Targets:
      - 'outputs'           : (N, T, Do)  outcome values to predict (normalized)
    
    Mask:
      - 'active_entries'    : (N, T) or (N, T, 1)  binary mask (1=valid, 0=padding)
    
    Returns tuple (x, y, mask) for each patient/trajectory.
    """

    def __init__(self, subset):
        """
        Args:
            subset: MIMIC3SyntheticDataset instance with .data dict containing processed arrays
        """
        data = subset.data

        # Targets (normalized outcomes)
        y = data["outputs"]                     # (N, T, Do), numpy
        self.y = torch.from_numpy(y).float()    # (N, T, Do)

        # Vitals 
        vitals = torch.from_numpy(data["vitals"]).float()                     # (N, T, Dv)

        # Previous outputs (autoregressive input)
        prev_outputs = torch.from_numpy(data["prev_outputs"]).float()         # (N, T, Do)

        # Time-varying treatments
        treatments = torch.from_numpy(data["current_treatments"]).float()     # (N, T, Dt)

        # Static features, broadcast over time
        static = torch.from_numpy(data["static_features"]).float()            # (N, Ds)
        N, T = self.y.shape[0], self.y.shape[1]
        static_rep = static.unsqueeze(1).expand(-1, T, -1)                    # (N, T, Ds)

        # Build input features: [vitals, prev_outputs, treatments, static]
        x = torch.cat([vitals, prev_outputs, treatments, static_rep], dim=-1) # (N, T, Dx)
        self.x = x

        # Mask for valid timesteps
        if "active_entries" in data:
            mask = torch.from_numpy(data["active_entries"]).float()
            if mask.dim() == 3 and mask.size(-1) == 1:
                mask = mask.squeeze(-1)  # (N, T, 1) -> (N, T)
        else:
            mask = torch.ones(self.y.shape[:2], dtype=torch.float32)  # (N, T)

        self.mask = mask  # (N, T)

    def __len__(self):
        """Number of patients/trajectories in dataset"""
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        Get one patient's/trajectory's data
        
        Returns:
            x: (T, Dx) - input features for all timesteps
            y: (T, Do) - target outcomes for all timesteps
            mask: (T,) - validity mask for all timesteps
        """
        return self.x[idx], self.y[idx], self.mask[idx]
