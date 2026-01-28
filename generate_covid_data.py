"""exp_covid.generate_data

Utilities for generating COVID-19 SEIR trajectories with Covasim.

"""

import os
import sys
import numpy as np
from pathlib import Path
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from functools import partial
from copy import deepcopy
from torch import multiprocessing

import covasim as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _run_one_weekly(args_spec_pair, sim_config_local):
    args, spec = args_spec_pair
    if spec is not None and 'days' in spec and 'changes' in spec:
        params = deepcopy(sim_config_local)
        params['rand_seed'] = args[0]
        params['pop_infected'] = args[1]
        params['interventions'] = cv.change_beta(days=spec['days'], changes=spec['changes'])
        sim = cv.Sim(pars=params)
        sim.initialize()
        sim.run(verbose=0)
        states = _extract_seir_states(sim)
        indicator = np.zeros(params['n_days'] + 1, dtype=np.float32)
        for s, e in zip(spec['days'][::2], spec['days'][1::2]):
            indicator[int(s):min(int(e), params['n_days'] + 1)] = 1.0
        return states, indicator
    return run_simulation_custom_intervention(args, sim_config_local, intervention_spec=spec, verbose=0)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
POP_SIZE = 100_000
INIT_INFECTED = 100  

# Intervention effect: beta multiplied by this factor during intervention
BETA_CHANGE = 0.25

# Weekly dataset defaults
WEEKS = 21
DAYS_PER_WEEK = 7
WEEKLY_ACTION_P = 0.2  # low-rate: ~10-30% weeks active
WEEKLY_STATE_MODE = 'eow'
WEEKLY_ACTION_MODE = 'mean'

# Default dataset split sizes
TRAIN_SIZE = 800
VAL_SIZE = 100
TEST_SIZE = 100



def _extract_seir_states(sim: cv.Sim) -> np.ndarray:
    S = sim.results.n_susceptible.values
    E = sim.results.n_exposed.values - sim.results.n_infectious.values
    I = sim.results.n_infectious.values
    R = sim.results.n_recovered.values + sim.results.n_dead.values
    return np.stack((S, E, I, R), axis=-1)


def run_simulation_custom_intervention(args, sim_config, intervention_spec=None, verbose=0):
    if len(args) == 4:
        seed, infected, intervention_day, intervention_end_day = args
    elif len(args) == 3:
        seed, infected, intervention_day = args
        intervention_end_day = None
    else:
        seed, infected = args
        intervention_day = None if intervention_spec is None else intervention_spec.get('day', None)
        intervention_end_day = None if intervention_spec is None else intervention_spec.get('end_day', None)
        if intervention_end_day is None and intervention_spec is not None and intervention_spec.get('duration') is not None and intervention_day is not None:
            intervention_end_day = intervention_day + int(intervention_spec['duration'])

    params = deepcopy(sim_config)
    params['rand_seed'] = int(seed)
    params['pop_infected'] = int(infected)

    n_days = params['n_days']

    intervention_indicator = np.zeros(n_days + 1, dtype=np.float32)
    if intervention_day is not None:
        beta_change = BETA_CHANGE if intervention_spec is None else float(intervention_spec.get('beta_change', BETA_CHANGE))

        if intervention_end_day is None:
            params['interventions'] = cv.change_beta(days=int(intervention_day), changes=beta_change)
            intervention_indicator[int(intervention_day):] = 1.0
        else:
            start = int(intervention_day)
            end = int(intervention_end_day)
            if end < start:
                raise ValueError(f"intervention end_day ({end}) must be >= day ({start})")
            params['interventions'] = cv.change_beta(days=[start, end], changes=[beta_change, 1.0])
            intervention_indicator[start:min(end, n_days + 1)] = 1.0

    sim = cv.Sim(pars=params)
    sim.initialize()
    sim.run(verbose=verbose)
    states = _extract_seir_states(sim)
    return states, intervention_indicator


def expand_weekly_actions_to_daily(actions_weekly: np.ndarray, days_per_week: int = 7) -> np.ndarray:
    a = np.asarray(actions_weekly)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    if a.ndim != 1:
        raise ValueError(f"actions_weekly must be 1D or (T_w,1), got shape {a.shape}")
    return np.repeat(a.astype(np.float32), repeats=int(days_per_week))


def aggregate_daily_to_weekly(states_daily: np.ndarray,
                             actions_daily: np.ndarray,
                             weeks: int,
                             days_per_week: int = 7,
                             state_mode: str = 'eow',
                             action_mode: str = 'mean'):
    x = np.asarray(states_daily)
    a = np.asarray(actions_daily)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]

    needed_days = int(weeks) * int(days_per_week)
    if x.shape[0] < needed_days or a.shape[0] < needed_days:
        raise ValueError(f"Not enough daily points: need {needed_days}, got states {x.shape[0]}, actions {a.shape[0]}")

    x = x[:needed_days]
    a = a[:needed_days]

    xw = x.reshape(int(weeks), int(days_per_week), x.shape[1])
    aw = a.reshape(int(weeks), int(days_per_week))

    if state_mode == 'eow':
        states_weekly = xw[:, -1, :]
    elif state_mode == 'mean':
        states_weekly = xw.mean(axis=1)
    else:
        raise ValueError(f"Unknown state_mode: {state_mode}")

    if action_mode == 'mean':
        actions_weekly = aw.mean(axis=1)
    elif action_mode == 'max':
        actions_weekly = aw.max(axis=1)
    else:
        raise ValueError(f"Unknown action_mode: {action_mode}")

    return states_weekly.astype(np.float32), actions_weekly[:, None].astype(np.float32)


def generate_covid_data_weekly(num_trajectories=20,
                              weeks=WEEKS,
                              pop_size=POP_SIZE,
                              multi_process=True,
                              length_offset_days=0,
                              seed=42,
                              weekly_actions=None,
                              days_per_week=DAYS_PER_WEEK,
                              state_mode=WEEKLY_STATE_MODE,
                              action_mode=WEEKLY_ACTION_MODE,
                              action_p: float = WEEKLY_ACTION_P,
                              beta_change: float = BETA_CHANGE):
    rng = np.random.default_rng(int(seed))
    weeks = int(weeks)
    days_per_week = int(days_per_week)
    offset = int(length_offset_days)

    total_days_after_offset = weeks * days_per_week
    n_days = (total_days_after_offset + offset) - 1

    sim_config = {
        'pop_size': pop_size,
        'n_days': n_days,
        'use_waning': False,
        'pop_type': 'random',
        'rescale': False,
    }

    seeds = rng.integers(0, 2**32 - 1, size=num_trajectories, dtype=np.uint32)
    num_infecteds = np.full(num_trajectories, INIT_INFECTED, dtype=np.int64)

    action_p = float(action_p)
    if not (0.0 <= action_p <= 1.0):
        raise ValueError(f"action_p must be in [0,1], got {action_p}")

    beta_change = float(beta_change)
    if beta_change <= 0:
        raise ValueError(f"beta_change must be >0, got {beta_change}")

    if weekly_actions is None:
        weekly_actions_arr = (rng.random((num_trajectories, weeks)) < action_p).astype(np.float32)
    else:
        weekly_actions_arr = np.asarray(weekly_actions, dtype=np.float32)
        if weekly_actions_arr.ndim == 1:
            if weekly_actions_arr.shape[0] != weeks:
                raise ValueError(f"weekly_actions length {weekly_actions_arr.shape[0]} must match weeks {weeks}")
            weekly_actions_arr = np.tile(weekly_actions_arr[None, :], (num_trajectories, 1))
        elif weekly_actions_arr.ndim == 2:
            if weekly_actions_arr.shape != (num_trajectories, weeks):
                raise ValueError(f"weekly_actions shape {weekly_actions_arr.shape} must be (num_trajectories,weeks)=({num_trajectories},{weeks})")
        else:
            raise ValueError("weekly_actions must be None, (weeks,), or (num_trajectories,weeks)")

    intervention_specs = []
    pairs = []
    for seed_i, infected_i, a_weekly in zip(seeds, num_infecteds, weekly_actions_arr):
        a_daily = expand_weekly_actions_to_daily(a_weekly, days_per_week=days_per_week)
        if offset > 0:
            a_daily = np.concatenate([np.zeros(offset, dtype=np.float32), a_daily], axis=0)

        windows = []
        in_one = False
        start = None
        for t, val in enumerate(a_daily):
            if (not in_one) and val >= 0.5:
                in_one = True
                start = t
            elif in_one and val < 0.5:
                windows.append((start, t))
                in_one = False
                start = None
        if in_one and start is not None:
            windows.append((start, len(a_daily)))

        if len(windows) == 0:
            spec = None
            args = (int(seed_i), int(infected_i))
        else:
            days = []
            changes = []
            for (s, e) in windows:
                days.extend([int(s), int(e)])
                changes.extend([float(beta_change), 1.0])
            spec = {'day': int(windows[0][0]), 'beta_change': float(beta_change), 'days': days, 'changes': changes}
            args = (int(seed_i), int(infected_i), int(windows[0][0]))

        intervention_specs.append(spec)
        pairs.append((args, spec))

    states_list = []
    actions_daily_list = []

    if multi_process:
        pool_outer = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 1))
        runner = partial(_run_one_weekly, sim_config_local=sim_config)
        results = [pool_outer.apply_async(runner, (pair,)) for pair in pairs]
        for r in results:
            st, ind = r.get()
            states_list.append(st)
            actions_daily_list.append(ind)
        pool_outer.close()
    else:
        for pair in pairs:
            st, ind = _run_one_weekly(pair, sim_config)
            states_list.append(st)
            actions_daily_list.append(ind)

    states_daily = np.stack(states_list, axis=0) / float(pop_size)
    actions_daily = np.stack(actions_daily_list, axis=0)

    if offset > 0:
        states_daily = states_daily[:, offset:, :]
        actions_daily = actions_daily[:, offset:]

    states_weekly = []
    actions_weekly = []
    for i in range(num_trajectories):
        sw, aw = aggregate_daily_to_weekly(states_daily[i], actions_daily[i], weeks=weeks,
                                           days_per_week=days_per_week,
                                           state_mode=state_mode,
                                           action_mode=action_mode)
        states_weekly.append(sw)
        actions_weekly.append(aw)

    return np.stack(states_weekly, axis=0).astype(np.float32), np.stack(actions_weekly, axis=0).astype(np.float32)


def plot_seir_grid(states: np.ndarray,
                   actions: np.ndarray,
                   output_path: Path,
                   num_trajectories: int = 9):
    """
    Create a 3x3 grid plot of SEIR trajectories.
    
    Args:
        states: (N, T, 4) SEIR states [S, E, I, R] in proportions
        actions: (N, T, 1) intervention indicators
        output_path: path to save the plot
        num_trajectories: number of trajectories to plot (default 9 for 3x3 grid)
    """
    N = min(num_trajectories, len(states))
    T = states.shape[1]
    
    # SEIR colors: S=blue, E=orange, I=red, R=green
    colors = {
        'S': '#1f77b4',  # Blue
        'E': '#ff7f0e',  # Orange
        'I': '#d62728',  # Red
        'R': '#2ca02c',  # Green
    }
    color_intervention = '#D1D3D4'  # Grey for interventions
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    weeks = np.arange(1, T + 1)  # Week numbers starting from 1
    
    for plot_idx in range(N):
        ax = axes[plot_idx]
        
        # Remove top and right borders, only show x and y axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Extract SEIR states for this trajectory
        S = states[plot_idx, :, 0]
        E = states[plot_idx, :, 1]
        I = states[plot_idx, :, 2]
        R = states[plot_idx, :, 3]
        actions_traj = actions[plot_idx, :, 0]
        
        # Shade intervention periods
        for week_idx, intervention in enumerate(actions_traj):
            if intervention > 0.5:
                week_num = week_idx + 1  # Week numbers start from 1
                ax.axvspan(week_num - 0.5, week_num + 0.5, color=color_intervention, 
                          alpha=0.5, lw=0, zorder=0)
        
        # Plot SEIR trajectories
        ax.plot(weeks, S, color=colors['S'], linewidth=2, label='S' if plot_idx == 0 else '', zorder=3)
        ax.plot(weeks, E, color=colors['E'], linewidth=2, label='E' if plot_idx == 0 else '', zorder=3)
        ax.plot(weeks, I, color=colors['I'], linewidth=2, label='I' if plot_idx == 0 else '', zorder=3)
        ax.plot(weeks, R, color=colors['R'], linewidth=2, label='R' if plot_idx == 0 else '', zorder=3)
        
        # Formatting
        ax.set_xlabel('Week' if plot_idx >= 6 else '')
        ax.set_ylabel('Proportion' if plot_idx % 3 == 0 else '')
        ax.set_xlim(0.5, T + 0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(1, T + 1, 2))  # Show odd weeks: 1, 3, 5, ...
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, alpha=0.3)
        
        if plot_idx == 0:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    fig.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved SEIR grid plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate and plot SEIR trajectories')
    parser.add_argument('--num_trajectories', type=int, default=9,
                        help='Number of trajectories to plot (default: 9 for 3x3 grid)')
    parser.add_argument('--weeks', type=int, default=WEEKS,
                        help='Number of weeks per trajectory')
    parser.add_argument('--pop_size', type=int, default=POP_SIZE,
                        help='Covasim population size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for trajectory generation')
    parser.add_argument('--action_p', type=float, default=WEEKLY_ACTION_P,
                        help='P(action=1) per week')
    parser.add_argument('--beta_change', type=float, default=BETA_CHANGE,
                        help='Beta multiplier during intervention')
    parser.add_argument('--output', type=str, default='plots/seir_grid.png',
                        help='Output path for the plot')
    parser.add_argument('--days_per_week', type=int, default=DAYS_PER_WEEK,
                        help='Days per week for simulation')
    
    args = parser.parse_args()
    
    print(f"Generating {args.num_trajectories} SEIR trajectories...")
    states, actions = generate_covid_data_weekly(
        num_trajectories=args.num_trajectories,
        weeks=args.weeks,
        pop_size=args.pop_size,
        multi_process=True,
        seed=args.seed,
        weekly_actions=None,
        days_per_week=args.days_per_week,
        state_mode=WEEKLY_STATE_MODE,
        action_mode=WEEKLY_ACTION_MODE,
        action_p=args.action_p,
        beta_change=args.beta_change,
    )
    
    print(f"Generated {len(states)} trajectories")
    print(f"Creating 3x3 grid plot...")
    
    plot_seir_grid(states, actions, args.output, num_trajectories=args.num_trajectories)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
