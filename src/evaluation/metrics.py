"""
Core MARL Evaluation Metrics (Track 3)
Calculates convergence, instability, adaptation latency, and communication cost.
"""
from typing import List, Dict, Any, Optional
import numpy as np


def compute_experiment_metrics(
    returns: List[float],
    shock_step: Optional[int] = None,
    comm_costs: Optional[List[float]] = None,
    window_size: int = 10,
) -> Dict[str, Any]:
    """
    Compute rigorous metrics from an experiment run.

    Args:
        returns: list of episodic returns over training
        shock_step: episode index where shock occurred
        comm_costs: list of average active communication ratios per episode
        window_size: smoothing window for convergence calculation
    """
    returns_arr = np.array(returns, dtype=np.float64)
    N = len(returns_arr)

    if N == 0:
        return {}

    # Smoothed returns
    kernel = np.ones(min(window_size, N)) / min(window_size, N)
    smoothed = np.convolve(returns_arr, kernel, mode="valid")

    max_return = float(np.max(smoothed)) if len(smoothed) > 0 else float(np.max(returns_arr))
    min_return = float(np.min(returns_arr))
    mean_return = float(np.mean(returns_arr))

    # 1. Convergence speed: first step exceeding 90% of max return
    threshold_90 = 0.9 * max_return if max_return > 0 else max_return - 0.1 * abs(max_return)
    conv_indices = np.where(smoothed >= threshold_90)[0]
    convergence_episode = int(conv_indices[0]) if len(conv_indices) > 0 else N

    # 2. Non-stationarity shock metrics
    instability_index = 0.0
    adaptation_latency = N
    shock_drop = 0.0

    if shock_step is not None and 0 < shock_step < N:
        # Pre-shock statistics (last 20 episodes before shock)
        pre_start = max(0, shock_step - 20)
        pre_shock_mean = float(np.mean(returns_arr[pre_start:shock_step]))

        # Post-shock statistics
        post_shock_slice = returns_arr[shock_step:]
        if len(post_shock_slice) > 0:
            post_min = float(np.min(post_shock_slice[:min(15, len(post_shock_slice))]))
            shock_drop = pre_shock_mean - post_min

            # Instability: standard deviation in window around shock
            window_slice = returns_arr[max(0, shock_step - 15) : min(N, shock_step + 15)]
            instability_index = float(np.std(window_slice))

            # Adaptation latency: steps post-shock for rolling 4-episode mean to recover to 85% of pre-shock performance
            recovery_target = 0.85 * pre_shock_mean if pre_shock_mean > 0 else pre_shock_mean - 1.5
            w_rec = min(4, len(post_shock_slice))
            post_smoothed = np.convolve(post_shock_slice, np.ones(w_rec) / w_rec, mode="valid")
            recovered_indices = np.where(post_smoothed >= recovery_target)[0]
            adaptation_latency = int(recovered_indices[0] + w_rec - 1) if len(recovered_indices) > 0 else len(post_shock_slice)

    # 3. Communication cost
    avg_comm_cost = float(np.mean(comm_costs)) if comm_costs is not None and len(comm_costs) > 0 else 1.0

    return {
        "mean_return": mean_return,
        "max_return": max_return,
        "convergence_episode": convergence_episode,
        "instability_index": instability_index,
        "adaptation_latency": adaptation_latency,
        "shock_drop": shock_drop,
        "avg_communication_cost": avg_comm_cost,
    }
