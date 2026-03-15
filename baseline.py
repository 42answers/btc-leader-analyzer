"""Random baseline Monte Carlo comparison.

Compares BTC-triggered entries against random entry times to prove
the trigger adds genuine value.
"""

import numpy as np
from config import StrategyParams, FeeProfile, TradeResult


def _simulate_random_entries(
    follower_prices: np.ndarray,
    n_trades: int,
    direction_probs: dict,
    params: StrategyParams,
    fee_profile: FeeProfile,
    leverage: float,
    rng: np.random.Generator,
) -> dict:
    """Run one random baseline trial with same TP/SL logic."""
    bin_s = 1.0  # assume 1-second bins
    max_hold_bins = max(1, int(params.max_hold_s / bin_s))
    cooldown_bins = max(1, int(params.cooldown_s / bin_s))
    fee_rt_pct = fee_profile.fee_per_leg * fee_profile.legs_per_trade * 100

    # Generate random entry points with cooldown spacing
    margin = max_hold_bins + params.execution_delay_s + 10
    valid_range = len(follower_prices) - margin
    if valid_range <= 0:
        return {"win_rate": 0, "avg_net_pct": 0, "total_net_pct": 0}

    entries = []
    last_entry = -cooldown_bins
    attempts = 0
    while len(entries) < n_trades and attempts < n_trades * 20:
        idx = rng.integers(params.execution_delay_s, valid_range)
        if idx - last_entry >= cooldown_bins:
            entries.append(idx)
            last_entry = idx
        attempts += 1

    entries.sort()

    # Random direction matching real distribution
    long_prob = direction_probs.get("LONG", 0.5)

    net_pnls = []
    for entry_idx in entries:
        direction = "LONG" if rng.random() < long_prob else "SHORT"
        entry_price = follower_prices[entry_idx]
        end_idx = min(entry_idx + max_hold_bins, len(follower_prices) - 1)

        exit_price = entry_price
        for i in range(entry_idx + 1, end_idx + 1):
            if direction == "LONG":
                ret = (follower_prices[i] - entry_price) / entry_price * 100
            else:
                ret = (entry_price - follower_prices[i]) / entry_price * 100

            if ret >= params.tp_pct:
                exit_price = follower_prices[i]
                break
            if ret <= -params.sl_pct:
                exit_price = follower_prices[i]
                break
        else:
            exit_price = follower_prices[end_idx]

        if direction == "LONG":
            follower_return = (exit_price - entry_price) / entry_price * 100
        else:
            follower_return = (entry_price - exit_price) / entry_price * 100

        net_pnl = follower_return * leverage - fee_rt_pct
        net_pnls.append(net_pnl)

    if not net_pnls:
        return {"win_rate": 0, "avg_net_pct": 0, "total_net_pct": 0}

    return {
        "win_rate": sum(1 for p in net_pnls if p > 0) / len(net_pnls) * 100,
        "avg_net_pct": float(np.mean(net_pnls)),
        "total_net_pct": float(np.sum(net_pnls)),
    }


def random_baseline_comparison(
    follower_prices: np.ndarray,
    strategy_trades: list[TradeResult],
    params: StrategyParams,
    fee_profile: FeeProfile,
    leverage: float = 1.0,
    n_trials: int = 500,
    seed: int = 42,
) -> dict:
    """Compare BTC-triggered entries against random entry times.

    Returns percentile rank, p-value, and distribution of random results.
    """
    if not strategy_trades:
        return {
            "strategy_win_rate": 0,
            "strategy_avg_net": 0,
            "percentile_rank": 0,
            "p_value": 1.0,
            "n_trials": n_trials,
        }

    n_trades = len(strategy_trades)
    strategy_wr = sum(1 for t in strategy_trades if t.net_pnl_pct > 0) / n_trades * 100
    strategy_avg = float(np.mean([t.net_pnl_pct for t in strategy_trades]))

    # Direction distribution from real trades
    long_count = sum(1 for t in strategy_trades if t.direction == "LONG")
    direction_probs = {"LONG": long_count / n_trades}

    rng = np.random.default_rng(seed)

    random_wrs = []
    random_avgs = []

    for _ in range(n_trials):
        result = _simulate_random_entries(
            follower_prices, n_trades, direction_probs,
            params, fee_profile, leverage, rng,
        )
        random_wrs.append(result["win_rate"])
        random_avgs.append(result["avg_net_pct"])

    # How often does strategy beat random?
    beats_wr = sum(1 for r in random_wrs if strategy_wr > r)
    beats_avg = sum(1 for r in random_avgs if strategy_avg > r)

    percentile_wr = beats_wr / n_trials * 100
    percentile_avg = beats_avg / n_trials * 100

    # p-value: fraction of random trials that beat strategy
    p_value = 1 - percentile_avg / 100

    return {
        "strategy_win_rate": strategy_wr,
        "strategy_avg_net": strategy_avg,
        "random_mean_win_rate": float(np.mean(random_wrs)),
        "random_std_win_rate": float(np.std(random_wrs)),
        "random_mean_avg_net": float(np.mean(random_avgs)),
        "random_std_avg_net": float(np.std(random_avgs)),
        "percentile_rank_wr": percentile_wr,
        "percentile_rank_avg": percentile_avg,
        "p_value": p_value,
        "n_trials": n_trials,
        "random_wr_distribution": random_wrs,
        "random_avg_distribution": random_avgs,
    }
