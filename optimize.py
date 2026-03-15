"""Optuna Bayesian parameter optimization for TP/SL strategy.

Extracted from tick_analysis.py — refactored to use TP/SL exit logic.
"""

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from config import StrategyParams, FeeProfile

console = Console()


def optimize_parameters(
    btc_ts: np.ndarray, btc_prices: np.ndarray, btc_vols: np.ndarray,
    follower_ts: np.ndarray, follower_prices: np.ndarray, follower_vols: np.ndarray,
    fee_profile: FeeProfile,
    leverage: float = 1.0,
    n_trials: int = 300,
    seed: int = 42,
) -> tuple[dict, dict]:
    """Run Optuna Bayesian optimization for TP/SL strategy parameters.

    Returns (best_params: dict, study_summary: dict).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    bin_s = btc_ts[1] - btc_ts[0] if len(btc_ts) > 1 else 1.0
    n = min(len(btc_prices), len(follower_prices))
    fee_rt_pct = fee_profile.fee_per_leg * fee_profile.legs_per_trade * 100

    def objective(trial):
        btc_window_s = trial.suggest_float("btc_window_s", 3, 300, log=True)
        btc_threshold = trial.suggest_float("btc_threshold_pct", 0.05, 2.0, log=True)
        tp_pct = trial.suggest_float("tp_pct", 0.05, 1.0, log=True)
        sl_pct = trial.suggest_float("sl_pct", 0.1, 2.0, log=True)
        max_hold_s = trial.suggest_float("max_hold_s", 30, 900, log=True)
        cooldown_s = trial.suggest_float("cooldown_s", 10, 300, log=True)
        min_vol_ratio = trial.suggest_float("min_volume_ratio", 1.0, 5.0)

        window_bins = max(1, int(btc_window_s / bin_s))
        max_hold_bins = max(1, int(max_hold_s / bin_s))
        cooldown_bins = max(1, int(cooldown_s / bin_s))
        exec_delay = 1

        trades_net = []
        last_trade_idx = -cooldown_bins

        for i in range(window_bins, n - max_hold_bins - exec_delay):
            if i - last_trade_idx < cooldown_bins:
                continue

            btc_move = (btc_prices[i] - btc_prices[i - window_bins]) / btc_prices[i - window_bins] * 100
            if abs(btc_move) < btc_threshold:
                continue

            # Volume filter
            if min_vol_ratio > 1.0 and i >= 120:
                burst = np.sum(btc_vols[max(0, i - 5):i]) / 5
                base = np.sum(btc_vols[max(0, i - 120):i]) / 120
                if base > 0 and burst / base < min_vol_ratio:
                    continue

            direction = "LONG" if btc_move > 0 else "SHORT"
            entry_idx = i + exec_delay
            if entry_idx >= n - 1:
                continue

            entry_price = follower_prices[entry_idx]
            end_idx = min(entry_idx + max_hold_bins, n - 1)

            # TP/SL exit scan
            exit_price = follower_prices[end_idx]
            for j in range(entry_idx + 1, end_idx + 1):
                if direction == "LONG":
                    ret = (follower_prices[j] - entry_price) / entry_price * 100
                else:
                    ret = (entry_price - follower_prices[j]) / entry_price * 100
                if ret >= tp_pct:
                    exit_price = follower_prices[j]
                    break
                if ret <= -sl_pct:
                    exit_price = follower_prices[j]
                    break

            if direction == "LONG":
                follower_ret = (exit_price - entry_price) / entry_price * 100
            else:
                follower_ret = (entry_price - exit_price) / entry_price * 100

            net = follower_ret * leverage - fee_rt_pct
            trades_net.append(net)
            last_trade_idx = i

        if len(trades_net) < 3:
            return -10

        total = sum(trades_net)
        wr = sum(1 for t in trades_net if t > 0) / len(trades_net)
        trade_factor = min(len(trades_net) / 5, 1.0)

        score = total * wr * trade_factor
        trial.report(score, 0)
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(),
    )

    console.print(f"  Running {n_trials} Optuna trials...")
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Optimizing...", total=n_trials)

        def callback(study, trial):
            progress.update(task, advance=1)

        study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)

    best = study.best_params
    console.print(f"\n  [bold green]Best score:[/] {study.best_value:.4f}")

    # Get top trials
    top_trials = []
    try:
        df = study.trials_dataframe()
        df = df.sort_values("value", ascending=False).head(10)
        for _, row in df.iterrows():
            top_trials.append({
                "number": int(row["number"]),
                "score": float(row["value"]),
                "params": {k.replace("params_", ""): float(row[k])
                           for k in row.index if k.startswith("params_")},
            })
    except Exception:
        pass

    # Parameter importance
    importance = {}
    try:
        importance = {k: float(v) for k, v in
                      optuna.importance.get_param_importances(study).items()}
    except Exception:
        pass

    summary = {
        "best_score": float(study.best_value),
        "best_params": best,
        "top_trials": top_trials,
        "param_importance": importance,
        "n_trials": n_trials,
    }

    return best, summary
