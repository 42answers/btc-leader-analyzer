"""Walk-forward validation for TP/SL strategy.

Splits the date range into rolling train/test windows.
For each fold: optimize on train, evaluate on test, roll forward.
Aggregates out-of-sample results to give a realistic performance estimate.
"""

from datetime import datetime, timedelta, timezone

import numpy as np

import data as data_mod
import strategy as strategy_mod
from optimize import optimize_parameters
from config import StrategyParams, FeeProfile


def run_walk_forward(
    leader_symbol: str,
    follower_symbol: str,
    start_date: datetime,
    total_days: int,
    train_days: int,
    test_days: int,
    fee_profile: FeeProfile,
    slippage_bps: float = 0.0,
    exec_delay_s: int = 1,
    n_trials: int = 150,
    load_fn=None,
) -> dict:
    """Run walk-forward validation across rolling windows.

    Args:
        leader_symbol:  e.g. "BTC/USDT"
        follower_symbol: e.g. "DOGE/USDT"
        start_date:     start of the full analysis period
        total_days:     total days of data
        train_days:     days per training window
        test_days:      days per test window
        fee_profile:    trading fee profile
        slippage_bps:   slippage in basis points per leg
        exec_delay_s:   execution delay in seconds
        n_trials:       Optuna trials per fold (fewer for speed)
        load_fn:        cached data loader (signature: leader, follower, start_iso, days)

    Returns dict with folds, aggregate OOS metrics, and degradation %.
    """
    if load_fn is None:
        def load_fn(leader, follower, start_iso, days):
            sd = datetime.fromisoformat(start_iso)
            return data_mod.load_aligned_pair(leader, follower, sd, days)

    folds = []
    fold_start = start_date
    fold_num = 0

    while True:
        train_end = fold_start + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)

        # Check if test window fits within total period
        total_end = start_date + timedelta(days=total_days)
        if test_end > total_end:
            break

        fold_num += 1

        # ── Load training data ───────────────────────────────────────
        try:
            ts_train, btc_train, btcv_train, f_train, fv_train = load_fn(
                leader_symbol, follower_symbol,
                fold_start.isoformat(), train_days,
            )
        except Exception as e:
            folds.append({
                "fold": fold_num,
                "train_range": f"{fold_start.strftime('%b %d')}–{train_end.strftime('%b %d')}",
                "test_range": f"{train_end.strftime('%b %d')}–{test_end.strftime('%b %d')}",
                "error": str(e),
            })
            fold_start = fold_start + timedelta(days=test_days)
            continue

        # ── Optimize on training data ────────────────────────────────
        best_params, opt_summary = optimize_parameters(
            ts_train, btc_train, btcv_train,
            ts_train, f_train, fv_train,
            fee_profile, leverage=1.0, n_trials=n_trials,
            slippage_bps=slippage_bps,
        )

        params = StrategyParams(
            btc_window_s=best_params["btc_window_s"],
            btc_threshold_pct=best_params["btc_threshold_pct"],
            tp_pct=best_params["tp_pct"],
            sl_pct=best_params["sl_pct"],
            max_hold_s=best_params["max_hold_s"],
            cooldown_s=best_params["cooldown_s"],
            min_volume_ratio=best_params["min_volume_ratio"],
            execution_delay_s=exec_delay_s,
            slippage_bps=slippage_bps,
        )

        # ── In-sample result ─────────────────────────────────────────
        is_result = strategy_mod.simulate_tpsl_strategy(
            ts_train, btc_train, btcv_train,
            ts_train, f_train, fv_train,
            params, fee_profile, leverage=1.0,
        )

        # ── Load test data ───────────────────────────────────────────
        try:
            ts_test, btc_test, btcv_test, f_test, fv_test = load_fn(
                leader_symbol, follower_symbol,
                train_end.isoformat(), test_days,
            )
        except Exception as e:
            folds.append({
                "fold": fold_num,
                "train_range": f"{fold_start.strftime('%b %d')}–{train_end.strftime('%b %d')}",
                "test_range": f"{train_end.strftime('%b %d')}–{test_end.strftime('%b %d')}",
                "error": f"Test data: {e}",
            })
            fold_start = fold_start + timedelta(days=test_days)
            continue

        # ── Out-of-sample result ─────────────────────────────────────
        oos_result = strategy_mod.simulate_tpsl_strategy(
            ts_test, btc_test, btcv_test,
            ts_test, f_test, fv_test,
            params, fee_profile, leverage=1.0,
        )

        folds.append({
            "fold": fold_num,
            "train_range": f"{fold_start.strftime('%b %d')}–{train_end.strftime('%b %d')}",
            "test_range": f"{train_end.strftime('%b %d')}–{test_end.strftime('%b %d')}",
            "params": best_params,
            "in_sample": {
                "win_rate": is_result["win_rate"],
                "avg_net": is_result["avg_net_pct"],
                "total_return": is_result["total_net_pct"],
                "n_trades": is_result["total_trades"],
            },
            "out_of_sample": {
                "win_rate": oos_result["win_rate"],
                "avg_net": oos_result["avg_net_pct"],
                "total_return": oos_result["total_net_pct"],
                "n_trades": oos_result["total_trades"],
            },
        })

        # Roll forward by test_days
        fold_start = fold_start + timedelta(days=test_days)

    # ── Aggregate OOS results ────────────────────────────────────────
    valid_folds = [f for f in folds if "out_of_sample" in f]

    if not valid_folds:
        return {
            "folds": folds,
            "aggregate_oos": {"win_rate": 0, "avg_net": 0, "total_return": 0, "n_trades": 0},
            "aggregate_is": {"win_rate": 0, "avg_net": 0, "total_return": 0, "n_trades": 0},
            "degradation_pct": 0,
            "n_folds": len(folds),
        }

    oos_wrs = [f["out_of_sample"]["win_rate"] for f in valid_folds]
    oos_avgs = [f["out_of_sample"]["avg_net"] for f in valid_folds]
    oos_totals = [f["out_of_sample"]["total_return"] for f in valid_folds]
    oos_trades = [f["out_of_sample"]["n_trades"] for f in valid_folds]

    is_wrs = [f["in_sample"]["win_rate"] for f in valid_folds]
    is_avgs = [f["in_sample"]["avg_net"] for f in valid_folds]
    is_totals = [f["in_sample"]["total_return"] for f in valid_folds]
    is_trades = [f["in_sample"]["n_trades"] for f in valid_folds]

    agg_oos = {
        "win_rate": float(np.mean(oos_wrs)),
        "avg_net": float(np.mean(oos_avgs)),
        "total_return": float(np.sum(oos_totals)),
        "n_trades": int(np.sum(oos_trades)),
    }
    agg_is = {
        "win_rate": float(np.mean(is_wrs)),
        "avg_net": float(np.mean(is_avgs)),
        "total_return": float(np.sum(is_totals)),
        "n_trades": int(np.sum(is_trades)),
    }

    # Degradation: how much worse is OOS vs IS
    if agg_is["avg_net"] != 0:
        degradation = (agg_is["avg_net"] - agg_oos["avg_net"]) / abs(agg_is["avg_net"]) * 100
    else:
        degradation = 0

    return {
        "folds": folds,
        "aggregate_oos": agg_oos,
        "aggregate_is": agg_is,
        "degradation_pct": float(degradation),
        "n_folds": len(valid_folds),
    }
