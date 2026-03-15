"""Market regime classification (bull/bear/flat days).

Classifies each day and computes per-regime strategy statistics.
"""

from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
from config import DayRegime, TradeResult


def classify_daily_regimes(
    btc_ts: np.ndarray, btc_prices: np.ndarray,
    follower_ts: np.ndarray, follower_prices: np.ndarray,
    trades: list[TradeResult],
    bull_threshold: float = 0.5,
    bear_threshold: float = -0.5,
) -> list[DayRegime]:
    """Classify each day as BULL/BEAR/FLAT and compute per-regime strategy stats."""
    # Find day boundaries
    start_ts = btc_ts[0]
    end_ts = btc_ts[-1]

    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0)

    # Group trades by day
    trades_by_day = defaultdict(list)
    for t in trades:
        day = datetime.fromtimestamp(t.entry_time, tz=timezone.utc).strftime("%Y-%m-%d")
        trades_by_day[day].append(t)

    regimes = []
    current = start_dt
    from datetime import timedelta
    while current <= end_dt:
        day_str = current.strftime("%Y-%m-%d")
        day_start_s = current.timestamp()
        day_end_s = (current + timedelta(days=1)).timestamp()

        # Find price at start and end of day
        btc_day_mask = (btc_ts >= day_start_s) & (btc_ts < day_end_s)
        f_day_mask = (follower_ts >= day_start_s) & (follower_ts < day_end_s)

        if btc_day_mask.sum() < 2 or f_day_mask.sum() < 2:
            current += timedelta(days=1)
            continue

        btc_day_prices = btc_prices[btc_day_mask]
        f_day_prices = follower_prices[f_day_mask]

        btc_ret = (btc_day_prices[-1] - btc_day_prices[0]) / btc_day_prices[0] * 100
        f_ret = (f_day_prices[-1] - f_day_prices[0]) / f_day_prices[0] * 100

        if f_ret > bull_threshold:
            regime = "BULL"
        elif f_ret < bear_threshold:
            regime = "BEAR"
        else:
            regime = "FLAT"

        day_trades = trades_by_day.get(day_str, [])
        if day_trades:
            day_nets = [t.net_pnl_pct for t in day_trades]
            wr = sum(1 for p in day_nets if p > 0) / len(day_nets) * 100
            avg_ret = float(np.mean(day_nets))
        else:
            wr = 0.0
            avg_ret = 0.0

        regimes.append(DayRegime(
            date=day_str,
            btc_return_pct=float(btc_ret),
            follower_return_pct=float(f_ret),
            regime=regime,
            trades_count=len(day_trades),
            win_rate=wr,
            avg_return_pct=avg_ret,
        ))

        current += timedelta(days=1)

    return regimes


def regime_summary(regimes: list[DayRegime]) -> dict:
    """Aggregate statistics by regime type.

    Returns {regime_name: {count, avg_win_rate, avg_return, total_trades, ...}}.
    """
    result = {}
    for regime_type in ["BULL", "BEAR", "FLAT"]:
        subset = [r for r in regimes if r.regime == regime_type]
        if not subset:
            result[regime_type] = {"days": 0, "total_trades": 0}
            continue

        trading_days = [r for r in subset if r.trades_count > 0]
        result[regime_type] = {
            "days": len(subset),
            "total_trades": sum(r.trades_count for r in subset),
            "avg_win_rate": float(np.mean([r.win_rate for r in trading_days])) if trading_days else 0,
            "avg_daily_return": float(np.mean([r.avg_return_pct for r in trading_days])) if trading_days else 0,
            "avg_follower_drift": float(np.mean([r.follower_return_pct for r in subset])),
            "avg_btc_drift": float(np.mean([r.btc_return_pct for r in subset])),
        }

    return result
