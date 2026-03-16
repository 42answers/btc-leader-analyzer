"""TP/SL directional strategy simulation.

This is the "David's strategy" — directional trades with take-profit / stop-loss exits.
Core new logic: not extracted from existing files, formalizes the inline scripts.
"""

import numpy as np
from config import StrategyParams, FeeProfile, TradeResult


def _find_exit(prices: np.ndarray, entry_idx: int, direction: str,
               tp_pct: float, sl_pct: float, max_hold_bins: int,
               ) -> tuple[int, float, str]:
    """Scan forward tick-by-tick for TP/SL/timeout exit.

    Returns (exit_idx, exit_price, exit_reason).
    """
    entry_price = prices[entry_idx]
    end_idx = min(entry_idx + max_hold_bins, len(prices) - 1)

    for i in range(entry_idx + 1, end_idx + 1):
        if direction == "LONG":
            ret_pct = (prices[i] - entry_price) / entry_price * 100
        else:
            ret_pct = (entry_price - prices[i]) / entry_price * 100

        if ret_pct >= tp_pct:
            return i, prices[i], "TP"
        if ret_pct <= -sl_pct:
            return i, prices[i], "SL"

    return end_idx, prices[end_idx], "TIMEOUT"


def simulate_tpsl_strategy(
    btc_ts: np.ndarray, btc_prices: np.ndarray, btc_vols: np.ndarray,
    follower_ts: np.ndarray, follower_prices: np.ndarray, follower_vols: np.ndarray,
    params: StrategyParams,
    fee_profile: FeeProfile,
    leverage: float = 1.0,
) -> dict:
    """Simulate directional TP/SL strategy on tick data.

    For each BTC impulse above threshold:
      1. Wait execution_delay_s
      2. Enter LONG on follower (for UP impulse) or SHORT (for DOWN)
      3. Monitor tick-by-tick for TP hit, SL hit, or timeout
      4. Record result with fees

    Returns dict with trades, metrics, and daily breakdown.
    """
    bin_s = btc_ts[1] - btc_ts[0] if len(btc_ts) > 1 else 1.0
    n = min(len(btc_prices), len(follower_prices))

    window_bins = max(1, int(params.btc_window_s / bin_s))
    cooldown_bins = max(1, int(params.cooldown_s / bin_s))
    max_hold_bins = max(1, int(params.max_hold_s / bin_s))
    exec_delay = params.execution_delay_s
    vol_baseline = max(1, int(params.vol_baseline_s / bin_s))
    vol_burst = max(1, int(params.vol_burst_s / bin_s))

    fee_rt_pct = fee_profile.fee_per_leg * fee_profile.legs_per_trade * 100
    slip = params.slippage_bps / 10_000

    trades = []
    last_trade_idx = -cooldown_bins

    for i in range(window_bins, n - max_hold_bins - exec_delay):
        if i - last_trade_idx < cooldown_bins:
            continue

        # BTC move in window
        btc_move = (btc_prices[i] - btc_prices[i - window_bins]) / btc_prices[i - window_bins] * 100
        if abs(btc_move) < params.btc_threshold_pct:
            continue

        # Volume filter
        if params.min_volume_ratio > 1.0 and i >= vol_baseline:
            burst_vol = np.sum(btc_vols[max(0, i - vol_burst):i]) / max(vol_burst * bin_s, 0.001)
            base_vol = np.sum(btc_vols[max(0, i - vol_baseline):i]) / max(vol_baseline * bin_s, 0.001)
            if base_vol > 0 and burst_vol / base_vol < params.min_volume_ratio:
                continue

        direction = "LONG" if btc_move > 0 else "SHORT"

        # Apply execution delay
        entry_idx = i + exec_delay
        if entry_idx >= n - 1:
            continue

        # Find exit via TP/SL/timeout
        exit_idx, exit_price, exit_reason = _find_exit(
            follower_prices, entry_idx, direction,
            params.tp_pct, params.sl_pct, max_hold_bins,
        )

        entry_price = follower_prices[entry_idx]

        # Apply slippage: worse fill on both legs
        if slip > 0:
            if direction == "LONG":
                entry_adj = entry_price * (1 + slip)
                exit_adj = exit_price * (1 - slip)
            else:
                entry_adj = entry_price * (1 - slip)
                exit_adj = exit_price * (1 + slip)
        else:
            entry_adj, exit_adj = entry_price, exit_price

        if direction == "LONG":
            follower_return = (exit_adj - entry_adj) / entry_adj * 100
        else:
            follower_return = (entry_adj - exit_adj) / entry_adj * 100

        gross_pnl = follower_return * leverage
        net_pnl = gross_pnl - fee_rt_pct

        trades.append(TradeResult(
            entry_time=btc_ts[entry_idx],
            exit_time=btc_ts[exit_idx],
            direction=direction,
            btc_impulse_pct=btc_move,
            follower_entry_price=entry_price,
            follower_exit_price=exit_price,
            follower_return_pct=follower_return,
            exit_reason=exit_reason,
            gross_pnl_pct=gross_pnl,
            fee_pct=fee_rt_pct,
            net_pnl_pct=net_pnl,
        ))

        last_trade_idx = i

    if not trades:
        return {
            "trades": [],
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_gross_pct": 0.0,
            "avg_net_pct": 0.0,
            "total_gross_pct": 0.0,
            "total_net_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "tp_rate": 0.0,
            "sl_rate": 0.0,
            "timeout_rate": 0.0,
        }

    net_pnls = [t.net_pnl_pct for t in trades]
    gross_pnls = [t.gross_pnl_pct for t in trades]
    winners = sum(1 for p in net_pnls if p > 0)

    # Compute max drawdown on compounded equity
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for pnl in net_pnls:
        equity *= (1 + pnl / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        "trades": trades,
        "total_trades": len(trades),
        "win_rate": winners / len(trades) * 100,
        "avg_gross_pct": float(np.mean(gross_pnls)),
        "avg_net_pct": float(np.mean(net_pnls)),
        "total_gross_pct": float(np.sum(gross_pnls)),
        "total_net_pct": float(np.sum(net_pnls)),
        "max_drawdown_pct": max_dd,
        "tp_rate": sum(1 for t in trades if t.exit_reason == "TP") / len(trades) * 100,
        "sl_rate": sum(1 for t in trades if t.exit_reason == "SL") / len(trades) * 100,
        "timeout_rate": sum(1 for t in trades if t.exit_reason == "TIMEOUT") / len(trades) * 100,
        "leverage": leverage,
        "fee_profile": fee_profile.name,
        "fee_rt_pct": fee_rt_pct,
    }
