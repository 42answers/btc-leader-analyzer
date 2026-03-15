"""BTC impulse detection and follower response measurement.

Extracted from tick_analysis.py — renamed xrp → follower throughout.
"""

import numpy as np
from config import ImpulseEvent


def detect_impulse_events(
    btc_ts: np.ndarray, btc_prices: np.ndarray, btc_vols: np.ndarray,
    follower_ts: np.ndarray, follower_prices: np.ndarray, follower_vols: np.ndarray,
    windows_s: list[int] = None,
    btc_thresholds: list[float] = None,
    response_horizons_s: list[int] = None,
    min_gap_s: int = 60,
) -> list[ImpulseEvent]:
    """Detect BTC impulse events and measure follower response.

    For each (window, threshold) combination, finds moments where BTC moved
    more than threshold% in window seconds. Then measures what the follower did.
    """
    if windows_s is None:
        windows_s = [5, 10, 30, 60, 120, 300]
    if btc_thresholds is None:
        btc_thresholds = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    if response_horizons_s is None:
        response_horizons_s = [10, 30, 60, 120, 300, 600]

    events = []
    bin_s = btc_ts[1] - btc_ts[0] if len(btc_ts) > 1 else 1.0

    for window_s in windows_s:
        window_bins = max(1, int(window_s / bin_s))

        if window_bins >= len(btc_prices):
            continue

        btc_returns = (btc_prices[window_bins:] - btc_prices[:-window_bins]) / btc_prices[:-window_bins] * 100

        for threshold in btc_thresholds:
            up_events = np.where(btc_returns > threshold)[0]
            down_events = np.where(btc_returns < -threshold)[0]

            for idx_arr, direction in [(up_events, "UP"), (down_events, "DOWN")]:
                if len(idx_arr) == 0:
                    continue

                # Deduplicate: take first event in each cluster
                deduped = [idx_arr[0]]
                for idx in idx_arr[1:]:
                    if (idx - deduped[-1]) * bin_s >= min_gap_s:
                        deduped.append(idx)

                for raw_idx in deduped:
                    idx = raw_idx + window_bins
                    if idx >= len(btc_prices):
                        continue

                    btc_move = btc_returns[raw_idx]
                    event_time = btc_ts[idx]

                    follower_start_idx = idx - window_bins
                    if follower_start_idx < 0 or idx >= len(follower_prices):
                        continue
                    follower_same = (follower_prices[idx] - follower_prices[follower_start_idx]) / follower_prices[follower_start_idx] * 100

                    already_followed = (follower_same / btc_move * 100) if btc_move != 0 else 0

                    evt = ImpulseEvent(
                        timestamp_s=event_time,
                        btc_move_pct=btc_move,
                        window_s=window_s,
                        direction=direction,
                        follower_move_same_window=follower_same,
                        follower_already_followed_pct=already_followed,
                    )

                    # Measure lag via cross-correlation
                    lag_window = int(60 / bin_s)
                    start = max(0, idx - lag_window)
                    end = min(len(btc_prices), idx + lag_window)

                    if end - start > 10:
                        btc_chunk = btc_prices[start:end]
                        f_chunk = follower_prices[start:end]

                        btc_norm = (btc_chunk - btc_chunk.mean()) / (btc_chunk.std() + 1e-12)
                        f_norm = (f_chunk - f_chunk.mean()) / (f_chunk.std() + 1e-12)

                        max_shift = min(int(30 / bin_s), len(btc_norm) // 4)
                        best_corr = -1
                        best_lag = 0
                        for lag in range(0, max_shift + 1):
                            if lag < len(btc_norm):
                                b = btc_norm[:len(btc_norm) - lag] if lag > 0 else btc_norm
                                x = f_norm[lag:] if lag > 0 else f_norm
                                n = min(len(b), len(x))
                                if n > 5:
                                    c = np.dot(b[:n], x[:n]) / n
                                    if c > best_corr:
                                        best_corr = c
                                        best_lag = lag

                        evt.measured_lag_ms = best_lag * bin_s * 1000

                    # Measure follower response at various horizons
                    for horizon_s in response_horizons_s:
                        horizon_bins = int(horizon_s / bin_s)
                        future_idx = idx + horizon_bins

                        if future_idx < len(follower_prices) and future_idx < len(btc_prices):
                            f_future = (follower_prices[future_idx] - follower_prices[idx]) / follower_prices[idx] * 100
                            btc_future = (btc_prices[future_idx] - btc_prices[idx]) / btc_prices[idx] * 100

                            evt.follower_response[horizon_s] = f_future
                            evt.btc_continuation[horizon_s] = btc_future

                            if direction == "UP":
                                evt.relative_gain[horizon_s] = f_future - btc_future
                            else:
                                evt.relative_gain[horizon_s] = btc_future - f_future

                    events.append(evt)

    return events


def summarize_impulse_events(events: list[ImpulseEvent]) -> dict:
    """Compute summary statistics for impulse events.

    Returns structured dict suitable for JSON serialization.
    """
    if not events:
        return {"total_events": 0}

    lags = [e.measured_lag_ms for e in events]
    followed = [e.follower_already_followed_pct for e in events]

    # Group by window
    by_window = {}
    for ws in sorted(set(e.window_s for e in events)):
        subset = [e for e in events if e.window_s == ws]
        sub_lags = [e.measured_lag_ms for e in subset]
        sub_followed = [e.follower_already_followed_pct for e in subset]

        by_window[ws] = {
            "count": len(subset),
            "median_lag_ms": float(np.median(sub_lags)) if sub_lags else 0,
            "mean_followed_pct": float(np.mean(sub_followed)) if sub_followed else 0,
        }

    # Response by horizon
    response_summary = {}
    for horizon in sorted(set(h for e in events for h in e.relative_gain)):
        gains = [e.relative_gain[horizon] for e in events if horizon in e.relative_gain]
        if gains:
            response_summary[horizon] = {
                "mean_relative_gain": float(np.mean(gains)),
                "win_rate": float(sum(1 for g in gains if g > 0) / len(gains) * 100),
                "count": len(gains),
            }

    # Binary response pattern
    large_events = [e for e in events if abs(e.btc_move_pct) >= 0.3]
    if large_events:
        fol_vals = [e.follower_already_followed_pct for e in large_events]
        binary = {
            "no_response_pct": sum(1 for f in fol_vals if abs(f) < 5) / len(fol_vals) * 100,
            "partial_pct": sum(1 for f in fol_vals if 5 <= abs(f) < 50) / len(fol_vals) * 100,
            "full_follow_pct": sum(1 for f in fol_vals if 50 <= abs(f) <= 100) / len(fol_vals) * 100,
            "overshoot_pct": sum(1 for f in fol_vals if abs(f) > 100) / len(fol_vals) * 100,
        }
    else:
        binary = {}

    return {
        "total_events": len(events),
        "median_lag_ms": float(np.median(lags)),
        "mean_followed_pct": float(np.mean(followed)),
        "by_window": by_window,
        "response_by_horizon": response_summary,
        "binary_response_pattern": binary,
    }
