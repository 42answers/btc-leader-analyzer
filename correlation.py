"""Market-structure & correlation analysis between BTC and a follower coin."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats
from scipy.signal import correlate as sp_correlate


# ── helpers ──────────────────────────────────────────────────────

def _log_returns(prices: np.ndarray) -> np.ndarray:
    """1-second log returns, clipping out zero/negative prices."""
    safe = np.where(prices > 0, prices, np.nan)
    lr = np.diff(np.log(safe))
    return np.nan_to_num(lr, nan=0.0)


# ── 1. Static correlation metrics ────────────────────────────────

def compute_correlation_metrics(
    btc_prices: np.ndarray,
    follower_prices: np.ndarray,
    btc_vols: np.ndarray,
    follower_vols: np.ndarray,
) -> dict:
    """Compute static correlation metrics between BTC and follower.

    Returns dict with:
        pearson_returns, spearman_returns, r_squared,
        beta, alpha, beta_up, beta_down,
        relative_volatility, volume_correlation
    """
    btc_ret = _log_returns(btc_prices)
    f_ret = _log_returns(follower_prices)

    # Pearson
    pearson = float(np.corrcoef(btc_ret, f_ret)[0, 1])

    # Spearman
    spearman, _ = sp_stats.spearmanr(btc_ret, f_ret)
    spearman = float(spearman)

    r_squared = pearson ** 2

    # Beta & Alpha via OLS
    btc_var = np.var(btc_ret)
    if btc_var > 0:
        beta = float(np.cov(btc_ret, f_ret)[0, 1] / btc_var)
    else:
        beta = 0.0
    alpha = float(np.mean(f_ret) - beta * np.mean(btc_ret))

    # Asymmetric beta
    up_mask = btc_ret > 0
    dn_mask = btc_ret < 0

    def _beta_subset(mask):
        b = btc_ret[mask]
        f = f_ret[mask]
        v = np.var(b)
        return float(np.cov(b, f)[0, 1] / v) if v > 0 and len(b) > 10 else 0.0

    beta_up = _beta_subset(up_mask)
    beta_down = _beta_subset(dn_mask)

    # Relative volatility
    btc_std = np.std(btc_ret)
    f_std = np.std(f_ret)
    relative_volatility = float(f_std / btc_std) if btc_std > 0 else 0.0

    # Volume correlation
    vol_corr = float(np.corrcoef(btc_vols, follower_vols)[0, 1])

    return {
        "pearson_returns": pearson,
        "spearman_returns": spearman,
        "r_squared": r_squared,
        "beta": beta,
        "alpha": alpha,
        "beta_up": beta_up,
        "beta_down": beta_down,
        "relative_volatility": relative_volatility,
        "volume_correlation": vol_corr,
    }


# ── 2. Rolling correlation ──────────────────────────────────────

def compute_rolling_correlation(
    btc_prices: np.ndarray,
    follower_prices: np.ndarray,
    timestamps: np.ndarray,
    window_s: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised rolling Pearson of 1-second log returns.

    Uses cumulative-sum identity for O(n) computation.
    Returns (timestamps_out, rolling_corr) — NaN-trimmed.
    """
    x = _log_returns(btc_prices)
    y = _log_returns(follower_prices)
    n = window_s

    if len(x) < n + 1:
        return timestamps[:0], np.array([])

    # cumulative sums
    cx = np.cumsum(x)
    cy = np.cumsum(y)
    cx2 = np.cumsum(x * x)
    cy2 = np.cumsum(y * y)
    cxy = np.cumsum(x * y)

    # windowed sums via differences
    def _w(cs):
        return cs[n:] - np.concatenate(([0.0], cs[:-1]))[: len(cs) - n + 1][: len(cs[n:])]

    # Proper windowed sums
    sx = cx[n - 1:] - np.concatenate(([0.0], cx[:-1]))[:len(cx[n - 1:])]
    sy = cy[n - 1:] - np.concatenate(([0.0], cy[:-1]))[:len(cy[n - 1:])]
    sx2 = cx2[n - 1:] - np.concatenate(([0.0], cx2[:-1]))[:len(cx2[n - 1:])]
    sy2 = cy2[n - 1:] - np.concatenate(([0.0], cy2[:-1]))[:len(cy2[n - 1:])]
    sxy = cxy[n - 1:] - np.concatenate(([0.0], cxy[:-1]))[:len(cxy[n - 1:])]

    num = n * sxy - sx * sy
    den = np.sqrt((n * sx2 - sx ** 2) * (n * sy2 - sy ** 2))

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(den > 0, num / den, 0.0)
    corr = np.clip(corr, -1.0, 1.0)

    # align timestamps (returns are 1 shorter than prices, window trims more)
    ts_out = timestamps[n:][:len(corr)]

    return ts_out, corr


# ── 3. Cross-correlation function (correlogram) ─────────────────

def compute_cross_correlation_function(
    btc_prices: np.ndarray,
    follower_prices: np.ndarray,
    max_lag_s: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Correlogram from -max_lag to +max_lag seconds.

    Negative lag → follower leads BTC.
    Positive lag → BTC leads follower.
    Returns (lags, correlations).
    """
    btc_ret = _log_returns(btc_prices)
    f_ret = _log_returns(follower_prices)

    lags = np.arange(-max_lag_s, max_lag_s + 1)
    corrs = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        if lag == 0:
            corrs[i] = np.corrcoef(btc_ret, f_ret)[0, 1]
        elif lag > 0:
            # BTC leads: compare btc_ret[:-lag] with f_ret[lag:]
            corrs[i] = np.corrcoef(btc_ret[:-lag], f_ret[lag:])[0, 1]
        else:
            # Follower leads: compare btc_ret[-lag:] with f_ret[:lag]
            alag = -lag
            corrs[i] = np.corrcoef(btc_ret[alag:], f_ret[:-alag])[0, 1]

    return lags, np.nan_to_num(corrs, nan=0.0)


# ── 4. Cross-coin comparison ────────────────────────────────────

def compute_cross_coin_comparison(
    coins: list[str],
    start_iso: str,
    days: int,
    load_fn,
) -> list[dict]:
    """Load data for multiple coins and compute comparison metrics.

    ``load_fn`` must match the signature of ``_load_pair_cached``
    in app.py: ``(leader, follower, start_iso, days) -> (ts, l_p, l_v, f_p, f_v)``.

    Each call returns BTC + follower already aligned, so we use the
    per-pair BTC data directly (avoids length-mismatch issues).
    """
    results = []
    for coin in coins:
        try:
            follower_sym = f"{coin}/USDT"
            _, bp, bv, fp, fv = load_fn("BTC/USDT", follower_sym, start_iso, days)

            metrics = compute_correlation_metrics(bp, fp, bv, fv)

            # quick lag via cross-correlation peak
            _, ccf = compute_cross_correlation_function(bp, fp, max_lag_s=30)
            lags_arr = np.arange(-30, 31)
            peak_idx = int(np.argmax(ccf))
            peak_lag = int(lags_arr[peak_idx])

            results.append({
                "coin": coin,
                "pearson": metrics["pearson_returns"],
                "beta": metrics["beta"],
                "median_lag_s": peak_lag,
                "relative_vol": metrics["relative_volatility"],
                "status": "ok",
            })
        except Exception as exc:
            results.append({
                "coin": coin,
                "pearson": 0.0,
                "beta": 0.0,
                "median_lag_s": 0,
                "relative_vol": 0.0,
                "status": f"error: {exc}",
            })
    return results
