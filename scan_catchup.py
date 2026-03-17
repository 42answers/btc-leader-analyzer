"""Scan altcoins for micro-level catch-up dynamics with BTC.

Computes Pearson r at 1s, 3s, 5s, 10s, 30s, 60s return horizons
and classifies each coin's correlation profile.
"""

import sys
import numpy as np
from datetime import datetime, timezone
from scipy import stats
from data import load_aligned_pair

COINS = [
    "ETH", "SOL", "DOGE", "XRP", "ADA", "AVAX", "LINK", "DOT",
    "MATIC", "BNB", "NEAR", "ATOM", "UNI", "AAVE", "LTC", "BCH",
    "ETC", "FIL", "APT", "ARB", "OP", "INJ", "SUI", "PEPE",
    "SHIB", "WIF", "BONK", "FET", "RENDER", "TIA",
]

LAGS = [1, 3, 5, 10, 30, 60]  # seconds
DAYS = 3
START = datetime(2026, 3, 14, tzinfo=timezone.utc)  # 3 days ago


def compute_returns(prices: np.ndarray, lag: int) -> np.ndarray:
    """Compute log returns at a given lag (in seconds / bins)."""
    return np.log(prices[lag:] / prices[:-lag])


def analyze_coin(coin: str) -> dict | None:
    """Load BTC + coin pair and compute correlations at each lag."""
    btc_sym = "BTCUSDT"
    coin_sym = f"{coin}USDT"

    try:
        ts, btc_p, btc_v, coin_p, coin_v = load_aligned_pair(
            btc_sym, coin_sym, START, DAYS, bin_ms=1000
        )
    except Exception as e:
        print(f"  ERROR loading {coin}: {e}")
        return None

    if len(btc_p) < 120:
        print(f"  SKIP {coin}: too few data points ({len(btc_p)})")
        return None

    correlations = {}
    for lag in LAGS:
        btc_ret = compute_returns(btc_p, lag)
        coin_ret = compute_returns(coin_p, lag)
        n = min(len(btc_ret), len(coin_ret))
        if n < 100:
            correlations[lag] = 0.0
            continue
        r, p = stats.pearsonr(btc_ret[:n], coin_ret[:n])
        correlations[lag] = round(r, 4)

    # Classify
    r1 = correlations[1]
    peak_r = max(correlations.values())
    rise = peak_r - r1

    if r1 >= 0.2:
        profile = "strong"
    elif rise > 0.10:
        profile = "rising"
    elif peak_r < 0.15:
        profile = "flat-weak"
    else:
        profile = "moderate"

    return {
        "coin": coin,
        "correlations": correlations,
        "peak_r": peak_r,
        "rise": rise,
        "profile": profile,
    }


def main():
    print(f"\n{'='*90}")
    print(f"  BTC CATCH-UP DYNAMICS SCANNER")
    print(f"  {DAYS} days from {START.strftime('%Y-%m-%d')} | {len(COINS)} coins | lags: {LAGS}s")
    print(f"{'='*90}\n")

    results = []
    for i, coin in enumerate(COINS):
        print(f"\n[{i+1}/{len(COINS)}] Analyzing {coin}...")
        result = analyze_coin(coin)
        if result:
            results.append(result)

    # Sort by 1s correlation descending
    results.sort(key=lambda x: x["correlations"][1], reverse=True)

    # Print table
    print(f"\n\n{'='*90}")
    print(f"  RESULTS: Pearson r of log-returns (BTC vs ALT) at each lag")
    print(f"{'='*90}")
    header = f"{'Coin':<8}" + "".join(f"{'r@'+str(l)+'s':>8}" for l in LAGS) + f"{'peak':>8}  {'rise':>6}  {'profile':<12}"
    print(header)
    print("-" * len(header))

    for r in results:
        row = f"{r['coin']:<8}"
        for lag in LAGS:
            row += f"{r['correlations'][lag]:>8.4f}"
        row += f"{r['peak_r']:>8.4f}  {r['rise']:>6.4f}  {r['profile']:<12}"
        print(row)

    # Recommendations
    print(f"\n\n{'='*90}")
    print("  RECOMMENDATIONS FOR CATCH-UP STRATEGY")
    print(f"{'='*90}\n")

    strong = [r for r in results if r["profile"] == "strong"]
    rising = [r for r in results if r["profile"] == "rising"]
    moderate = [r for r in results if r["profile"] == "moderate"]
    weak = [r for r in results if r["profile"] == "flat-weak"]

    print(f"  STRONG (r>=0.2 at 1s) - Already correlated, instant movers:")
    if strong:
        for r in strong:
            print(f"    {r['coin']:>8}  r@1s={r['correlations'][1]:.4f}  peak={r['peak_r']:.4f}")
    else:
        print(f"    (none)")

    print(f"\n  RISING (correlation grows >0.10 from 1s to peak) - Best for catch-up:")
    if rising:
        for r in rising:
            peak_lag = max(LAGS, key=lambda l: r['correlations'][l])
            print(f"    {r['coin']:>8}  r@1s={r['correlations'][1]:.4f} -> peak r@{peak_lag}s={r['peak_r']:.4f}  (rise={r['rise']:.4f})")
    else:
        print(f"    (none)")

    print(f"\n  MODERATE (some correlation but no clear pattern):")
    if moderate:
        for r in moderate:
            print(f"    {r['coin']:>8}  r@1s={r['correlations'][1]:.4f}  peak={r['peak_r']:.4f}")
    else:
        print(f"    (none)")

    print(f"\n  FLAT-WEAK (r<0.15 everywhere) - Not suitable:")
    if weak:
        for r in weak:
            print(f"    {r['coin']:>8}  r@1s={r['correlations'][1]:.4f}  peak={r['peak_r']:.4f}")
    else:
        print(f"    (none)")

    suitable = strong + rising
    print(f"\n  >>> SUITABLE COINS FOR CATCH-UP STRATEGY ({len(suitable)} total):")
    if suitable:
        print(f"      {', '.join(r['coin'] for r in suitable)}")
    else:
        print(f"      (none found)")

    print()


if __name__ == "__main__":
    main()
