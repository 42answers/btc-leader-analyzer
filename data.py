"""Data fetching, caching, and VWAP binning.

Reused from tick_analysis.py — these functions are already symbol-agnostic.
"""

import time
import json
import gzip
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import requests
from rich.console import Console

console = Console()

# Cache: prefer shared cache with xrp-btc-analyzer, fall back to local
_shared = Path(__file__).parent.parent / "xrp-btc-analyzer" / ".tick_cache"
CACHE_DIR = _shared if _shared.exists() else Path(__file__).parent / ".tick_cache"


def fetch_agg_trades(symbol: str, start_ms: int, end_ms: int,
                     cache_dir: Path = None) -> list[dict]:
    """Fetch compressed aggregate trades from Binance REST API with GZIP cache.

    Returns list of dicts: {ts, p, q, m} (timestamp_ms, price, quantity, is_buyer_maker).
    """
    cache = cache_dir or CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    cache_file = cache / f"{symbol.replace('/', '_')}_{start_ms}_{end_ms}.json.gz"

    if cache_file.exists():
        console.print(f"  [dim]Loading {symbol} from cache...[/]")
        with gzip.open(cache_file, "rt") as f:
            return json.load(f)

    console.print(f"  Fetching {symbol} aggTrades...")

    # Try endpoints in order: data-api (no geo-block) → main api → mirrors
    API_ENDPOINTS = [
        "https://data-api.binance.vision/api/v3/aggTrades",
        "https://api.binance.com/api/v3/aggTrades",
        "https://api1.binance.com/api/v3/aggTrades",
    ]

    # Find a working endpoint
    base_url = None
    clean_sym = symbol.replace("/", "")
    for url in API_ENDPOINTS:
        try:
            test_resp = requests.get(url, params={"symbol": clean_sym, "limit": 1}, timeout=5)
            if test_resp.status_code == 200:
                base_url = url
                break
        except requests.RequestException:
            continue

    if base_url is None:
        raise RuntimeError(
            f"All Binance API endpoints are unreachable for {symbol}. "
            "This can happen when running from a US-based server. "
            "Try running the app locally instead."
        )

    all_trades = []
    cursor = start_ms
    batch = 0

    while cursor < end_ms:
        params = {
            "symbol": clean_sym,
            "startTime": cursor,
            "endTime": min(cursor + 3_600_000, end_ms),
            "limit": 1000,
        }

        while True:
            resp = requests.get(base_url, params=params)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10))
                console.print(f"  [yellow]Rate limited, waiting {wait}s...[/]")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break

        trades = resp.json()
        if not trades:
            cursor += 3_600_000
            continue

        for t in trades:
            all_trades.append({
                "ts": t["T"],
                "p": float(t["p"]),
                "q": float(t["q"]),
                "m": t["m"],
            })

        last_ts = trades[-1]["T"]

        if len(trades) < 1000:
            cursor = params["endTime"]
        else:
            params["startTime"] = last_ts + 1
            while True:
                resp = requests.get(base_url, params=params)
                if resp.status_code == 429:
                    time.sleep(int(resp.headers.get("Retry-After", 10)))
                    continue
                resp.raise_for_status()
                break
            extra = resp.json()
            for t in extra:
                all_trades.append({
                    "ts": t["T"],
                    "p": float(t["p"]),
                    "q": float(t["q"]),
                    "m": t["m"],
                })
            if extra:
                last_ts = extra[-1]["T"]
            cursor = params["endTime"]

        batch += 1
        if batch % 4 == 0:
            console.print(f"    ... {len(all_trades):,} trades so far "
                          f"({datetime.fromtimestamp(cursor/1000, tz=timezone.utc).strftime('%H:%M')} UTC)")

        time.sleep(0.15)

    all_trades = [t for t in all_trades if start_ms <= t["ts"] < end_ms]

    console.print(f"  [green]Got {len(all_trades):,} trades, caching...[/]")
    with gzip.open(cache_file, "wt") as f:
        json.dump(all_trades, f)

    return all_trades


def trades_to_time_series(trades: list[dict], bin_ms: int = 1000
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert trade list to time-binned VWAP arrays.

    Returns (timestamps_s, prices, volumes) at bin_ms resolution.
    """
    if not trades:
        return np.array([]), np.array([]), np.array([])

    min_ts = trades[0]["ts"]
    max_ts = trades[-1]["ts"]
    n_bins = (max_ts - min_ts) // bin_ms + 1

    price_sum = np.zeros(n_bins)
    vol_sum = np.zeros(n_bins)

    for t in trades:
        idx = (t["ts"] - min_ts) // bin_ms
        if 0 <= idx < n_bins:
            price_sum[idx] += t["p"] * t["q"]
            vol_sum[idx] += t["q"]

    prices = np.zeros(n_bins)
    for i in range(n_bins):
        if vol_sum[i] > 0:
            prices[i] = price_sum[i] / vol_sum[i]
        elif i > 0:
            prices[i] = prices[i - 1]

    first_valid = np.argmax(prices > 0)
    if first_valid > 0:
        prices[:first_valid] = prices[first_valid]

    timestamps = np.array([min_ts / 1000 + i * bin_ms / 1000 for i in range(n_bins)])

    return timestamps, prices, vol_sum


def load_aligned_pair(leader_symbol: str, follower_symbol: str,
                      start_date: datetime, days: int, bin_ms: int = 1000,
                      cache_dir: Path = None
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray]:
    """Load and align leader + follower time series.

    Fetches per-day data, bins to VWAP, and aligns to common timestamps.

    Returns (ts, leader_prices, leader_vols, follower_prices, follower_vols).
    """
    all_leader_trades = []
    all_follower_trades = []

    for day_offset in range(days):
        day_start = start_date + timedelta(days=day_offset)
        day_end = day_start + timedelta(days=1)
        day_start_ms = int(day_start.timestamp() * 1000)
        day_end_ms = int(day_end.timestamp() * 1000)

        console.print(f"\n  [bold]{day_start.strftime('%Y-%m-%d')}:[/]")
        leader_day = fetch_agg_trades(leader_symbol, day_start_ms, day_end_ms, cache_dir)
        follower_day = fetch_agg_trades(follower_symbol, day_start_ms, day_end_ms, cache_dir)
        console.print(f"    {leader_symbol}: {len(leader_day):,} | "
                      f"{follower_symbol}: {len(follower_day):,} trades")

        all_leader_trades.extend(leader_day)
        all_follower_trades.extend(follower_day)

    console.print(f"\n  [bold green]Total: {leader_symbol} {len(all_leader_trades):,} | "
                  f"{follower_symbol} {len(all_follower_trades):,} trades[/]")

    # Bin to VWAP time series
    console.print(f"  Binning to {bin_ms}ms VWAP...")
    l_ts, l_prices, l_vols = trades_to_time_series(all_leader_trades, bin_ms)
    f_ts, f_prices, f_vols = trades_to_time_series(all_follower_trades, bin_ms)

    # Align to common timestamps
    common_start = max(l_ts[0], f_ts[0])
    common_end = min(l_ts[-1], f_ts[-1])

    l_mask = (l_ts >= common_start) & (l_ts <= common_end)
    f_mask = (f_ts >= common_start) & (f_ts <= common_end)

    n = min(l_mask.sum(), f_mask.sum())
    ts = l_ts[l_mask][:n]
    l_prices = l_prices[l_mask][:n]
    l_vols = l_vols[l_mask][:n]
    f_prices = f_prices[f_mask][:n]
    f_vols = f_vols[f_mask][:n]

    console.print(f"  Aligned: {n:,} seconds ({n/3600:.1f} hours / {n/86400:.1f} days)\n")

    return ts, l_prices, l_vols, f_prices, f_vols
