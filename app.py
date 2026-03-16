#!/usr/bin/env python3
"""Streamlit frontend for the BTC Leader Analyzer.

Run: streamlit run app.py
"""

import sys
import io
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Suppress Rich console output from backend modules
from rich.console import Console as RichConsole
_buf = io.StringIO()
_quiet_console = RichConsole(file=_buf, force_terminal=False)

import data as data_mod
import impulse as impulse_mod
import strategy as strategy_mod
import baseline as baseline_mod
import regime as regime_mod
import risk as risk_mod
import correlation as correlation_mod
import walkforward as wf_mod
import history as history_mod

# Redirect all module consoles to buffer
data_mod.console = _quiet_console

from config import (
    StrategyParams, FeeProfile, AnalysisConfig,
    FEE_MAP, BINANCE_SPOT_TAKER, BINANCE_FUTURES_TAKER,
    BINANCE_FUTURES_MAKER, BYBIT_FUTURES_MAKER,
)

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Leader Analyzer",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("BTC Leader Analyzer")
    st.caption("Test any coin's catch-up behavior after BTC impulses")

    st.header("Configuration")

    coin_options = ["XRP", "ETH", "SOL", "DOGE", "ADA", "AVAX", "LINK", "DOT"]
    coin = st.selectbox("Follower Coin", coin_options, index=0)
    custom_coin = st.text_input("Or enter custom symbol", "", placeholder="e.g. MATIC")
    if custom_coin.strip():
        coin = custom_coin.strip().upper()

    days = st.slider("Days to analyze", 1, 30, 14,
                      help="14+ days recommended for regime diversity (bull/bear/flat).")

    fee_options = {
        "Binance Spot Taker (0.10%/leg)": "spot",
        "Binance Futures Taker (0.04%/leg)": "futures-taker",
        "Binance Futures Maker (0.02%/leg)": "futures-maker",
        "Bybit Futures Maker (0.01%/leg)": "bybit-maker",
    }
    fee_label = st.selectbox("Fee Profile", list(fee_options.keys()), index=3)
    fee_key = fee_options[fee_label]
    fee_profile = FEE_MAP[fee_key]

    exec_delay_s = st.number_input("Execution Latency (s)", min_value=0, max_value=5, value=1, step=1,
                                    help="Seconds of API latency before trade entry (data resolution is 1s).")

    slippage_bps = st.number_input("Slippage (bps/leg)", min_value=0, max_value=20, value=2, step=1,
                                    help="Market impact per leg in basis points. "
                                         "2 bps is conservative for liquid futures. "
                                         "Set to 0 for ideal (no slippage).")

    skip_optuna = st.checkbox("Skip Optuna optimization", value=False,
                              help="Optuna finds the best strategy parameters for this coin. "
                                   "Takes 2-5 min. Without it, fallback defaults are used.")

    run_walkforward = st.checkbox("Walk-forward validation", value=False,
                                   help="Split data into train/test folds. Optimizes on train, tests on unseen data. "
                                        "Proves whether the edge survives out-of-sample. Requires Optuna enabled. "
                                        "Adds 5-15 min depending on data size.")

    with st.expander("Fallback / Manual Parameters", expanded=False):
        st.caption("Used when Optuna is skipped. When Optuna runs, it optimizes "
                   "btc_window, btc_threshold, TP, and SL automatically. "
                   "Max hold, cooldown, and volume ratio are fixed based on research.")
        tp_pct = st.number_input("Take-Profit %", min_value=0.10, max_value=2.0, value=0.20, step=0.05, format="%.2f",
                                 help="Min 0.15% to clear fees+slippage on altcoins.")
        sl_pct = st.number_input("Stop-Loss %", min_value=0.10, max_value=3.0, value=0.50, step=0.10, format="%.2f",
                                 help="Min 0.20% to avoid noise stops from bid-ask bounce.")
        btc_threshold = st.number_input("BTC Threshold %", min_value=0.05, max_value=2.0, value=0.30, step=0.05, format="%.2f",
                                        help="Optuna searches 0.10-1.0%.")
        btc_window = st.number_input("BTC Window (seconds)", min_value=5, max_value=300, value=60, step=15,
                                     help="Optuna searches 5-120s. Catch-up is typically 10-120s.")
        st.divider()
        st.caption("**Fixed parameters** (not optimized — reduces overfitting)")
        max_hold = st.number_input("Max Hold (seconds)", min_value=60, max_value=600, value=180, step=30,
                                   help="Fixed at 180s. Catch-up completes in 60-120s per literature.")
        cooldown = st.number_input("Cooldown (seconds)", min_value=15, max_value=120, value=45, step=15,
                                   help="Fixed at 45s. Prevents correlated signals.")
        vol_ratio = st.number_input("Volume Ratio Filter", min_value=1.0, max_value=5.0, value=2.0, step=0.5, format="%.1f",
                                    help="Fixed at 2.0x. Filters noise trades without being too restrictive.")

    leverage_levels = st.multiselect("Leverage Levels", [1, 2, 3, 5, 10, 20], default=[1, 3, 5, 10])
    capital = st.number_input("Initial Capital (EUR)", min_value=100, max_value=100000, value=1000, step=100)

    st.divider()
    run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

    # Cache info
    cache_info = data_mod.get_cache_summary()
    if cache_info["files"] > 0:
        with st.expander("Cache Info"):
            st.caption(f"{cache_info['files']} cached day-files ({cache_info['size_mb']:.1f} MB)")
            if cache_info["symbols"]:
                st.caption(f"Symbols: {', '.join(cache_info['symbols'])}")


# ── Cached data loader ────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_pair_cached(leader: str, follower: str, start_iso: str, days: int):
    """Cache tick data so repeat runs with same coin/period are instant.

    Uses start_iso (string) instead of datetime for cache-key hashability.
    Disk cache (.tick_cache/) is the real persistence layer; this avoids re-reading.
    """
    start_date = datetime.fromisoformat(start_iso)
    ts, l_p, l_v, f_p, f_v = data_mod.load_aligned_pair(
        leader, follower, start_date, days,
    )
    return ts, l_p, l_v, f_p, f_v


# ── Helper: build trades DataFrame ────────────────────────────────
def trades_to_df(trades):
    rows = []
    for t in trades:
        rows.append({
            "Time (UTC)": datetime.fromtimestamp(t.entry_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Dir": t.direction,
            "BTC Impulse %": round(t.btc_impulse_pct, 3),
            "Return %": round(t.follower_return_pct, 3),
            "Exit": t.exit_reason,
            "Gross %": round(t.gross_pnl_pct, 3),
            "Net %": round(t.net_pnl_pct, 3),
            "Result": "WIN" if t.net_pnl_pct > 0 else "LOSS",
        })
    return pd.DataFrame(rows)


def compute_equity_curve(trades):
    equity = [1.0]
    for t in trades:
        equity.append(equity[-1] * (1 + t.net_pnl_pct / 100))
    return equity


# ── Run Pipeline ──────────────────────────────────────────────────
if run_clicked:
    follower_symbol = f"{coin}USDT"
    leader_symbol = "BTCUSDT"

    exec_delay_s = max(0, int(exec_delay_s))

    slippage_bps_val = float(slippage_bps)

    # Fallback params (used when Optuna is skipped)
    fallback_params = StrategyParams(
        btc_window_s=btc_window,
        btc_threshold_pct=btc_threshold,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        max_hold_s=max_hold,
        cooldown_s=cooldown,
        min_volume_ratio=vol_ratio,
        execution_delay_s=exec_delay_s,
        slippage_bps=slippage_bps_val,
    )

    end_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)

    n_steps = 7  # base steps (data, optuna/skip, impulse, market, strategy, baseline, regime, risk... but counted as 7 below)
    if not skip_optuna:
        n_steps = 8
    if not skip_optuna:
        n_steps += 1  # OOS test
    if run_walkforward and not skip_optuna:
        n_steps += 1  # walk-forward

    with st.status(f"Analyzing {coin}/BTC over {days} days...", expanded=True):
        # Step 1: Load data
        st.write(f"Step 1/{n_steps}: Loading tick data...")
        _buf.truncate(0)
        _buf.seek(0)
        data_mod.clear_fetch_status()
        ts, btc_p, btc_v, f_p, f_v = _load_pair_cached(
            leader_symbol.replace("USDT", "/USDT"),
            follower_symbol.replace("USDT", "/USDT"),
            start_date.isoformat(), days,
        )
        # Show cache status
        fetch_st = data_mod.get_fetch_status()
        cached_n = sum(1 for v in fetch_st.values() if v == "cached")
        dl_n = sum(1 for v in fetch_st.values() if v == "downloaded")
        if fetch_st:
            if dl_n == 0:
                st.write(f"  All {cached_n} day-segments loaded from cache")
            elif cached_n == 0:
                st.write(f"  Downloaded {dl_n} day-segments from Binance")
            else:
                st.write(f"  {cached_n} from cache, {dl_n} freshly downloaded")
        st.write(f"  Aligned: {len(ts):,} seconds ({len(ts)/3600:.1f} hours)")

        # Step 2: Optuna optimization (finds optimal params)
        optuna_result = None
        params_source = "manual"
        if not skip_optuna:
            st.write(f"Step 2/{n_steps}: Optuna parameter optimization (300 trials)...")
            from optimize import optimize_parameters
            best_params, opt_summary = optimize_parameters(
                ts, btc_p, btc_v, ts, f_p, f_v,
                fee_profile, leverage=1.0, n_trials=300,
                slippage_bps=slippage_bps_val,
            )
            optuna_result = {"best_params": best_params, "summary": opt_summary}

            # Build strategy params from Optuna results
            params = StrategyParams(
                btc_window_s=best_params["btc_window_s"],
                btc_threshold_pct=best_params["btc_threshold_pct"],
                tp_pct=best_params["tp_pct"],
                sl_pct=best_params["sl_pct"],
                max_hold_s=best_params["max_hold_s"],
                cooldown_s=best_params["cooldown_s"],
                min_volume_ratio=best_params["min_volume_ratio"],
                execution_delay_s=exec_delay_s,
                slippage_bps=slippage_bps_val,
            )
            params_source = "optuna"
            st.write(f"  Best score: {opt_summary['best_score']:.4f}")
            st.write(f"  Optimal: TP={params.tp_pct:.3f}% SL={params.sl_pct:.3f}% "
                     f"BTC>{params.btc_threshold_pct:.3f}% window={params.btc_window_s:.0f}s")
        else:
            params = fallback_params
            st.write("Step 2: Optuna skipped — using manual parameters")

        step = 3

        # Step 3: Impulse detection
        st.write(f"Step {step}/{n_steps}: Detecting BTC impulse events...")
        events = impulse_mod.detect_impulse_events(
            ts, btc_p, btc_v, ts, f_p, f_v,
        )
        impulse_summary = impulse_mod.summarize_impulse_events(events)
        st.write(f"  Found {len(events):,} impulse events")
        step += 1

        # Step 4: Market structure analysis
        st.write(f"Step {step}/{n_steps}: Computing market structure metrics...")
        correlation_result = correlation_mod.compute_correlation_metrics(
            btc_p, f_p, btc_v, f_v,
        )
        rolling_corr_ts, rolling_corr = correlation_mod.compute_rolling_correlation(
            btc_p, f_p, ts, window_s=300,
        )
        ccf_lags, ccf_values = correlation_mod.compute_cross_correlation_function(
            btc_p, f_p, max_lag_s=30,
        )
        catchup_result = correlation_mod.compute_catchup_time(
            btc_p, f_p, window_s=300, threshold_pct=0.3, max_scan_s=600,
        )
        st.write(f"  Pearson={correlation_result['pearson_returns']:.3f}, "
                 f"Beta={correlation_result['beta']:.2f}, "
                 f"Median catch-up={catchup_result['median_catchup_s']:.0f}s")
        step += 1

        # Step 5: Strategy simulation (with optimal or fallback params)
        st.write(f"Step {step}/{n_steps}: Simulating TP/SL strategy ({params_source} params)...")
        strat_result = strategy_mod.simulate_tpsl_strategy(
            ts, btc_p, btc_v, ts, f_p, f_v,
            params, fee_profile, leverage=1.0,
        )
        strategy_trades = strat_result["trades"]
        st.write(f"  {strat_result['total_trades']} trades, {strat_result['win_rate']:.1f}% win rate")
        step += 1

        # Step 6: Baseline comparison
        st.write(f"Step {step}/{n_steps}: Running random baseline (500 trials)...")
        baseline_result = baseline_mod.random_baseline_comparison(
            f_p, strategy_trades, params, fee_profile,
            leverage=1.0, n_trials=500,
        )
        st.write(f"  BTC trigger beats random: {baseline_result['percentile_rank_wr']:.0f}%")
        step += 1

        # Step 7: Regime classification
        st.write(f"Step {step}/{n_steps}: Classifying market regimes...")
        regimes = regime_mod.classify_daily_regimes(ts, btc_p, ts, f_p, strategy_trades)
        regime_sum = regime_mod.regime_summary(regimes)
        step += 1

        # Risk Monte Carlo
        st.write(f"Step {step}/{n_steps}: Risk Monte Carlo (10k permutations)...")
        risk_result = risk_mod.risk_profile_monte_carlo(
            strategy_trades,
            leverage_levels=sorted(leverage_levels),
            n_permutations=10000,
            initial_capital=capital,
            fee_rt_pct=fee_profile.round_trip_pct,
        )
        step += 1

        # Out-of-sample test (run Optuna params on preceding period)
        oos_result_data = None
        if not skip_optuna:
            oos_days = min(days, 14)  # match analysis length or cap at 14
            oos_start = start_date - timedelta(days=oos_days)
            st.write(f"Step {step}/{n_steps}: Out-of-sample test ({oos_days}d before analysis window)...")
            try:
                ts_oos, btc_oos, btcv_oos, f_oos, fv_oos = _load_pair_cached(
                    leader_symbol.replace("USDT", "/USDT"),
                    follower_symbol.replace("USDT", "/USDT"),
                    oos_start.isoformat(), oos_days,
                )
                oos_strat = strategy_mod.simulate_tpsl_strategy(
                    ts_oos, btc_oos, btcv_oos, ts_oos, f_oos, fv_oos,
                    params, fee_profile, leverage=1.0,
                )
                oos_result_data = {
                    "period": f"{oos_start.strftime('%Y-%m-%d')} to {start_date.strftime('%Y-%m-%d')}",
                    "days": oos_days,
                    "win_rate": oos_strat["win_rate"],
                    "avg_net": oos_strat["avg_net_pct"],
                    "total_return": oos_strat["total_net_pct"],
                    "n_trades": oos_strat["total_trades"],
                    "max_dd": oos_strat["max_drawdown_pct"],
                }
                st.write(f"  OOS: {oos_strat['total_trades']} trades, "
                         f"{oos_strat['win_rate']:.1f}% WR, "
                         f"{oos_strat['total_net_pct']:+.2f}% return")
            except Exception as e:
                st.write(f"  OOS data not available: {e}")
            step += 1

        # Walk-forward validation
        wf_result_data = None
        if run_walkforward and not skip_optuna:
            train_d = max(7, days - 4)  # leave at least 4 days for test
            test_d = min(4, days - 7)
            if test_d < 2:
                test_d = 2
                train_d = days - test_d
            st.write(f"Step {step}/{n_steps}: Walk-forward validation "
                     f"(train={train_d}d, test={test_d}d)...")
            wf_result_data = wf_mod.run_walk_forward(
                leader_symbol.replace("USDT", "/USDT"),
                follower_symbol.replace("USDT", "/USDT"),
                start_date, days,
                train_days=train_d, test_days=test_d,
                fee_profile=fee_profile,
                slippage_bps=slippage_bps_val,
                exec_delay_s=exec_delay_s,
                n_trials=150,
                load_fn=_load_pair_cached,
            )
            st.write(f"  {wf_result_data['n_folds']} folds | "
                     f"OOS avg net: {wf_result_data['aggregate_oos']['avg_net']:+.4f}% | "
                     f"Degradation: {wf_result_data['degradation_pct']:.0f}%")

    # Store in session state
    st.session_state.results = {
        "meta": {
            "coin": coin,
            "follower": follower_symbol,
            "leader": leader_symbol,
            "days": days,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "data_points": len(ts),
            "strategy_params": params.__dict__,
            "params_source": params_source,
            "fee_profile": fee_profile.name,
            "fee_rt_pct": fee_profile.round_trip_pct,
            "slippage_bps": slippage_bps_val,
        },
        "impulse_summary": impulse_summary,
        "strategy": {k: v for k, v in strat_result.items() if k != "trades"},
        "baseline": {k: v for k, v in baseline_result.items() if not k.endswith("_distribution")},
        "correlation": correlation_result,
        "catchup": catchup_result,
        "regime_summary": regime_sum,
        "risk": risk_result,
    }
    if optuna_result:
        st.session_state.results["optuna"] = optuna_result["summary"]
    if oos_result_data:
        st.session_state.results["oos"] = oos_result_data
    if wf_result_data:
        st.session_state.results["walkforward"] = wf_result_data

    st.session_state.trades = strategy_trades
    st.session_state.regimes = regimes
    st.session_state.strat_result = strat_result
    st.session_state.baseline_result = baseline_result
    st.session_state.coin = coin
    # Chart arrays for Market Structure tab (too large for JSON results dict)
    st.session_state.rolling_corr_ts = rolling_corr_ts
    st.session_state.rolling_corr = rolling_corr
    st.session_state.ccf_lags = ccf_lags
    st.session_state.ccf_values = ccf_values
    # Store for cross-coin comparison
    st.session_state.start_iso = start_date.isoformat()
    st.session_state.analysis_days = days

    # Persist key metrics for cross-run comparison
    peak_lag_s = int(ccf_lags[int(np.argmax(ccf_values))])
    correlation_mod.save_market_structure(
        coin, days,
        f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        {
            "pearson": correlation_result["pearson_returns"],
            "spearman": correlation_result["spearman_returns"],
            "beta": correlation_result["beta"],
            "beta_up": correlation_result["beta_up"],
            "beta_down": correlation_result["beta_down"],
            "relative_volatility": correlation_result["relative_volatility"],
            "volume_correlation": correlation_result["volume_correlation"],
            "peak_lag_s": peak_lag_s,
            "impulse_median_lag_ms": impulse_summary["median_lag_ms"],
            "median_catchup_s": catchup_result["median_catchup_s"],
            "pct_no_catchup": catchup_result["pct_no_catchup"],
        },
    )

    # Auto-save full analysis to history for cross-run comparison
    history_mod.save_analysis_run(st.session_state.results)


# ── Display Results ───────────────────────────────────────────────
if "results" not in st.session_state:
    st.info("Configure parameters in the sidebar and click **Run Analysis** to start.")
    st.stop()

res = st.session_state.results
coin = st.session_state.coin
strat = res["strategy"]
baseline = res["baseline"]
trades = st.session_state.trades
regimes = st.session_state.regimes

st.header(f"{coin}/BTC Catch-Up Trade Analysis")
params_source = res['meta'].get('params_source', 'manual')
source_label = "Optuna-optimized" if params_source == "optuna" else "Manual fallback"
slip_label = f" | Slippage: {res['meta'].get('slippage_bps', 0):.0f} bps/leg" if res['meta'].get('slippage_bps', 0) > 0 else ""
st.caption(f"{res['meta']['start_date']} to {res['meta']['end_date']} ({res['meta']['days']} days) | "
           f"Fee: {res['meta']['fee_profile']} ({res['meta']['fee_rt_pct']:.2f}% r/t){slip_label} | "
           f"Params: {source_label}")

# ── Tabs ──────────────────────────────────────────────────────────
tab_overview, tab_structure, tab_strategy, tab_baseline, tab_regime, tab_risk, tab_optuna, tab_history = st.tabs(
    ["Overview", "Market Structure", "Strategy", "Baseline", "Regime", "Risk", "Optuna", "History"]
)

# ── Tab 1: Overview ───────────────────────────────────────────────
with tab_overview:
    is_label = " (In-Sample)" if params_source == "optuna" else ""
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"Win Rate{is_label}", f"{strat['win_rate']:.1f}%")
    c2.metric("Trades", strat["total_trades"])
    c3.metric(f"Net Return{is_label}", f"{strat['total_net_pct']:+.2f}%")
    c4.metric("Max Drawdown", f"{strat['max_drawdown_pct']:.1f}%")
    c5.metric("Beats Random", f"{baseline['percentile_rank_wr']:.0f}%")

    # ── Sample quality warnings ──────────────────────────────────
    regime_sum = res["regime_summary"]
    regimes_present = [r for r in ["BULL", "BEAR", "FLAT"] if regime_sum.get(r, {}).get("days", 0) > 0]
    regimes_missing = [r for r in ["BULL", "BEAR", "FLAT"] if r not in regimes_present]

    if regimes_missing:
        st.warning(
            f"**Regime coverage gap:** No {', '.join(regimes_missing).lower()} days in this sample "
            f"(only {', '.join(regimes_present).lower()}). "
            f"Results may not generalize to all market conditions. "
            f"Try increasing the analysis period to 14-30 days."
        )
    elif strat["total_trades"] < 50:
        st.warning(
            f"**Low trade count:** Only {strat['total_trades']} trades found. "
            f"Results may not be statistically robust. "
            f"Try increasing the analysis period."
        )
    else:
        regime_days = {r: regime_sum.get(r, {}).get("days", 0) for r in ["BULL", "BEAR", "FLAT"]}
        st.success(
            f"**Good sample quality:** {sum(regime_days.values())} days covering "
            f"{regime_days['BULL']} bull, {regime_days['BEAR']} bear, {regime_days['FLAT']} flat "
            f"with {strat['total_trades']} trades."
        )

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Strategy Breakdown")
        st.write(f"- **TP exits:** {strat['tp_rate']:.0f}%")
        st.write(f"- **SL exits:** {strat['sl_rate']:.0f}%")
        st.write(f"- **Timeout exits:** {strat['timeout_rate']:.0f}%")
        st.write(f"- **Avg gross/trade:** {strat['avg_gross_pct']:+.4f}%")
        st.write(f"- **Avg net/trade:** {strat['avg_net_pct']:+.4f}%")

    with col2:
        corr_ov = res["correlation"]
        st.subheader("Market Structure")
        st.write(f"- **Pearson r:** {corr_ov['pearson_returns']:.3f}")
        st.write(f"- **Beta:** {corr_ov['beta']:.2f}x")
        st.write(f"- **Rel. volatility:** {corr_ov['relative_volatility']:.2f}x")
        st.caption("See Market Structure tab for full analysis")

    with col3:
        st.subheader("Impulse Summary")
        imp = res["impulse_summary"]
        st.write(f"- **Total events detected:** {imp['total_events']:,}")
        st.write(f"- **Mean already followed:** {imp['mean_followed_pct']:.0f}%")
        binary = imp.get("binary_response_pattern", {})
        if binary:
            st.write(f"- **No response:** {binary.get('no_response_pct', 0):.0f}%")
            st.write(f"- **Full follow:** {binary.get('full_follow_pct', 0):.0f}%")
            st.write(f"- **Overshoot:** {binary.get('overshoot_pct', 0):.0f}%")

    # ── Validation results (OOS + Walk-Forward) ──────────────────
    oos_data = res.get("oos")
    wf_data = res.get("walkforward")

    if oos_data or wf_data:
        st.divider()
        st.subheader("Validation (Out-of-Sample)")
        st.caption("These metrics are tested on data Optuna never saw \u2014 the numbers you should trust.")

        if oos_data and wf_data:
            vc1, vc2 = st.columns(2)
        elif oos_data:
            vc1 = st.container()
        else:
            vc2 = st.container()

        if oos_data:
            with vc1:
                st.markdown("**Out-of-Sample Test**")
                oos_wr = oos_data.get("win_rate", 0)
                oos_avg = oos_data.get("avg_net", 0)
                oos_total = oos_data.get("total_return", 0)
                oos_trades = oos_data.get("n_trades", 0)

                oc1, oc2, oc3, oc4 = st.columns(4)
                oc1.metric("OOS Win Rate", f"{oos_wr:.1f}%",
                           delta=f"{oos_wr - strat['win_rate']:.1f}pp vs IS")
                oc2.metric("OOS Avg Net", f"{oos_avg:+.4f}%",
                           delta=f"{oos_avg - strat['avg_net_pct']:+.4f}% vs IS")
                oc3.metric("OOS Total Return", f"{oos_total:+.2f}%")
                oc4.metric("OOS Trades", oos_trades)

                # Verdict
                if oos_avg > 0 and oos_wr > 50:
                    st.success("Edge survives out-of-sample.")
                elif oos_avg > 0:
                    st.info("Positive OOS returns but low win rate \u2014 edge is fragile.")
                else:
                    st.error("OOS returns are negative \u2014 in-sample results are likely overfitted.")

        if wf_data:
            with vc2:
                st.markdown("**Walk-Forward Validation**")
                wf_agg = wf_data.get("aggregate_oos", {})
                wf_deg = wf_data.get("degradation_pct", 0)
                wf_folds = wf_data.get("n_folds", 0)
                wf_wr = wf_agg.get("win_rate", 0)
                wf_avg = wf_agg.get("avg_net", 0)

                wc1, wc2, wc3, wc4 = st.columns(4)
                wc1.metric("WF OOS Win Rate", f"{wf_wr:.1f}%")
                wc2.metric("WF OOS Avg Net", f"{wf_avg:+.4f}%")
                wc3.metric("Degradation", f"{wf_deg:.0f}%",
                           help="How much worse OOS is vs in-sample. Lower = more robust.")
                wc4.metric("Folds", wf_folds)

                # Verdict
                if wf_deg < 30 and wf_avg > 0:
                    st.success(f"Robust: only {wf_deg:.0f}% degradation across {wf_folds} folds.")
                elif wf_avg > 0:
                    st.warning(f"Moderate degradation ({wf_deg:.0f}%) \u2014 edge exists but weakens out-of-sample.")
                else:
                    st.error(f"Walk-forward OOS is negative \u2014 strategy does not generalize.")
    elif params_source == "optuna":
        st.divider()
        st.info("Enable **Walk-forward validation** in the sidebar for out-of-sample robustness testing.")

# ── Tab 2: Market Structure ───────────────────────────────────────
with tab_structure:
    corr = res["correlation"]

    # ── Section 1: Correlation metric cards ────────────────────────
    st.subheader("Correlation Analysis")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Pearson r", f"{corr['pearson_returns']:.3f}")
    mc2.metric("Spearman rho", f"{corr['spearman_returns']:.3f}")
    mc3.metric("R-squared", f"{corr['r_squared']:.3f}")
    mc4.metric("Beta", f"{corr['beta']:.2f}")
    mc5.metric("Rel. Volatility", f"{corr['relative_volatility']:.2f}x")

    # Interpretation
    p = corr["pearson_returns"]
    if p > 0.8:
        st.success(f"Strong positive correlation ({p:.3f}): {coin} closely tracks BTC price movements.")
    elif p > 0.5:
        st.info(f"Moderate correlation ({p:.3f}): {coin} generally follows BTC but with notable divergences.")
    elif p > 0.2:
        st.warning(f"Weak correlation ({p:.3f}): {coin} shows limited coupling to BTC at the 1-second level.")
    else:
        st.error(f"Very weak correlation ({p:.3f}): {coin} moves largely independently from BTC.")

    st.divider()

    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        st.subheader("Asymmetry")
        st.write(f"- **Beta (BTC up):** {corr['beta_up']:.3f}")
        st.write(f"- **Beta (BTC down):** {corr['beta_down']:.3f}")
        if corr["beta_up"] != 0:
            ratio = corr["beta_down"] / corr["beta_up"]
            if ratio > 1.1:
                st.write(f"- {coin} amplifies BTC drops more than rallies ({ratio:.2f}x)")
            elif ratio < 0.9:
                st.write(f"- {coin} amplifies BTC rallies more than drops ({1/ratio:.2f}x)")
            else:
                st.write("- Symmetric response to BTC moves")

    with ac2:
        st.subheader("Regression")
        st.write(f"- **Alpha (drift):** {corr['alpha'] * 86400 * 100:.4f}%/day")
        if corr["beta"] > 1.0:
            st.write(f"- {coin} amplifies BTC moves by {corr['beta']:.1f}x on average")
        elif corr["beta"] < 1.0:
            st.write(f"- {coin} dampens BTC moves (only {corr['beta']:.1f}x)")
        else:
            st.write(f"- {coin} moves 1:1 with BTC")

    with ac3:
        st.subheader("Volume")
        st.write(f"- **Volume correlation:** {corr['volume_correlation']:.3f}")
        if corr["volume_correlation"] > 0.5:
            st.write("- Strong volume coupling with BTC")
        elif corr["volume_correlation"] > 0.2:
            st.write("- Moderate volume coupling")
        else:
            st.write("- Weak volume coupling — independent activity patterns")

    # ── Section 2: Rolling correlation chart ───────────────────────
    st.divider()
    st.subheader("Rolling Correlation (5-min window)")

    if "rolling_corr_ts" in st.session_state:
        roll_ts = st.session_state.rolling_corr_ts
        roll_corr = st.session_state.rolling_corr

        if len(roll_corr) > 0:
            # Downsample to ~1 per minute for chart performance
            step_size = max(1, 60)
            roll_dates = pd.to_datetime(roll_ts[::step_size], unit="s", utc=True)

            roll_df = pd.DataFrame({
                "Time": roll_dates,
                "Correlation": roll_corr[::step_size],
            }).set_index("Time")

            st.line_chart(roll_df, y="Correlation", use_container_width=True)
            st.caption(
                f"Rolling 5-minute Pearson correlation of 1-second returns. "
                f"Mean: {np.mean(roll_corr):.3f}, "
                f"Min: {np.min(roll_corr):.3f}, "
                f"Max: {np.max(roll_corr):.3f}"
            )

    # ── Section 3: Lead-Lag cross-correlation ─────────────────────
    st.divider()
    st.subheader("Lead-Lag Structure")

    col_lag1, col_lag2 = st.columns([2, 1])

    with col_lag1:
        if "ccf_lags" in st.session_state:
            lags = st.session_state.ccf_lags
            ccf = st.session_state.ccf_values

            ccf_df = pd.DataFrame({
                "Lag (seconds)": lags,
                "Correlation": ccf,
            }).set_index("Lag (seconds)")
            st.bar_chart(ccf_df, y="Correlation", use_container_width=True)
            st.caption(
                "Cross-correlation function. Negative lag = follower leads BTC. "
                "Positive lag = BTC leads follower. Peak shows the dominant relationship."
            )

    with col_lag2:
        imp = res["impulse_summary"]
        catchup = res.get("catchup", {})
        if "ccf_lags" in st.session_state:
            ccf = st.session_state.ccf_values
            lags = st.session_state.ccf_lags
            peak_idx = int(np.argmax(ccf))
            peak_lag = int(lags[peak_idx])

            st.metric("CCF Peak Lag", f"{peak_lag}s")
            st.metric("Impulse Median Lag", f"{imp['median_lag_ms']:.0f}ms")
            if catchup.get("n_events", 0) > 0:
                st.metric("Median Catch-Up", f"{catchup['median_catchup_s']:.0f}s",
                          help="Time for follower to reach 50% of BTC's move")

            if peak_lag > 0:
                st.info(
                    f"BTC leads {coin} by ~{peak_lag}s on average. "
                    f"This supports the catch-up trading thesis."
                )
            elif peak_lag < 0:
                st.warning(
                    f"{coin} appears to lead BTC by ~{abs(peak_lag)}s. "
                    f"Unusual — may indicate reverse causality or noise."
                )
            else:
                st.write(f"No measurable lag between BTC and {coin} at 1-second resolution.")

            # Catch-up detail
            if catchup.get("n_events", 0) > 0:
                st.caption(
                    f"Catch-up: P25={catchup['p25_catchup_s']:.0f}s, "
                    f"P75={catchup['p75_catchup_s']:.0f}s | "
                    f"{catchup['pct_no_catchup']:.0f}% never caught up | "
                    f"{catchup['n_events']} events"
                )

    # ── Section 4: Cross-coin comparison ──────────────────────────
    st.divider()
    st.subheader("Cross-Coin Lag Comparison")

    compare_coins = ["ETH", "SOL", "DOGE", "ADA", "AVAX", "LINK"]
    compare_coins = [c for c in compare_coins if c != coin]

    run_comparison = st.checkbox(
        f"Run comparison across {len(compare_coins)} coins",
        value=False,
        help="Fetches data for other top coins to compare lag and correlation. "
             "Uses cached data if previously analyzed. Takes 1-5 min if uncached.",
    )

    if run_comparison:
        comparison_key = f"cross_coin_{coin}_{st.session_state.analysis_days}"
        if comparison_key not in st.session_state:
            with st.spinner(f"Loading and analyzing {len(compare_coins)} coins..."):
                comp_results = correlation_mod.compute_cross_coin_comparison(
                    compare_coins,
                    st.session_state.start_iso,
                    st.session_state.analysis_days,
                    _load_pair_cached,
                )
                # Add current coin
                comp_results.append({
                    "coin": coin,
                    "pearson": corr["pearson_returns"],
                    "beta": corr["beta"],
                    "median_lag_s": int(lags[int(np.argmax(ccf))]) if "ccf_lags" in st.session_state else 0,
                    "relative_vol": corr["relative_volatility"],
                    "status": "ok",
                })
                st.session_state[comparison_key] = comp_results

        comp_data = st.session_state[comparison_key]
        ok_data = [c for c in comp_data if c["status"] == "ok"]
        err_data = [c for c in comp_data if c["status"] != "ok"]

        if ok_data:
            comp_df = pd.DataFrame(ok_data).sort_values("median_lag_s", ascending=True)

            display_df = comp_df[["coin", "median_lag_s", "pearson", "beta", "relative_vol"]].copy()
            display_df.columns = ["Coin", "Peak Lag (s)", "Pearson r", "Beta", "Rel. Vol"]

            def _highlight_current(row):
                if row["Coin"] == coin:
                    return ["background-color: #2a4a2a"] * len(row)
                return [""] * len(row)

            st.dataframe(
                display_df.style.apply(_highlight_current, axis=1).format({
                    "Peak Lag (s)": "{:.0f}",
                    "Pearson r": "{:.3f}",
                    "Beta": "{:.2f}",
                    "Rel. Vol": "{:.2f}x",
                }),
                use_container_width=True,
            )

            sorted_coins = list(comp_df.sort_values("median_lag_s")["coin"])
            if coin in sorted_coins:
                rank = sorted_coins.index(coin) + 1
                st.write(
                    f"**{coin}** ranks #{rank} out of {len(sorted_coins)} coins by lag "
                    f"(lower = faster follower response to BTC)."
                )

        if err_data:
            for e in err_data:
                st.caption(f"Could not load {e['coin']}: {e['status']}")

    # ── Section 5: Previous analyses history ──────────────────────
    history = correlation_mod.load_market_structure_history()
    if len(history) > 1:
        st.divider()
        st.subheader("Previous Analyses")
        st.caption("Saved from all prior runs — compare market structure across coins and periods.")

        hist_rows = []
        for key, h in history.items():
            hist_rows.append({
                "Coin": h.get("coin", "?"),
                "Days": h.get("days", 0),
                "Period": h.get("date_range", ""),
                "Pearson r": h.get("pearson", 0),
                "Beta": h.get("beta", 0),
                "Rel. Vol": h.get("relative_volatility", 0),
                "Peak Lag (s)": h.get("peak_lag_s", 0),
                "Catch-Up (s)": h.get("median_catchup_s", 0),
                "No Catch-Up %": h.get("pct_no_catchup", 0),
            })

        hist_df = pd.DataFrame(hist_rows)

        def _highlight_active(row):
            active_key = f"{coin}_{res['meta']['days']}d"
            row_key = f"{row['Coin']}_{row['Days']}d"
            if row_key == active_key:
                return ["background-color: #2a4a2a"] * len(row)
            return [""] * len(row)

        st.dataframe(
            hist_df.style.apply(_highlight_active, axis=1).format({
                "Pearson r": "{:.3f}",
                "Beta": "{:.2f}",
                "Rel. Vol": "{:.2f}x",
                "Peak Lag (s)": "{:.0f}",
                "Catch-Up (s)": "{:.0f}",
                "No Catch-Up %": "{:.0f}%",
            }),
            use_container_width=True,
        )


# ── Tab 3: Strategy ───────────────────────────────────────────────
with tab_strategy:
    if trades:
        st.subheader("Equity Curve (1x leverage)")
        equity = compute_equity_curve(trades)
        eq_df = pd.DataFrame({
            "Trade #": range(len(equity)),
            "Equity": [e * 100 for e in equity],
        }).set_index("Trade #")
        st.line_chart(eq_df, y="Equity", use_container_width=True)

        st.subheader(f"All Trades ({len(trades)})")
        df = trades_to_df(trades)

        # Color the Result column
        st.dataframe(
            df.style.map(
                lambda v: "color: green" if v == "WIN" else "color: red" if v == "LOSS" else "",
                subset=["Result"]
            ),
            use_container_width=True,
            height=400,
        )

        # Exit reason distribution
        st.subheader("Exit Reasons")
        exit_counts = df["Exit"].value_counts()
        st.bar_chart(exit_counts)
    else:
        st.warning("No trades generated with current parameters.")

    # ── Out-of-Sample Validation ─────────────────────────────────
    if "oos" in res:
        st.divider()
        st.subheader("Out-of-Sample Validation")
        st.caption("Same Optuna parameters tested on data *before* the analysis window (unseen during optimization).")

        oos = res["oos"]
        oc1, oc2, oc3, oc4, oc5 = st.columns(5)
        oc1.metric("OOS Trades", oos["n_trades"])
        oc2.metric("OOS Win Rate", f"{oos['win_rate']:.1f}%")
        oc3.metric("OOS Avg Net", f"{oos['avg_net']:+.4f}%")
        oc4.metric("OOS Total Return", f"{oos['total_return']:+.2f}%")
        oc5.metric("OOS Max DD", f"{oos['max_dd']:.1f}%")

        # Side-by-side comparison
        comp_data = {
            "Metric": ["Win Rate", "Avg Net/Trade", "Total Return", "Trades", "Max DD"],
            "In-Sample": [
                f"{strat['win_rate']:.1f}%",
                f"{strat['avg_net_pct']:+.4f}%",
                f"{strat['total_net_pct']:+.2f}%",
                str(strat["total_trades"]),
                f"{strat['max_drawdown_pct']:.1f}%",
            ],
            f"Out-of-Sample ({oos['days']}d)": [
                f"{oos['win_rate']:.1f}%",
                f"{oos['avg_net']:+.4f}%",
                f"{oos['total_return']:+.2f}%",
                str(oos["n_trades"]),
                f"{oos['max_dd']:.1f}%",
            ],
        }
        st.table(pd.DataFrame(comp_data))

        # Edge retention assessment
        if strat["avg_net_pct"] > 0 and oos["avg_net"] > 0:
            retention = oos["avg_net"] / strat["avg_net_pct"] * 100
            if retention >= 50:
                st.success(f"Edge retained: OOS keeps {retention:.0f}% of in-sample edge. The signal appears robust.")
            elif retention >= 25:
                st.warning(f"Partial edge: OOS retains {retention:.0f}% of in-sample edge. Some overfitting likely.")
            else:
                st.error(f"Weak OOS: Only {retention:.0f}% of in-sample edge survives. Heavy overfitting suspected.")
        elif strat["avg_net_pct"] > 0 and oos["avg_net"] <= 0:
            st.error("Edge lost: OOS shows negative returns. In-sample results are likely overfitted.")
        elif oos["n_trades"] == 0:
            st.warning("No OOS trades — parameters may be too restrictive for different market conditions.")

    # ── Walk-Forward Validation ──────────────────────────────────
    if "walkforward" in res:
        st.divider()
        st.subheader("Walk-Forward Validation")
        st.caption("Rolling train/test: optimize on train window, test on unseen test window, roll forward.")

        wf = res["walkforward"]

        wc1, wc2, wc3, wc4 = st.columns(4)
        wc1.metric("Folds", wf["n_folds"])
        wc2.metric("OOS Win Rate", f"{wf['aggregate_oos']['win_rate']:.1f}%")
        wc3.metric("OOS Avg Net", f"{wf['aggregate_oos']['avg_net']:+.4f}%")
        wc4.metric("Degradation", f"{wf['degradation_pct']:.0f}%",
                    help="How much worse OOS is vs in-sample (lower = better)")

        # Fold detail table
        fold_rows = []
        for f in wf["folds"]:
            if "error" in f:
                fold_rows.append({
                    "Fold": f["fold"],
                    "Train": f["train_range"],
                    "Test": f["test_range"],
                    "IS WR": "—",
                    "OOS WR": f"Error: {f['error']}",
                    "IS Avg Net": "—",
                    "OOS Avg Net": "—",
                    "IS Trades": "—",
                    "OOS Trades": "—",
                })
            else:
                fold_rows.append({
                    "Fold": f["fold"],
                    "Train": f["train_range"],
                    "Test": f["test_range"],
                    "IS WR": f"{f['in_sample']['win_rate']:.1f}%",
                    "OOS WR": f"{f['out_of_sample']['win_rate']:.1f}%",
                    "IS Avg Net": f"{f['in_sample']['avg_net']:+.4f}%",
                    "OOS Avg Net": f"{f['out_of_sample']['avg_net']:+.4f}%",
                    "IS Trades": f['in_sample']['n_trades'],
                    "OOS Trades": f['out_of_sample']['n_trades'],
                })

        if fold_rows:
            st.dataframe(pd.DataFrame(fold_rows), use_container_width=True)

        # Assessment
        deg = wf["degradation_pct"]
        oos_avg = wf["aggregate_oos"]["avg_net"]
        if oos_avg > 0 and deg < 50:
            st.success(f"Walk-forward positive: OOS avg net {oos_avg:+.4f}% with {deg:.0f}% degradation. "
                       "Edge appears real and tradeable.")
        elif oos_avg > 0:
            st.warning(f"Walk-forward marginal: OOS avg net {oos_avg:+.4f}% but {deg:.0f}% degradation. "
                       "Edge exists but is weaker than in-sample suggests.")
        else:
            st.error(f"Walk-forward negative: OOS avg net {oos_avg:+.4f}%. "
                     "In-sample results are likely overfitted to the specific time period.")

# ── Tab 3: Baseline ───────────────────────────────────────────────
with tab_baseline:
    st.subheader("BTC Trigger vs Random Entries")

    bc1, bc2, bc3 = st.columns(3)
    bc1.metric("Strategy Win Rate", f"{baseline['strategy_win_rate']:.1f}%")
    bc2.metric("Random Mean WR", f"{baseline['random_mean_win_rate']:.1f}%")
    bc3.metric("Percentile Rank", f"{baseline['percentile_rank_wr']:.0f}%",
               help="How often strategy beats random (higher = better)")

    st.divider()

    comp_df = pd.DataFrame({
        "Metric": ["Win Rate", "Avg Net/Trade"],
        "BTC Trigger": [
            f"{baseline['strategy_win_rate']:.1f}%",
            f"{baseline['strategy_avg_net']:+.4f}%",
        ],
        "Random Mean": [
            f"{baseline['random_mean_win_rate']:.1f}% +/- {baseline['random_std_win_rate']:.1f}%",
            f"{baseline['random_mean_avg_net']:+.4f}% +/- {baseline['random_std_avg_net']:.4f}%",
        ],
    })
    st.table(comp_df)

    pctl = baseline["percentile_rank_wr"]
    if pctl >= 95:
        st.success(f"The BTC trigger is statistically significant — beats random {pctl:.0f}% of the time.")
    elif pctl >= 80:
        st.info(f"The BTC trigger shows edge — beats random {pctl:.0f}% of the time.")
    else:
        st.warning(f"The BTC trigger is weak — beats random only {pctl:.0f}% of the time.")

    # Distribution histogram
    full_baseline = st.session_state.baseline_result
    if "random_wr_distribution" in full_baseline:
        st.subheader("Win Rate Distribution (500 random trials)")
        wr_dist = full_baseline["random_wr_distribution"]
        hist_df = pd.DataFrame({"Random Win Rate %": wr_dist})
        st.bar_chart(hist_df["Random Win Rate %"].value_counts().sort_index())

# ── Tab 4: Regime ─────────────────────────────────────────────────
with tab_regime:
    st.subheader("Market Regime Analysis")

    regime_sum = res["regime_summary"]
    regime_rows = []
    for r_type in ["BULL", "BEAR", "FLAT"]:
        r = regime_sum.get(r_type, {})
        if r.get("days", 0) > 0:
            regime_rows.append({
                "Regime": r_type,
                "Days": r["days"],
                "Trades": r["total_trades"],
                "Avg Win Rate": f"{r.get('avg_win_rate', 0):.0f}%",
                f"Avg {coin} Drift": f"{r.get('avg_follower_drift', 0):+.2f}%",
                "Avg BTC Drift": f"{r.get('avg_btc_drift', 0):+.2f}%",
            })
    if regime_rows:
        st.table(pd.DataFrame(regime_rows))

    st.subheader("Per-Day Breakdown")
    day_rows = []
    for rg in regimes:
        day_rows.append({
            "Date": rg.date,
            "Regime": rg.regime,
            "BTC Return %": round(rg.btc_return_pct, 2),
            f"{coin} Return %": round(rg.follower_return_pct, 2),
            "Trades": rg.trades_count,
            "Win Rate %": round(rg.win_rate, 1) if rg.trades_count > 0 else 0.0,
        })
    if day_rows:
        st.dataframe(pd.DataFrame(day_rows), use_container_width=True)

# ── Tab 5: Risk ───────────────────────────────────────────────────
with tab_risk:
    st.subheader("Monte Carlo Risk Profile")

    risk_data = res["risk"]
    risk_rows = []
    for key in sorted(risk_data.keys(), key=lambda x: float(x)):
        r = risk_data[key]
        risk_rows.append({
            "Leverage": f"{r['leverage']}x",
            "Median Return %": round(r["median_return_pct"], 1),
            "Median Max DD %": round(r["median_max_dd_pct"], 1),
            "P95 Max DD %": round(r["p95_max_dd_pct"], 1),
            "Prob Net Loss %": round(r["prob_net_loss"], 0),
            f"Median Final (EUR {int(st.session_state.get('capital', 1000)):,})": f"EUR {r['median_final_capital']:,.0f}",
        })

    if risk_rows:
        st.table(pd.DataFrame(risk_rows))

    # Chart: return vs drawdown by leverage
    if len(risk_data) >= 2:
        st.subheader("Return vs Drawdown by Leverage")
        chart_data = []
        for key in sorted(risk_data.keys(), key=lambda x: float(x)):
            r = risk_data[key]
            chart_data.append({
                "Leverage": f"{r['leverage']}x",
                "Median Return %": r["median_return_pct"],
                "Median Max DD %": r["median_max_dd_pct"],
                "P95 Max DD %": r["p95_max_dd_pct"],
            })
        chart_df = pd.DataFrame(chart_data).set_index("Leverage")
        st.bar_chart(chart_df)

# ── Tab 6: Optuna ─────────────────────────────────────────────────
with tab_optuna:
    if "optuna" in res:
        opt = res["optuna"]
        st.success("Optuna found the optimal parameters below. All strategy results use these values.")

        st.subheader("Optimal Parameters (used for all results)")
        best = opt["best_params"]

        pc1, pc2 = st.columns(2)
        with pc1:
            st.write(f"- **BTC Window:** {best.get('btc_window_s', 0):.1f}s")
            st.write(f"- **BTC Threshold:** >{best.get('btc_threshold_pct', 0):.3f}%")
            st.write(f"- **Take-Profit:** {best.get('tp_pct', 0):.3f}%")
            st.write(f"- **Stop-Loss:** {best.get('sl_pct', 0):.3f}%")
        with pc2:
            st.write(f"- **Max Hold:** {best.get('max_hold_s', 0):.0f}s")
            st.write(f"- **Cooldown:** {best.get('cooldown_s', 0):.0f}s")
            st.write(f"- **Volume Ratio:** >{best.get('min_volume_ratio', 0):.1f}x")
            st.write(f"- **Best Score:** {opt.get('best_score', 0):.4f}")

        importance = opt.get("param_importance", {})
        if importance:
            st.subheader("Parameter Importance")
            imp_df = pd.DataFrame({
                "Parameter": list(importance.keys()),
                "Importance": list(importance.values()),
            }).sort_values("Importance", ascending=True)
            st.bar_chart(imp_df.set_index("Parameter"))
    else:
        st.warning("Optuna was skipped — results use manual fallback parameters. "
                   "Uncheck 'Skip Optuna' for best results.")

# ── Downloads ─────────────────────────────────────────────────────
st.divider()
dcol1, dcol2 = st.columns(2)

with dcol1:
    json_str = json.dumps(history_mod.serialize(res), indent=2, default=str)
    st.download_button(
        "Download Results (JSON)",
        json_str,
        file_name=f"{coin}_BTC_analysis.json",
        mime="application/json",
    )

with dcol2:
    try:
        from report import generate_pdf_report
        pdf_path = f"/tmp/{coin}_BTC_analysis.pdf"
        generate_pdf_report(f"{coin}USDT", res, pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download Report (PDF)",
                f.read(),
                file_name=f"{coin}_BTC_analysis.pdf",
                mime="application/pdf",
            )
    except Exception as e:
        st.caption(f"PDF generation unavailable: {e}")

# ── Tab 8: History ────────────────────────────────────────────────
with tab_history:
    st.subheader("Analysis History")
    st.caption("Results are auto-saved after each run. Compare coins and strategies across analyses.")

    history = history_mod.load_analysis_history()

    if not history:
        st.info("No saved analyses yet. Run an analysis to start building history.")
    else:
        # Filter by coin
        coins_in_hist = sorted(set(
            e.get("meta", {}).get("coin", "?") for e in history.values()
        ))
        filter_coins = st.multiselect(
            "Filter by coin", coins_in_hist, default=coins_in_hist,
            key="history_coin_filter",
        )

        comp_df = history_mod.get_comparison_dataframe(history, filter_coins=filter_coins)

        if comp_df.empty:
            st.info("No matching analyses for selected coins.")
        else:
            # Style: color-code win rate and total net
            def _color_val(val):
                if isinstance(val, (int, float)) and not pd.isna(val):
                    if val > 0:
                        return "color: green"
                    elif val < 0:
                        return "color: red"
                return ""

            display_df = comp_df.drop(columns=["Run ID"])
            styled = display_df.style.map(
                _color_val, subset=["Avg Net %", "Total Net %"]
            )
            st.dataframe(styled, use_container_width=True, height=400)

            st.caption(f"{len(comp_df)} analyses saved")

        # Manage history
        with st.expander("Manage History"):
            run_ids = list(history.keys())
            del_id = st.selectbox("Select run to delete", run_ids, key="del_run_id")
            col_del1, col_del2 = st.columns(2)
            with col_del1:
                if st.button("Delete Selected"):
                    history_mod.delete_analysis_run(del_id)
                    st.rerun()
            with col_del2:
                if st.button("Clear All History"):
                    for rid in list(history.keys()):
                        history_mod.delete_analysis_run(rid)
                    st.rerun()
