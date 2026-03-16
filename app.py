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
    fee_label = st.selectbox("Fee Profile", list(fee_options.keys()), index=1)
    fee_key = fee_options[fee_label]
    fee_profile = FEE_MAP[fee_key]

    skip_optuna = st.checkbox("Skip Optuna optimization", value=False,
                              help="Optuna finds the best strategy parameters for this coin. "
                                   "Takes 2-5 min. Without it, fallback defaults are used.")

    with st.expander("Fallback / Manual Parameters", expanded=False):
        st.caption("These are only used when Optuna is skipped. "
                   "When Optuna runs, it finds optimal values automatically.")
        tp_pct = st.number_input("Take-Profit %", min_value=0.01, max_value=5.0, value=0.15, step=0.05, format="%.2f")
        sl_pct = st.number_input("Stop-Loss %", min_value=0.05, max_value=10.0, value=0.50, step=0.10, format="%.2f")
        btc_threshold = st.number_input("BTC Threshold %", min_value=0.05, max_value=3.0, value=0.50, step=0.10, format="%.2f")
        btc_window = st.number_input("BTC Window (seconds)", min_value=5, max_value=900, value=300, step=30)
        max_hold = st.number_input("Max Hold (seconds)", min_value=30, max_value=3600, value=600, step=60)
        cooldown = st.number_input("Cooldown (seconds)", min_value=10, max_value=600, value=60, step=10)
        vol_ratio = st.number_input("Volume Ratio Filter", min_value=1.0, max_value=10.0, value=1.0, step=0.5, format="%.1f",
                                    help="1.0 = disabled. Higher = only trade on volume spikes.")

    leverage_levels = st.multiselect("Leverage Levels", [1, 2, 3, 5, 10, 20], default=[1, 3, 5, 10])
    capital = st.number_input("Initial Capital (EUR)", min_value=100, max_value=100000, value=1000, step=100)

    st.divider()
    run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)


# ── Cached data loader ────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def _load_pair_cached(leader: str, follower: str, start_iso: str, days: int):
    """Cache tick data so repeat runs with same coin/period are instant.

    Uses start_iso (string) instead of datetime for cache-key hashability.
    TTL=3600s (1 hour) keeps data fresh without re-fetching every click.
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

    # Fallback params (used when Optuna is skipped)
    fallback_params = StrategyParams(
        btc_window_s=btc_window,
        btc_threshold_pct=btc_threshold,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        max_hold_s=max_hold,
        cooldown_s=cooldown,
        min_volume_ratio=vol_ratio,
    )

    end_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)

    n_steps = 8 if not skip_optuna else 7

    with st.status(f"Analyzing {coin}/BTC over {days} days...", expanded=True):
        # Step 1: Load data
        st.write(f"Step 1/{n_steps}: Loading tick data...")
        _buf.truncate(0)
        _buf.seek(0)
        ts, btc_p, btc_v, f_p, f_v = _load_pair_cached(
            leader_symbol.replace("USDT", "/USDT"),
            follower_symbol.replace("USDT", "/USDT"),
            start_date.isoformat(), days,
        )
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

        # Step 8: Risk Monte Carlo
        st.write(f"Step {step}/{n_steps}: Risk Monte Carlo (10k permutations)...")
        risk_result = risk_mod.risk_profile_monte_carlo(
            strategy_trades,
            leverage_levels=sorted(leverage_levels),
            n_permutations=10000,
            initial_capital=capital,
            fee_rt_pct=fee_profile.round_trip_pct,
        )

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
st.caption(f"{res['meta']['start_date']} to {res['meta']['end_date']} ({res['meta']['days']} days) | "
           f"Fee: {res['meta']['fee_profile']} ({res['meta']['fee_rt_pct']:.2f}% r/t) | "
           f"Params: {source_label}")

# ── Tabs ──────────────────────────────────────────────────────────
tab_overview, tab_structure, tab_strategy, tab_baseline, tab_regime, tab_risk, tab_optuna = st.tabs(
    ["Overview", "Market Structure", "Strategy", "Baseline", "Regime", "Risk", "Optuna"]
)

# ── Tab 1: Overview ───────────────────────────────────────────────
with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Win Rate", f"{strat['win_rate']:.1f}%")
    c2.metric("Trades", strat["total_trades"])
    c3.metric("Net Return", f"{strat['total_net_pct']:+.2f}%")
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
            df.style.applymap(
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
            "Win Rate %": round(rg.win_rate, 1) if rg.trades_count > 0 else "-",
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
    # Serialize results for download (exclude non-serializable objects)
    def _serialize(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _serialize(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [_serialize(i) for i in obj]
        if isinstance(obj, dict):
            return {str(k): _serialize(v) for k, v in obj.items()}
        return obj

    json_str = json.dumps(_serialize(res), indent=2, default=str)
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
