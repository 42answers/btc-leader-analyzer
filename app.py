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

    days = st.slider("Days to analyze", 1, 30, 7)

    fee_options = {
        "Binance Spot Taker (0.10%/leg)": "spot",
        "Binance Futures Taker (0.04%/leg)": "futures-taker",
        "Binance Futures Maker (0.02%/leg)": "futures-maker",
        "Bybit Futures Maker (0.01%/leg)": "bybit-maker",
    }
    fee_label = st.selectbox("Fee Profile", list(fee_options.keys()), index=1)
    fee_key = fee_options[fee_label]
    fee_profile = FEE_MAP[fee_key]

    with st.expander("Strategy Parameters", expanded=False):
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

    skip_optuna = st.checkbox("Skip Optuna optimization", value=True,
                              help="Optuna is slow (5-10 min). Skip for faster results.")

    st.divider()
    run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)


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

    params = StrategyParams(
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

    with st.status(f"Analyzing {coin}/BTC over {days} days...", expanded=True):
        # Step 1: Load data
        st.write("Step 1/6: Loading tick data...")
        _buf.truncate(0)
        _buf.seek(0)
        ts, btc_p, btc_v, f_p, f_v = data_mod.load_aligned_pair(
            leader_symbol.replace("USDT", "/USDT"),
            follower_symbol.replace("USDT", "/USDT"),
            start_date, days,
        )
        st.write(f"  Aligned: {len(ts):,} seconds ({len(ts)/3600:.1f} hours)")

        # Step 2: Impulse detection
        st.write("Step 2/6: Detecting BTC impulse events...")
        events = impulse_mod.detect_impulse_events(
            ts, btc_p, btc_v, ts, f_p, f_v,
        )
        impulse_summary = impulse_mod.summarize_impulse_events(events)
        st.write(f"  Found {len(events):,} impulse events")

        # Step 3: Strategy simulation
        st.write("Step 3/6: Simulating TP/SL strategy...")
        strat_result = strategy_mod.simulate_tpsl_strategy(
            ts, btc_p, btc_v, ts, f_p, f_v,
            params, fee_profile, leverage=1.0,
        )
        strategy_trades = strat_result["trades"]
        st.write(f"  {strat_result['total_trades']} trades, {strat_result['win_rate']:.1f}% win rate")

        # Step 4: Baseline comparison
        st.write("Step 4/6: Running random baseline (500 trials)...")
        baseline_result = baseline_mod.random_baseline_comparison(
            f_p, strategy_trades, params, fee_profile,
            leverage=1.0, n_trials=500,
        )
        st.write(f"  BTC trigger beats random: {baseline_result['percentile_rank_wr']:.0f}%")

        # Step 5: Regime classification
        st.write("Step 5/6: Classifying market regimes...")
        regimes = regime_mod.classify_daily_regimes(ts, btc_p, ts, f_p, strategy_trades)
        regime_sum = regime_mod.regime_summary(regimes)

        # Step 6: Risk Monte Carlo
        st.write("Step 6/6: Risk Monte Carlo (10k permutations)...")
        risk_result = risk_mod.risk_profile_monte_carlo(
            strategy_trades,
            leverage_levels=sorted(leverage_levels),
            n_permutations=10000,
            initial_capital=capital,
            fee_rt_pct=fee_profile.round_trip_pct,
        )

        # Step 7: Optuna (optional)
        optuna_result = None
        if not skip_optuna:
            st.write("Step 7: Optuna optimization (this may take several minutes)...")
            from optimize import optimize_parameters
            best_params, opt_summary = optimize_parameters(
                ts, btc_p, btc_v, ts, f_p, f_v,
                fee_profile, leverage=1.0, n_trials=300,
            )
            optuna_result = {"best_params": best_params, "summary": opt_summary}

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
            "fee_profile": fee_profile.name,
            "fee_rt_pct": fee_profile.round_trip_pct,
        },
        "impulse_summary": impulse_summary,
        "strategy": {k: v for k, v in strat_result.items() if k != "trades"},
        "baseline": {k: v for k, v in baseline_result.items() if not k.endswith("_distribution")},
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
st.caption(f"{res['meta']['start_date']} to {res['meta']['end_date']} ({res['meta']['days']} days) | "
           f"Fee: {res['meta']['fee_profile']} ({res['meta']['fee_rt_pct']:.2f}% r/t)")

# ── Tabs ──────────────────────────────────────────────────────────
tab_overview, tab_strategy, tab_baseline, tab_regime, tab_risk, tab_optuna = st.tabs(
    ["Overview", "Strategy", "Baseline", "Regime", "Risk", "Optuna"]
)

# ── Tab 1: Overview ───────────────────────────────────────────────
with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Win Rate", f"{strat['win_rate']:.1f}%")
    c2.metric("Trades", strat["total_trades"])
    c3.metric("Net Return", f"{strat['total_net_pct']:+.2f}%")
    c4.metric("Max Drawdown", f"{strat['max_drawdown_pct']:.1f}%")
    c5.metric("Beats Random", f"{baseline['percentile_rank_wr']:.0f}%")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Strategy Breakdown")
        st.write(f"- **TP exits:** {strat['tp_rate']:.0f}%")
        st.write(f"- **SL exits:** {strat['sl_rate']:.0f}%")
        st.write(f"- **Timeout exits:** {strat['timeout_rate']:.0f}%")
        st.write(f"- **Avg gross/trade:** {strat['avg_gross_pct']:+.4f}%")
        st.write(f"- **Avg net/trade:** {strat['avg_net_pct']:+.4f}%")

    with col2:
        st.subheader("Impulse Summary")
        imp = res["impulse_summary"]
        st.write(f"- **Total events detected:** {imp['total_events']:,}")
        st.write(f"- **Median lag:** {imp['median_lag_ms']:.0f}ms")
        st.write(f"- **Mean already followed:** {imp['mean_followed_pct']:.0f}%")
        binary = imp.get("binary_response_pattern", {})
        if binary:
            st.write(f"- **No response:** {binary.get('no_response_pct', 0):.0f}%")
            st.write(f"- **Full follow:** {binary.get('full_follow_pct', 0):.0f}%")
            st.write(f"- **Overshoot:** {binary.get('overshoot_pct', 0):.0f}%")

# ── Tab 2: Strategy ───────────────────────────────────────────────
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
        st.subheader("Optuna Best Parameters")
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
        st.info("Optuna was skipped. Uncheck 'Skip Optuna' in the sidebar to run optimization.")

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
