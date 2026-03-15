#!/usr/bin/env python3
"""BTC-Leader Catch-Up Trade Analyzer — test any follower coin.

Usage:
    python3 analyze.py ETH                           # ETH/BTC, 7 days
    python3 analyze.py SOL --days 30                 # SOL/BTC, 30 days
    python3 analyze.py DOGE --no-optuna              # Skip optimization
    python3 analyze.py ADA --stage impulse           # Only impulse detection
    python3 analyze.py ETH --leverage 3 5 10         # Specific leverage levels
    python3 analyze.py ETH --fee bybit-maker         # Fee profile
    python3 analyze.py ETH --tp 0.20 --sl 0.75       # Custom TP/SL
    python3 analyze.py XRP --days 7 --no-pdf         # Skip PDF report
"""

import argparse
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from config import (
    AnalysisConfig, StrategyParams, FeeProfile,
    FEE_MAP, BINANCE_FUTURES_TAKER,
)
from data import load_aligned_pair

console = Console()


def build_config(args) -> AnalysisConfig:
    """Build AnalysisConfig from CLI arguments."""
    symbol = args.symbol.upper().replace("/", "").replace("USDT", "")
    follower = f"{symbol}USDT"
    leader = "BTCUSDT"

    # Fee profile
    if args.custom_fee is not None:
        fee = FeeProfile("Custom", args.custom_fee, 2, f"Custom {args.custom_fee*100:.3f}%/leg")
    else:
        fee = FEE_MAP.get(args.fee, BINANCE_FUTURES_TAKER)

    strategy = StrategyParams(
        btc_window_s=args.btc_window,
        btc_threshold_pct=args.btc_threshold,
        tp_pct=args.tp,
        sl_pct=args.sl,
        max_hold_s=args.max_hold,
        cooldown_s=args.cooldown,
        min_volume_ratio=args.vol_ratio,
    )

    return AnalysisConfig(
        leader_symbol=leader,
        follower_symbol=follower,
        days=args.days,
        strategy_params=strategy,
        fee_profile=fee,
        leverage_levels=args.leverage,
        initial_capital=args.capital,
        mc_random_trials=args.mc_trials,
        mc_risk_permutations=args.mc_risk,
        optuna_trials=args.optuna_trials,
        output_dir=args.output_dir,
        generate_pdf=not args.no_pdf,
        stage=args.stage,
        skip_optuna=args.no_optuna,
    )


def run_pipeline(config: AnalysisConfig):
    """Run the full analysis pipeline."""
    coin = config.follower_symbol.replace("USDT", "")
    run_all = config.stage == "all"

    console.print(Panel(
        f"[bold]{coin}/BTC Catch-Up Trade Analysis[/]\n"
        f"Days: {config.days} | Fee: {config.fee_profile.name} "
        f"({config.fee_profile.round_trip_pct:.2f}% r/t)\n"
        f"TP: {config.strategy_params.tp_pct}% | SL: {config.strategy_params.sl_pct}% | "
        f"BTC threshold: >{config.strategy_params.btc_threshold_pct}%",
        style="blue",
    ))

    # ── 1. DATA ───────────────────────────────────────────────────
    end_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=config.days)

    console.print(Panel("[bold]STEP 1: Loading Data[/]", style="blue"))
    ts, btc_prices, btc_vols, f_prices, f_vols = load_aligned_pair(
        config.leader_symbol.replace("USDT", "/USDT"),
        config.follower_symbol.replace("USDT", "/USDT"),
        start_date, config.days,
        bin_ms=config.bin_ms,
    )

    if config.stage == "data":
        console.print("[green]Data loaded successfully.[/]")
        return

    # Collect all results for JSON/PDF
    results = {
        "meta": {
            "leader": config.leader_symbol,
            "follower": config.follower_symbol,
            "coin": coin,
            "days": config.days,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "data_points": len(ts),
            "strategy_params": config.strategy_params.__dict__,
            "fee_profile": config.fee_profile.name,
            "fee_rt_pct": config.fee_profile.round_trip_pct,
        },
    }

    # ── 2. IMPULSE DETECTION ──────────────────────────────────────
    if run_all or config.stage == "impulse":
        console.print(Panel("[bold]STEP 2: Impulse Detection[/]", style="blue"))
        from impulse import detect_impulse_events, summarize_impulse_events
        from output import print_impulse_summary

        events = detect_impulse_events(
            ts, btc_prices, btc_vols,
            ts, f_prices, f_vols,
            windows_s=config.windows_s,
            btc_thresholds=config.btc_thresholds,
            response_horizons_s=config.response_horizons_s,
            min_gap_s=config.min_gap_s,
        )
        print_impulse_summary(events, coin)
        results["impulse_summary"] = summarize_impulse_events(events)

        if config.stage == "impulse":
            _save_and_exit(results, config, coin)
            return

    # ── 3. STRATEGY SIMULATION ────────────────────────────────────
    if run_all or config.stage == "strategy":
        console.print(Panel("[bold]STEP 3: Strategy Simulation[/]", style="blue"))
        from strategy import simulate_tpsl_strategy
        from output import print_strategy_results

        strat_result = simulate_tpsl_strategy(
            ts, btc_prices, btc_vols,
            ts, f_prices, f_vols,
            config.strategy_params,
            config.fee_profile,
            leverage=1.0,
        )
        print_strategy_results(strat_result, coin)
        results["strategy"] = {k: v for k, v in strat_result.items() if k != "trades"}
        results["strategy"]["total_trades"] = strat_result["total_trades"]
        strategy_trades = strat_result["trades"]

        if config.stage == "strategy":
            _save_and_exit(results, config, coin)
            return
    else:
        strategy_trades = []

    # ── 4. BASELINE COMPARISON ────────────────────────────────────
    if (run_all or config.stage == "baseline") and strategy_trades:
        console.print(Panel("[bold]STEP 4: Random Baseline Comparison[/]", style="blue"))
        from baseline import random_baseline_comparison
        from output import print_baseline_results

        baseline_result = random_baseline_comparison(
            f_prices, strategy_trades,
            config.strategy_params, config.fee_profile,
            leverage=1.0,
            n_trials=config.mc_random_trials,
        )
        # Remove distributions from saved results (too large)
        results["baseline"] = {k: v for k, v in baseline_result.items()
                               if not k.endswith("_distribution")}
        print_baseline_results(baseline_result, coin)

        if config.stage == "baseline":
            _save_and_exit(results, config, coin)
            return

    # ── 5. REGIME ANALYSIS ────────────────────────────────────────
    if (run_all or config.stage == "regime") and strategy_trades:
        console.print(Panel("[bold]STEP 5: Market Regime Analysis[/]", style="blue"))
        from regime import classify_daily_regimes, regime_summary
        from output import print_regime_results

        regimes = classify_daily_regimes(ts, btc_prices, ts, f_prices, strategy_trades)
        reg_summary = regime_summary(regimes)
        print_regime_results(regimes, reg_summary, coin)
        results["regime_summary"] = reg_summary

        if config.stage == "regime":
            _save_and_exit(results, config, coin)
            return

    # ── 6. RISK PROFILE ──────────────────────────────────────────
    if (run_all or config.stage == "risk") and strategy_trades:
        console.print(Panel("[bold]STEP 6: Monte Carlo Risk Profile[/]", style="blue"))
        from risk import risk_profile_monte_carlo
        from output import print_risk_results

        risk_result = risk_profile_monte_carlo(
            strategy_trades,
            leverage_levels=config.leverage_levels,
            n_permutations=config.mc_risk_permutations,
            initial_capital=config.initial_capital,
            fee_rt_pct=config.fee_profile.round_trip_pct,
        )
        print_risk_results(risk_result, config.initial_capital)
        results["risk"] = risk_result

        if config.stage == "risk":
            _save_and_exit(results, config, coin)
            return

    # ── 7. OPTUNA OPTIMIZATION ────────────────────────────────────
    if (run_all or config.stage == "optuna") and not config.skip_optuna:
        console.print(Panel("[bold]STEP 7: Optuna Parameter Optimization[/]", style="blue"))
        from optimize import optimize_parameters
        from output import print_optuna_results

        best_params, opt_summary = optimize_parameters(
            ts, btc_prices, btc_vols,
            ts, f_prices, f_vols,
            config.fee_profile,
            leverage=1.0,
            n_trials=config.optuna_trials,
        )
        print_optuna_results(best_params, opt_summary, coin)
        results["optuna"] = opt_summary

        if config.stage == "optuna":
            _save_and_exit(results, config, coin)
            return

    # ── 8. SAVE & REPORT ──────────────────────────────────────────
    _save_and_exit(results, config, coin)


def _save_and_exit(results: dict, config: AnalysisConfig, coin: str):
    """Save JSON results and optionally generate PDF."""
    from output import save_json_results

    output_dir = Path(config.output_dir) / f"{coin}_{datetime.now().strftime('%Y-%m-%d')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "results.json"
    save_json_results(results, str(json_path))

    if config.generate_pdf and "strategy" in results:
        try:
            from report import generate_pdf_report
            pdf_path = str(output_dir / f"{coin}_BTC_analysis.pdf")
            generate_pdf_report(config.follower_symbol, results, pdf_path)
            console.print(f"  [green]PDF report saved to:[/] {pdf_path}")
        except ImportError:
            console.print("  [yellow]reportlab not installed — skipping PDF.[/]")
        except Exception as e:
            console.print(f"  [red]PDF generation failed:[/] {e}")

    console.print(Panel(
        f"[bold green]Analysis complete for {coin}/BTC[/]\n"
        f"Results: {output_dir}",
        border_style="green",
    ))


def main():
    parser = argparse.ArgumentParser(
        description="BTC-Leader Catch-Up Trade Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 analyze.py ETH\n"
               "  python3 analyze.py SOL --days 30 --no-optuna\n"
               "  python3 analyze.py DOGE --fee bybit-maker --leverage 5 10\n",
    )

    parser.add_argument("symbol", help="Follower coin (e.g., ETH, SOL, DOGE, XRP)")

    # Data range
    parser.add_argument("--days", type=int, default=7)

    # Pipeline control
    parser.add_argument("--stage", default="all",
                        choices=["data", "impulse", "strategy", "baseline",
                                 "regime", "risk", "optuna", "report", "all"])
    parser.add_argument("--no-optuna", action="store_true", help="Skip Optuna optimization")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF report")

    # Strategy parameters
    parser.add_argument("--tp", type=float, default=0.15, help="Take-profit %% (default: 0.15)")
    parser.add_argument("--sl", type=float, default=0.50, help="Stop-loss %% (default: 0.50)")
    parser.add_argument("--btc-threshold", type=float, default=0.5, help="Min BTC move %% (default: 0.5)")
    parser.add_argument("--btc-window", type=int, default=300, help="BTC window seconds (default: 300)")
    parser.add_argument("--max-hold", type=int, default=600, help="Max hold seconds (default: 600)")
    parser.add_argument("--cooldown", type=int, default=60, help="Cooldown seconds (default: 60)")
    parser.add_argument("--vol-ratio", type=float, default=1.0, help="Volume ratio filter (default: 1.0=off)")

    # Fees
    parser.add_argument("--fee", default="futures-taker",
                        choices=list(FEE_MAP.keys()))
    parser.add_argument("--custom-fee", type=float, default=None,
                        help="Custom fee per leg as decimal")

    # Leverage & capital
    parser.add_argument("--leverage", type=float, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial capital EUR")

    # Monte Carlo
    parser.add_argument("--mc-trials", type=int, default=500)
    parser.add_argument("--mc-risk", type=int, default=10000)
    parser.add_argument("--optuna-trials", type=int, default=300)

    # Output
    parser.add_argument("--output-dir", default="./output")

    args = parser.parse_args()
    config = build_config(args)
    run_pipeline(config)


if __name__ == "__main__":
    main()
