"""Console formatting and JSON serialization."""

import json
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from config import ImpulseEvent, TradeResult, DayRegime

console = Console()


def print_impulse_summary(events: list[ImpulseEvent], follower_name: str):
    """Rich console display of impulse event analysis."""
    console.print(Panel(f"[bold]IMPULSE ANALYSIS — BTC → {follower_name}[/]", style="blue"))
    console.print(f"  Detected [bold]{len(events)}[/] impulse events total\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Window", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Events", justify="right")
    table.add_column("Median Lag", justify="right")
    table.add_column(f"Avg {follower_name}\nalready\nfollowed", justify="right")
    table.add_column(f"{follower_name} resp\n+30s", justify="right")
    table.add_column(f"{follower_name} resp\n+60s", justify="right")
    table.add_column("Rel.gain\n+60s", justify="right")
    table.add_column("Win rate\n+60s", justify="right")

    for window_s in [5, 10, 30, 60, 120, 300]:
        for min_th in [0.1, 0.3, 0.5, 1.0]:
            subset = [e for e in events if e.window_s == window_s and abs(e.btc_move_pct) >= min_th]
            if len(subset) < 2:
                continue

            lags = [e.measured_lag_ms for e in subset]
            followed = [e.follower_already_followed_pct for e in subset]

            def avg_resp(horizon):
                vals = [e.follower_response.get(horizon, 0) for e in subset if horizon in e.follower_response]
                return np.mean(vals) if vals else 0

            def avg_rg(horizon):
                vals = [e.relative_gain.get(horizon, 0) for e in subset if horizon in e.relative_gain]
                return np.mean(vals) if vals else 0

            def wr(horizon):
                vals = [e.relative_gain.get(horizon, 0) for e in subset if horizon in e.relative_gain]
                return sum(1 for v in vals if v > 0) / len(vals) * 100 if vals else 0

            rg60 = avg_rg(60)
            wr60 = wr(60)

            wr_style = "bold green" if wr60 >= 55 else "yellow" if wr60 >= 45 else "dim red"

            table.add_row(
                f"{window_s}s", f">{min_th}%", str(len(subset)),
                f"{np.median(lags):.0f}ms",
                f"{np.mean(followed):.0f}%",
                f"{avg_resp(30):+.3f}%",
                f"{avg_resp(60):+.3f}%",
                Text(f"{rg60:+.3f}%", style="green" if rg60 > 0.01 else "red" if rg60 < -0.01 else "dim"),
                Text(f"{wr60:.0f}%", style=wr_style),
            )

    console.print(table)
    console.print()


def print_strategy_results(results: dict, follower_name: str):
    """Rich console display of strategy simulation results."""
    console.print(Panel(f"[bold]STRATEGY RESULTS — {follower_name}[/]", style="blue"))

    if results["total_trades"] == 0:
        console.print("  [red]No trades generated.[/]\n")
        return

    style = "bold green" if results["total_net_pct"] > 0 else "bold red"
    console.print(Panel(
        f"  Trades: {results['total_trades']} | "
        f"Win rate: {results['win_rate']:.1f}%\n"
        f"  TP: {results['tp_rate']:.0f}% | SL: {results['sl_rate']:.0f}% | "
        f"Timeout: {results['timeout_rate']:.0f}%\n"
        f"  Avg gross: {results['avg_gross_pct']:+.4f}% | "
        f"Avg net: {results['avg_net_pct']:+.4f}%\n"
        f"  Total gross: {results['total_gross_pct']:+.3f}% | "
        f"Total net: {results['total_net_pct']:+.3f}%\n"
        f"  Max drawdown: {results['max_drawdown_pct']:.2f}%\n"
        f"  Leverage: {results.get('leverage', 1)}x | "
        f"Fees: {results.get('fee_profile', '?')} ({results.get('fee_rt_pct', 0):.2f}% r/t)",
        border_style=style,
    ))

    # Show last 20 trades
    trades = results["trades"]
    if trades:
        shown = trades[-20:] if len(trades) > 20 else trades
        title = f"Trades ({len(trades)} total"
        title += f", last 20)" if len(trades) > 20 else ")"

        t_table = Table(title=title, show_header=True, header_style="bold magenta")
        t_table.add_column("Time UTC", style="dim")
        t_table.add_column("Dir", justify="center")
        t_table.add_column("BTC imp.", justify="right")
        t_table.add_column(f"{follower_name} ret.", justify="right")
        t_table.add_column("Exit", justify="center")
        t_table.add_column("Gross", justify="right")
        t_table.add_column("Net", justify="right")
        t_table.add_column("", justify="center")

        for t in shown:
            dt = datetime.fromtimestamp(t.entry_time, tz=timezone.utc)
            pnl_style = "green" if t.net_pnl_pct > 0 else "red"
            t_table.add_row(
                dt.strftime("%m-%d %H:%M:%S"),
                Text("▲L" if t.direction == "LONG" else "▼S",
                     style="bold green" if t.direction == "LONG" else "bold red"),
                f"{t.btc_impulse_pct:+.2f}%",
                f"{t.follower_return_pct:+.3f}%",
                t.exit_reason,
                f"{t.gross_pnl_pct:+.3f}%",
                Text(f"{t.net_pnl_pct:+.3f}%", style=pnl_style),
                Text("WIN" if t.net_pnl_pct > 0 else "LOSS",
                     style="bold green" if t.net_pnl_pct > 0 else "bold red"),
            )

        console.print(t_table)
    console.print()


def print_baseline_results(baseline: dict, follower_name: str):
    """Display random baseline comparison results."""
    console.print(Panel(f"[bold]BASELINE — BTC Trigger vs Random ({follower_name})[/]", style="blue"))

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric", justify="left")
    table.add_column("BTC Trigger", justify="right")
    table.add_column("Random Mean", justify="right")
    table.add_column("Random Std", justify="right")

    table.add_row(
        "Win Rate",
        f"{baseline['strategy_win_rate']:.1f}%",
        f"{baseline['random_mean_win_rate']:.1f}%",
        f"±{baseline['random_std_win_rate']:.1f}%",
    )
    table.add_row(
        "Avg Net/Trade",
        f"{baseline['strategy_avg_net']:+.4f}%",
        f"{baseline['random_mean_avg_net']:+.4f}%",
        f"±{baseline['random_std_avg_net']:.4f}%",
    )
    console.print(table)

    pctl = baseline["percentile_rank_wr"]
    style = "bold green" if pctl >= 95 else "yellow" if pctl >= 80 else "red"
    console.print(f"\n  BTC trigger beats random: [bold]{pctl:.0f}%[/] of {baseline['n_trials']} trials")
    console.print(f"  p-value: {baseline['p_value']:.4f}")
    console.print()


def print_regime_results(regimes: list[DayRegime], summary: dict, follower_name: str):
    """Display market regime analysis."""
    console.print(Panel(f"[bold]REGIME ANALYSIS — {follower_name}[/]", style="blue"))

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Regime")
    table.add_column("Days", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Avg WR", justify="right")
    table.add_column(f"Avg {follower_name}\nDrift", justify="right")
    table.add_column("Avg BTC\nDrift", justify="right")

    for regime in ["BULL", "BEAR", "FLAT"]:
        s = summary.get(regime, {})
        if s.get("days", 0) == 0:
            continue
        wr_style = "green" if s.get("avg_win_rate", 0) >= 60 else "yellow"
        table.add_row(
            regime, str(s["days"]), str(s["total_trades"]),
            Text(f"{s.get('avg_win_rate', 0):.0f}%", style=wr_style),
            f"{s.get('avg_follower_drift', 0):+.2f}%",
            f"{s.get('avg_btc_drift', 0):+.2f}%",
        )

    console.print(table)
    console.print()


def print_risk_results(risk_data: dict, initial_capital: float):
    """Display Monte Carlo risk profiling."""
    console.print(Panel("[bold]RISK PROFILE — Monte Carlo[/]", style="blue"))

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Leverage", justify="right")
    table.add_column("Median\nReturn", justify="right")
    table.add_column("Median\nMax DD", justify="right")
    table.add_column("P95\nMax DD", justify="right")
    table.add_column("Prob\nNet Loss", justify="right")
    table.add_column("Prob\n>50% DD", justify="right")
    table.add_column("Median\nFinal", justify="right")

    for key in sorted(risk_data.keys(), key=lambda x: float(x)):
        r = risk_data[key]
        ret_style = "green" if r["median_return_pct"] > 0 else "red"
        table.add_row(
            f"{r['leverage']}x",
            Text(f"{r['median_return_pct']:+.1f}%", style=ret_style),
            f"{r['median_max_dd_pct']:.1f}%",
            f"{r['p95_max_dd_pct']:.1f}%",
            f"{r['prob_net_loss']:.0f}%",
            f"{r['prob_50pct_dd']:.1f}%",
            f"€{r['median_final_capital']:,.0f}",
        )

    console.print(table)
    console.print()


def print_optuna_results(best_params: dict, summary: dict, follower_name: str):
    """Display Optuna optimization results."""
    console.print(Panel(f"[bold]OPTUNA OPTIMIZATION — {follower_name}[/]", style="blue"))

    console.print(Panel(
        f"  BTC Window: {best_params.get('btc_window_s', 0):.1f}s\n"
        f"  BTC Threshold: >{best_params.get('btc_threshold_pct', 0):.3f}%\n"
        f"  Take-Profit: {best_params.get('tp_pct', 0):.3f}%\n"
        f"  Stop-Loss: {best_params.get('sl_pct', 0):.3f}%\n"
        f"  Max Hold: {best_params.get('max_hold_s', 0):.0f}s\n"
        f"  Cooldown: {best_params.get('cooldown_s', 0):.0f}s\n"
        f"  Volume Ratio: >{best_params.get('min_volume_ratio', 0):.1f}x",
        title="BEST PARAMETERS", border_style="bold green",
    ))

    # Parameter importance
    importance = summary.get("param_importance", {})
    if importance:
        console.print("\n[bold cyan]Parameter Importance:[/]")
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(imp * 40)
            console.print(f"  {param:25s} {bar} {imp:.1%}")

    console.print()


def save_json_results(results: dict, output_path: str):
    """Save all analysis results as structured JSON.

    Converts dataclass instances to dicts for serialization.
    """
    def serialize(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: serialize(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [serialize(i) for i in obj]
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        return obj

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(serialize(results), f, indent=2, default=str)

    console.print(f"\n  [green]Results saved to:[/] {path}")
