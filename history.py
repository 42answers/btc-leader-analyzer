"""Analysis history: persist and compare results across runs.

Saves a slim snapshot of each analysis run to analysis_history.json.
Enables cross-coin and cross-strategy comparison.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_HISTORY_PATH = Path(__file__).parent / "analysis_history.json"


def serialize(obj):
    """Recursively serialize numpy/dataclass types to JSON-safe Python types."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: serialize(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [serialize(i) for i in obj]
    if isinstance(obj, dict):
        return {str(k): serialize(v) for k, v in obj.items()}
    return obj


def save_analysis_run(results: dict) -> str:
    """Auto-save a completed analysis run. Returns the run_id."""
    meta = results.get("meta", {})
    coin = meta.get("coin", "UNK")
    days = meta.get("days", 0)
    now = datetime.now()
    run_id = f"{coin}_{days}d_{now.strftime('%Y%m%d_%H%M%S')}"

    # Build slim snapshot (exclude trade-level data and large distributions)
    snapshot = {
        "run_id": run_id,
        "saved_at": now.isoformat(),
        "meta": serialize(meta),
    }

    # Strategy metrics (no trades list)
    strat = results.get("strategy", {})
    snapshot["strategy"] = {
        k: serialize(v) for k, v in strat.items()
        if k != "trades"
    }

    # Correlation
    if "correlation" in results:
        snapshot["correlation"] = serialize(results["correlation"])

    # Baseline (no distribution arrays)
    if "baseline" in results:
        bl = results["baseline"]
        snapshot["baseline"] = {
            k: serialize(v) for k, v in bl.items()
            if k not in ("random_wr_distribution", "random_avg_distribution")
        }

    # Risk
    if "risk" in results:
        snapshot["risk"] = serialize(results["risk"])

    # Regime summary
    if "regime_summary" in results:
        snapshot["regime_summary"] = serialize(results["regime_summary"])

    # Optuna (best params + score only)
    if "optuna" in results and results["optuna"]:
        opt = results["optuna"]
        snapshot["optuna"] = {
            "best_params": serialize(opt.get("best_params", {})),
            "best_score": serialize(opt.get("best_score", opt.get("summary", {}).get("best_score", 0))),
        }
        if "param_importance" in opt:
            snapshot["optuna"]["param_importance"] = serialize(opt["param_importance"])
        elif "summary" in opt and "param_importance" in opt["summary"]:
            snapshot["optuna"]["param_importance"] = serialize(opt["summary"]["param_importance"])

    # Out-of-sample (if present)
    if "oos" in results and results["oos"]:
        snapshot["oos"] = serialize(results["oos"])

    # Walk-forward (aggregate only, no fold details)
    if "walkforward" in results and results["walkforward"]:
        wf = results["walkforward"]
        snapshot["walkforward"] = {
            "n_folds": wf.get("n_folds", 0),
            "aggregate_oos": serialize(wf.get("aggregate_oos", {})),
            "degradation_pct": serialize(wf.get("degradation_pct", 0)),
        }

    # Impulse summary
    if "impulse_summary" in results:
        imp = results["impulse_summary"]
        snapshot["impulse_summary"] = {
            "total_events": imp.get("total_events", 0),
            "median_lag_ms": serialize(imp.get("median_lag_ms", 0)),
            "mean_followed_pct": serialize(imp.get("mean_followed_pct", 0)),
        }

    # Catchup
    if "catchup" in results:
        snapshot["catchup"] = serialize(results["catchup"])

    # Load existing history, append, save
    history = load_analysis_history()
    history[run_id] = snapshot

    with open(_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2, default=str)

    return run_id


def load_analysis_history() -> dict:
    """Load all saved analysis runs."""
    if not _HISTORY_PATH.exists():
        return {}
    try:
        with open(_HISTORY_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def delete_analysis_run(run_id: str) -> bool:
    """Delete a single run by ID. Returns True if found and deleted."""
    history = load_analysis_history()
    if run_id not in history:
        return False
    del history[run_id]
    with open(_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2, default=str)
    return True


def get_comparison_dataframe(history: dict, filter_coins: list = None) -> pd.DataFrame:
    """Build a comparison DataFrame from history entries.

    One row per run with key metrics for easy scanning.
    """
    rows = []
    for run_id, entry in sorted(history.items(), key=lambda x: x[1].get("saved_at", ""), reverse=True):
        meta = entry.get("meta", {})
        strat = entry.get("strategy", {})
        bl = entry.get("baseline", {})
        corr = entry.get("correlation", {})
        oos = entry.get("oos", {})
        wf = entry.get("walkforward", {})

        coin = meta.get("coin", "?")
        if filter_coins and coin not in filter_coins:
            continue

        row = {
            "Run ID": run_id,
            "Coin": coin,
            "Days": meta.get("days", 0),
            "Period": f"{meta.get('start_date', '?')} \u2192 {meta.get('end_date', '?')}",
            "Fee": meta.get("fee_profile", "?"),
            "Slippage": meta.get("slippage_bps", 0),
            "Params": meta.get("params_source", "manual"),
            "Trades": strat.get("total_trades", 0),
            "Win Rate %": round(strat.get("win_rate", 0), 1),
            "Avg Net %": round(strat.get("avg_net_pct", 0), 4),
            "Total Net %": round(strat.get("total_net_pct", 0), 2),
            "Max DD %": round(strat.get("max_drawdown_pct", 0), 1),
            "Beats Random %": round(bl.get("percentile_rank_avg", 0), 0),
            "p-value": round(bl.get("p_value", 1.0), 3),
            "Pearson r": round(corr.get("pearson_returns", 0), 3),
            "Beta": round(corr.get("beta", 0), 2),
        }

        # Optional OOS columns
        if oos:
            row["OOS Win Rate %"] = round(oos.get("win_rate", 0), 1)
            row["OOS Avg Net %"] = round(oos.get("avg_net", 0), 4)
        else:
            row["OOS Win Rate %"] = None
            row["OOS Avg Net %"] = None

        # Optional walk-forward
        if wf:
            agg = wf.get("aggregate_oos", {})
            row["WF OOS WR %"] = round(agg.get("win_rate", 0), 1)
            row["WF Degrad %"] = round(wf.get("degradation_pct", 0), 1)
        else:
            row["WF OOS WR %"] = None
            row["WF Degrad %"] = None

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)
