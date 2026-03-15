"""Monte Carlo risk profiling — drawdown distributions at various leverage levels."""

import numpy as np
from config import TradeResult


def risk_profile_monte_carlo(
    trades: list[TradeResult],
    leverage_levels: list[float] = None,
    n_permutations: int = 10000,
    initial_capital: float = 1000.0,
    fee_rt_pct: float = 0.08,
    seed: int = 42,
) -> dict:
    """Monte Carlo drawdown analysis at various leverage levels.

    Randomly permutes the trade sequence to estimate drawdown distributions.

    Returns dict keyed by leverage level with drawdown percentiles and return stats.
    """
    if leverage_levels is None:
        leverage_levels = [1, 3, 5, 10]

    if not trades:
        return {str(lev): {"median_max_dd_pct": 0, "prob_net_loss": 100}
                for lev in leverage_levels}

    # Extract raw follower returns (before leverage/fees)
    raw_returns = [t.follower_return_pct for t in trades]
    directions = [t.direction for t in trades]
    n_trades = len(raw_returns)

    rng = np.random.default_rng(seed)

    results = {}
    for lev in leverage_levels:
        max_dds = []
        final_capitals = []
        max_consecutive_losses = []

        for _ in range(n_permutations):
            # Shuffle trade order
            perm = rng.permutation(n_trades)
            equity = initial_capital
            peak = initial_capital
            max_dd = 0.0
            consec_loss = 0
            max_consec = 0

            for idx in perm:
                ret = raw_returns[idx]
                # Apply leverage to return, fee stays fixed
                pnl_pct = ret * lev - fee_rt_pct
                equity *= (1 + pnl_pct / 100)

                if equity <= 0:
                    equity = 0
                    max_dd = 100.0
                    break

                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak * 100
                if dd > max_dd:
                    max_dd = dd

                if pnl_pct < 0:
                    consec_loss += 1
                    max_consec = max(max_consec, consec_loss)
                else:
                    consec_loss = 0

            max_dds.append(max_dd)
            final_capitals.append(equity)
            max_consecutive_losses.append(max_consec)

        max_dds = np.array(max_dds)
        finals = np.array(final_capitals)

        results[str(lev)] = {
            "leverage": lev,
            "median_max_dd_pct": float(np.median(max_dds)),
            "p75_max_dd_pct": float(np.percentile(max_dds, 75)),
            "p95_max_dd_pct": float(np.percentile(max_dds, 95)),
            "p99_max_dd_pct": float(np.percentile(max_dds, 99)),
            "prob_50pct_dd": float(np.mean(max_dds >= 50) * 100),
            "prob_ruin": float(np.mean(max_dds >= 90) * 100),
            "median_final_capital": float(np.median(finals)),
            "mean_final_capital": float(np.mean(finals)),
            "p5_final_capital": float(np.percentile(finals, 5)),
            "p95_final_capital": float(np.percentile(finals, 95)),
            "median_return_pct": float((np.median(finals) / initial_capital - 1) * 100),
            "prob_net_loss": float(np.mean(finals < initial_capital) * 100),
            "median_max_consec_losses": float(np.median(max_consecutive_losses)),
        }

    return results
