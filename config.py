"""Configuration dataclasses, fee profiles, and result structures."""

from dataclasses import dataclass, field


# ── Fee Profiles ──────────────────────────────────────────────────

@dataclass
class FeeProfile:
    """Trading fee configuration for a specific platform/tier."""
    name: str
    fee_per_leg: float          # e.g. 0.0004 = 0.04%
    legs_per_trade: int = 2     # 2 for directional, 4 for pairs
    description: str = ""

    @property
    def round_trip_pct(self) -> float:
        return self.fee_per_leg * self.legs_per_trade * 100


BINANCE_SPOT_TAKER = FeeProfile("Binance Spot Taker", 0.001, 2, "Spot market orders (0.10%/leg)")
BINANCE_FUTURES_TAKER = FeeProfile("Binance Futures Taker", 0.0004, 2, "USDT-M futures taker (0.04%/leg)")
BINANCE_FUTURES_MAKER = FeeProfile("Binance Futures Maker", 0.0002, 2, "USDT-M futures limit (0.02%/leg)")
BYBIT_FUTURES_MAKER = FeeProfile("Bybit Futures Maker", 0.0001, 2, "USDT perp maker (0.01%/leg)")

FEE_MAP = {
    "spot": BINANCE_SPOT_TAKER,
    "futures-taker": BINANCE_FUTURES_TAKER,
    "futures-maker": BINANCE_FUTURES_MAKER,
    "bybit-maker": BYBIT_FUTURES_MAKER,
}


# ── Strategy Parameters ──────────────────────────────────────────

@dataclass
class StrategyParams:
    """Parameters for the TP/SL directional strategy."""
    btc_window_s: float = 300       # BTC impulse detection window
    btc_threshold_pct: float = 0.5  # Minimum BTC move % to trigger
    tp_pct: float = 0.15            # Take-profit target %
    sl_pct: float = 0.50            # Stop-loss limit %
    max_hold_s: float = 600         # Maximum hold time before forced exit
    cooldown_s: float = 60          # Min time between trades
    execution_delay_s: int = 1      # API execution latency (seconds/bins)
    slippage_bps: float = 0.0      # Slippage per leg in basis points
    min_volume_ratio: float = 1.0   # Volume burst filter (1.0 = disabled)
    vol_baseline_s: int = 120       # Volume baseline window
    vol_burst_s: int = 5            # Volume burst window


# ── Analysis Configuration ────────────────────────────────────────

@dataclass
class AnalysisConfig:
    """Top-level analysis configuration."""
    leader_symbol: str = "BTCUSDT"
    follower_symbol: str = "ETHUSDT"
    days: int = 7
    bin_ms: int = 1000

    # Impulse detection
    windows_s: list[int] = field(default_factory=lambda: [5, 10, 30, 60, 120, 300])
    btc_thresholds: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    response_horizons_s: list[int] = field(default_factory=lambda: [10, 30, 60, 120, 300, 600])
    min_gap_s: int = 60

    # Strategy
    strategy_params: StrategyParams = field(default_factory=StrategyParams)

    # Fees
    fee_profile: FeeProfile = field(default_factory=lambda: BINANCE_FUTURES_TAKER)

    # Monte Carlo
    mc_random_trials: int = 500
    mc_risk_permutations: int = 10000
    leverage_levels: list[float] = field(default_factory=lambda: [1, 3, 5, 10])
    initial_capital: float = 1000.0

    # Optuna
    optuna_trials: int = 300

    # Output
    output_dir: str = "./output"
    generate_pdf: bool = True

    # Pipeline control
    stage: str = "all"       # "all", "data", "impulse", "strategy", "baseline", "regime", "risk", "optuna", "report"
    skip_optuna: bool = False


# ── Result Structures ─────────────────────────────────────────────

@dataclass
class ImpulseEvent:
    """A detected BTC impulse event with follower response metrics."""
    timestamp_s: float
    btc_move_pct: float
    window_s: float
    direction: str                          # "UP" or "DOWN"
    follower_move_same_window: float = 0.0
    follower_already_followed_pct: float = 0.0
    measured_lag_ms: float = 0.0
    follower_response: dict = field(default_factory=dict)
    btc_continuation: dict = field(default_factory=dict)
    relative_gain: dict = field(default_factory=dict)


@dataclass
class TradeResult:
    """A single simulated trade."""
    entry_time: float
    exit_time: float
    direction: str              # "LONG" or "SHORT"
    btc_impulse_pct: float
    follower_entry_price: float
    follower_exit_price: float
    follower_return_pct: float
    exit_reason: str            # "TP", "SL", "TIMEOUT"
    gross_pnl_pct: float
    fee_pct: float
    net_pnl_pct: float


@dataclass
class DayRegime:
    """Market regime classification for a single day."""
    date: str
    btc_return_pct: float
    follower_return_pct: float
    regime: str                 # "BULL", "BEAR", "FLAT"
    trades_count: int = 0
    win_rate: float = 0.0
    avg_return_pct: float = 0.0
