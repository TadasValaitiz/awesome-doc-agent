# Smooth Trailing Trend Strategy, Pyright-safe: handles Position entry price field with fallback.
# Comments embedded for reasoning and clarity.

from dataclasses import dataclass
from typing import Optional, Literal
import uuid
import numpy as np
import pandas as pd
import talib  # TA-Lib for indicator calculation

from sandbox_files.strategy_module import BaseStrategy, CurrentData
from sandbox_files.strategy_module.optimizer import StrategyOptimizer, StaticParams, BacktestResults
from skopt.space import Integer, Real, Dimension


@dataclass
class TrendCurrentData(CurrentData):
    """
    Holds all required indicator values for the strategy decision step.
    """

    rsi: float


class SmoothTrailingTrendStrategy(BaseStrategy[TrendCurrentData]):
    """
    Smooth Trailing Trend (LONG only, can be extended for SHORT):
      - Entry on price above fast EMA, EMA fast > slow, MACD cross up.
      - Exit on price below EMA fast or trailing ATR stop or take-profit.
      - Risk mgmt: 2% equity per trade, auto halt >3 losses.
    """

    # === Strategy Dimensions used in optimization with default values ===
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    def init(self) -> None:
        # Always call base class init first.
        super().init()

        # Calculate RSI
        self.rsi_series: np.ndarray = talib.RSI(
            self.data.Close, timeperiod=max(2, self.rsi_period)
        )

        # Debug logging - convert numpy types to native Python types
        self.logger.info(
            "Strategy initialized",
            data={
                "data_length": int(len(self.data.Close)),
                "rsi_length": int(len(self.rsi_series)),
                "initial_equity": float(self.equity),
                "params": {
                    "rsi_period": int(self.rsi_period),
                    "rsi_overbought": float(self.rsi_overbought),
                    "rsi_oversold": float(self.rsi_oversold),
                    "buy_treshold": float(self.buy_treshold),
                    "sell_treshold": float(self.sell_treshold),
                },
            },
        )

    def create_current_data(self, idx: int) -> TrendCurrentData | None:
        # Get current index and close price for this step
        price: float = float(self.data.Close[idx])
        time: str = str(self.data.index[idx]) if hasattr(self.data, "index") else ""
        rsi = self.rsi_series[idx]

        if not np.isfinite(rsi) or not np.isnan(rsi):
            rsi = float(rsi)
            return TrendCurrentData(price=price, time=time, rsi=rsi)

        return None

    def signal(self, current_data: TrendCurrentData) -> float:
        """
        Returns a signal value between -1 and 1 based on RSI
        """
        # Clamp RSI to valid range
        rsi = np.clip(current_data.rsi, 0.0, 100.0)

        # Calculate oversold signal (buy zone - positive values)
        # Example: If RSI=0, oversold=30: signal = 1.0 (strongest buy)
        # Example: If RSI=15, oversold=30: signal = 0.5 (moderate buy)
        # Example: If RSI=29, oversold=30: signal = 0.03 (weak buy)
        oversold_signal = ((self.rsi_oversold - rsi) / self.rsi_oversold) * (
            rsi <= self.rsi_oversold
        )

        # Calculate overbought signal (sell zone - negative values)
        # Example: If RSI=70, overbought=70: signal = 0 (threshold)
        # Example: If RSI=85, overbought=70: signal = -0.5 (moderate sell)
        # Example: If RSI=100, overbought=70: signal = -1.0 (strongest sell)
        overbought_signal = -(
            (rsi - self.rsi_overbought) / (100.0 - self.rsi_overbought)
        ) * (rsi >= self.rsi_overbought)

        # Combine signals (only one will be non-zero based on RSI value)
        final_signal = oversold_signal + overbought_signal

        return float(final_signal)


def run_backtest() -> None:
    data = pd.read_parquet("data/optimization/backtest_data.pkl")
    if data.empty:
        raise ValueError("Data is empty")

    session_id = str(uuid.uuid4())

    # Set up optimization parameters
    params = StaticParams(
        cash=1000000,
        commission=0.001,
        cv_splits=5,
        spread=0.002,  # Changed from 0.1 (10%) to 0.002 (0.2%) which is more realistic
        trade_on_close=True,  # Changed to True so orders execute at the close price
        exclusive_orders=True,  # Prevent conflicting orders
    )

    # Fix optimization space to ensure valid MACD parameters
    space_tuples = [
        ("rsi_period", Integer(3, 30)),
        ("rsi_overbought", Real(50.0, 95.0)),
        ("rsi_oversold", Real(5.0, 40.0)),
        ("buy_treshold", Real(0.1, 1)),
        ("sell_treshold", Real(-1, -0.1)),
        ("risk_per_trade", Real(0.01, 0.1)),
        ("stop_loss_pct", Real(0.01, 0.1)),
        ("take_profit_pct", Real(0.01, 0.1)),
    ]

    optimizer = StrategyOptimizer(
        session_id,
        data,
        SmoothTrailingTrendStrategy,
        params,
        space_tuples,
    )

    params = {
        "rsi_period": 6,
        "rsi_overbought": 51.689886013552325,
        "rsi_oversold": 11.662041347071453,
        "buy_treshold": 0.6483473839450034,
        "sell_treshold": -0.49751884355497267,
        "risk_per_trade": 0.0948290261454341,
        "stop_loss_pct": 0.08545823955499404,
        "take_profit_pct": 0.09931050988780836,
    }
    # series, results = optimizer.singe_run(params)
    optimizer.optimize(n_calls=10, n_random_starts=2)
    optimizer.save_optimization_results("data/optimization/results.json")
    optimizer.plot_optimization_results("data/optimization")


if __name__ == "__main__":
    run_backtest()
