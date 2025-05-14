# Example strategy blueprint
# File name base_example.py
from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple
import uuid
import numpy as np
import pandas as pd
import talib  # TA-Lib for indicator calculation

from strategy_module import BaseStrategy, CurrentData
from strategy_module.optimizer import StrategyOptimizer, StaticParams, BacktestResults
from skopt.space import Integer, Real, Dimension


@dataclass
class ExampleData(CurrentData):
    """
    Holds all required indicator values for your strategy decision step.
    Each field represents a calculated indicator value at the current time step.
    """

    # Inherit price and time from CurrentData
    # Add your indicator fields here, for example:
    # indicator1: float
    # indicator2: float
    # Add any other indicators your strategy will use


class ExampleStrategy(BaseStrategy[ExampleData]):

    # === Strategy Dimensions used in optimization with default values ===
    example_indicator_period: int = 14

    def init(self) -> None:
        # Always call base class init first.
        super().init()

        # Calculate and store indicator series
        # Example:
        # self.indicator1_series = talib.YOUR_INDICATOR(self.data.Close, timeperiod=self.indicator1_period)

    def create_current_data(self, idx: int) -> ExampleData | None:
        price: float = float(self.data.Close[idx])
        time: str = str(self.data.index[idx]) if hasattr(self.data, "index") else ""

        # Collect values from all the indicators
        # indicator1 = self.indicator1_series[idx]

        # check if indicator is valid np.isnan or other method
        # if not np.isfinite(indicator1) or not np.isnan(indicator1):
        #     return ExampleData(price=price, time=time, indicator1=float(indicator1))

        return None

    def signal(self, current_data: ExampleData) -> float:
        """
        Generate a trading signal between -1 and 1 based on current indicators.

        Args:
            current_data: Object containing all calculated indicators for current step

        Returns:
            float: Signal value between -1 (strongest sell) and 1 (strongest buy)
                   where 0 represents a neutral position
        """
        # Implement your signal generation logic here
        # The value MUST be between -1.0 and 1.0
        # Examples of signal generation approaches:

        # 1. Threshold-based approach
        # if current_data.indicator1 > self.indicator1_threshold:
        #     return 0.5  # Buy signal
        # elif current_data.indicator1 < self.indicator1_threshold:
        #     return -0.5  # Sell signal

        # 2. Scaled approach: Map indicators to the -1 to 1 range
        # normalized_value = (current_data.indicator1 - min_value) / (max_value - min_value) * 2 - 1

        # 3. Combined signals: Merge multiple indicators
        # final_signal = (indicator1_signal + indicator2_signal) / 2

        # IMPORTANT: The signal value alone should not account for buy/sell thresholds,
        # as these thresholds are applied separately in the trading logic.

        # return float(final_signal)  # Ensure return value is a Python float
        return 0


def run_backtest() -> None:
    data = pd.read_parquet("data/optimization/backtest_data.pkl")
    if data.empty:
        raise ValueError("Data is empty")

    session_id = str(uuid.uuid4())

    # Leave these parameters as is
    params = StaticParams(
        cash=1000000,
        commission=0.001,
        cv_splits=5,
        spread=0.002,
        trade_on_close=True,
        exclusive_orders=True,
    )

    # Add additional parameters to the optimization space
    space_tuples: List[Tuple[str, Integer | Real]] = [
        ("buy_treshold", Real(0.1, 1)),
        ("sell_treshold", Real(-1, -0.1)),
        ("risk_per_trade", Real(0.01, 0.1)),
        ("stop_loss_pct", Real(0.01, 0.1)),
        ("take_profit_pct", Real(0.01, 0.1)),
        # ("example_indicator_period", Integer(1, 100)),
    ]

    optimizer = StrategyOptimizer(
        session_id,
        data,
        ExampleStrategy,
        params,
        space_tuples,
    )

    optimizer.optimize(n_calls=50, n_random_starts=5)
    # Don't change the file names
    optimizer.save_optimization_results("data/optimization/results.json")
    optimizer.plot_optimization_results("data/optimization")


if __name__ == "__main__":
    run_backtest()
