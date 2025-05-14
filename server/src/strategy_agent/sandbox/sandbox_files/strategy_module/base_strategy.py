# Base Strategy implementation
# File name base_strategy.py
import abc
import uuid
from backtesting import Strategy
from typing import List, Literal, Optional, TypeVar, Generic
import pandas as pd
from .trading_logger import TradingLogger
from .logger import ConsoleLogger, FileLogger
from .current_data import CurrentData

# Define a TypeVar for the custom CurrentData subclass
Data = TypeVar("Data", bound=CurrentData)


class BaseStrategy(Generic[Data], Strategy, metaclass=abc.ABCMeta):
    """
    Base abstract class for trading strategies using the backtesting.py framework.
    Requires implementation of specific abstract methods for strategy logic.
    """

    # --- Strategy Data ---
    current_data: Data | None = None
    current_idx: int = 0

    # --- Strategy Optimization Parameters ---
    risk_per_trade: float = 0.02
    buy_treshold: float = 0.3
    sell_treshold: float = -0.3
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06

    # --- Internal ---
    logger: TradingLogger

    def _create_logger(self, session_id: str) -> TradingLogger:
        """Creates the logger instance for the strategy session."""
        # Default implementation uses ConsoleLogger. Override if different logging is needed.
        return TradingLogger(
            session_id=session_id,
            loggers=[FileLogger(session_id=session_id)],
        )

    def init(self) -> None:
        """
        Initialize the strategy. Called once before the backtest starts.
        Use this to:
        1. Call super().init()
        2. Initialize logger.
        3. Prepare data series (like self.close_series).
        4. Initialize strategy parameters.
        5. Set up indicators (e.g., create indicator objects from libraries like talib).
           Calculation of indicator *values* for each step happens later.
        """
        self.session_id = str(uuid.uuid4())
        self.logger = self._create_logger(session_id=self.session_id)
        self.close_series = pd.Series(
            self.data.Close
        )  # Ensure self.data.Close is available
        # Initialize other common variables if needed

    def pre_trading(self) -> None:
        """
        Called before the trading logic for the current step.
        """
        self.current_data = self.create_current_data(self.current_idx)
        self.current_equity = self.equity
        self.current_closed_trades = tuple(self.closed_trades)

    def post_trading(self) -> None:
        """
        Called after the trading logic for the current step.
        """
        closed_trades = tuple(
            trade
            for trade in self.closed_trades
            if trade not in self.current_closed_trades
        )

        closed_trades_size = len(closed_trades)
        if closed_trades_size > 0:
            self.logger.info(
                f"[Post Trading] Closed trades: {len(closed_trades)}",
                data={
                    "size": sum([trade.size for trade in closed_trades]),
                    "entry_price": sum([trade.entry_price for trade in closed_trades]),
                    "exit_price": sum(
                        [
                            trade.exit_price
                            for trade in closed_trades
                            if trade.exit_price is not None
                        ]
                    ),
                    "commissions": sum([trade._commissions for trade in closed_trades]),
                },
            )
        self.current_idx += 1

    def next(self) -> None:
        """
        Called on each data point (candle/bar) after init().
        Orchestrates the strategy logic flow for a single step.
        """
        self.pre_trading()

        if self.current_data is not None:
            portion_to_close = self.close_portion_of_trade(self.current_data)
            if portion_to_close > 0:
                self.position.close(portion_to_close)
            else:
                self.trading(self.current_data)

        self.post_trading()

    # --- Abstract Methods (MUST be implemented by subclasses) ---

    @abc.abstractmethod
    def create_current_data(self, index: int) -> Data | None:
        """
        Calculate and return an instance of the strategy's specific CurrentData subclass.
        This method should compute all required indicator values for the *current* step
        based on historical data (e.g., self.data.Close, self.close_series) and
        return them structured in the custom Data object.
        Example: Fetch self.data.Close[-1], calculate EMA[-1], RSI[-1], etc.
        """
        pass

    @abc.abstractmethod
    def signal(self, current_data: Data) -> float:
        """
        Calculate and return a signal value between -1 and 1 based on current data
        """
        pass

    def close_portion_of_trade(self, current_data: Data) -> float:
        """
        Close a portion of the current trade based on the current data
        This is for risk management purpose, to close a portion of the trade
        """
        return 0

    def trading(self, current_data: Data) -> None:
        """
        Execute trading logic based on signal and position size
        """
        signal: float = self.signal(current_data)

        if signal > self.buy_treshold:
            self.long_position(current_data)
        elif signal < self.sell_treshold:
            self.short_position(current_data)

    def position_size(self, current_data: Data) -> float:
        """
        Returns a position size value based on risk per trade
        Ensures size is either a positive fraction (0 < size < 1) or positive whole number of units
        """
        risk_amount = self.equity * self.risk_per_trade
        units = risk_amount / current_data.price

        # Handle edge cases to ensure valid position size
        if 0 < units < 1:
            # Return as fraction of equity (between 0 and 1)
            return units
        elif units >= 1:
            # Return as whole number of units
            return round(units)
        else:
            # Fallback for any negative or zero values
            return 1.0  # Default to 1 unit as minimum valid position

    def long_position(self, current_data: Data) -> None:
        """Handle opening a long position or closing a short position"""
        position_size = self.position_size(current_data)

        # Close short position
        if self.position.is_short:
            self.position.close()

            self.logger.info(
                "SHORT position closed by LONG signal",
                data={
                    "price": current_data.price,
                    "size": position_size,
                    "equity": self.equity,
                    "orders": len(self.orders),
                },
            )
        # Open new long position
        elif not self.position.is_long:
            # Calculate stop loss by applying the percentage loss to the current price
            # Ensure stop_loss is strictly lower than price for long positions
            stop_loss = current_data.price * (1 - self.stop_loss_pct)
            take_profit = current_data.price * (1 + self.take_profit_pct)
            limit = current_data.price * (1 - 0.005)

            self.logger.info(
                "LONG position about to be opened",
                data={
                    "price": current_data.price,
                    "size": position_size,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "equity": self.equity,
                    "orders": len(self.orders),
                },
            )

            # Use market order instead of limit order to ensure execution
            self.buy(
                size=position_size,
                limit=limit,
                sl=stop_loss,
                tp=take_profit,
            )

    def short_position(self, current_data: Data) -> None:
        """Handle opening a short position or closing a long position"""
        position_size = self.position_size(current_data)

        # Close long position
        if self.position.is_long:
            self.position.close()
            self.logger.info(
                "LONG position closed by SHORT signal",
                data={
                    "price": current_data.price,
                    "size": position_size,
                    "equity": self.equity,
                    "orders": len(self.orders),
                },
            )
        # Open new short position
        elif not self.position.is_short:
            # Calculate stop loss by applying the percentage gain to the current price
            # Ensure stop_loss is strictly higher than price for short positions
            stop_loss = current_data.price * (1 + self.stop_loss_pct)
            take_profit = current_data.price * (1 - self.take_profit_pct)
            limit = current_data.price * (1 + 0.005)  # spread

            self.logger.info(
                "SHORT position about to be opened",
                data={
                    "price": current_data.price,
                    "size": position_size,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "equity": self.equity,
                    "orders": len(self.orders),
                },
            )

            # Use market order instead of limit order to ensure execution
            self.sell(
                size=position_size,
                limit=limit,
                sl=stop_loss,
                tp=take_profit,
            )
