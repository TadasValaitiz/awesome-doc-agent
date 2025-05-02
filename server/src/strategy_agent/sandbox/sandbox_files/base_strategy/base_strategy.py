# Base Strategy implementation
# File name base_strategy.py
import abc
import uuid
from backtesting import Strategy  # type: ignore
from typing import Literal, Optional, TypeVar, Generic
import pandas as pd
from base_strategy.trading_logger import TradingLogger
from base_strategy.logger import ConsoleLogger
from base_strategy.current_data import CurrentData

# Define a TypeVar for the custom CurrentData subclass
Data = TypeVar("Data", bound=CurrentData)


class BaseStrategy(Generic[Data], Strategy, metaclass=abc.ABCMeta):
    """
    Base abstract class for trading strategies using the backtesting.py framework.
    Requires implementation of specific abstract methods for strategy logic.
    """

    # --- Strategy Data ---
    current_data: (
        Data  # Holds the specific CurrentData subclass instance for the current step
    )
    last_order_data: Optional[Data] = (
        None  # Data at the time of the last order placement attempt
    )
    size: float = 1.0  # Default position size, should be overridden or set in init
    close_series: pd.Series  # Series of closing prices

    # --- Order Tracking ---
    pending_order_type: Optional[Literal["LONG", "SHORT", "CLOSE"]] = None
    pending_order_data: Optional[Data] = None
    candles_since_order: int = 0

    # --- Internal ---
    logger: TradingLogger

    def _create_logger(self, session_id: str) -> TradingLogger:
        """Creates the logger instance for the strategy session."""
        # Default implementation uses ConsoleLogger. Override if different logging is needed.
        return TradingLogger(
            session_id=session_id,
            loggers=[ConsoleLogger(session_id=session_id)],
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
        self.logger = self._create_logger(session_id=str(uuid.uuid4()))
        self.close_series = pd.Series(
            self.data.Close
        )  # Ensure self.data.Close is available
        # Initialize other common variables if needed

    def next(self) -> None:
        """
        Called on each data point (candle/bar) after init().
        Orchestrates the strategy logic flow for a single step.
        """
        # 1. Prepare data for the current step
        self.pre_trading()
        # 2. Execute trading logic
        self.trading()
        # 3. Perform post-trade actions for the current step
        self.post_trading()

    def pre_trading(self) -> None:
        """
        Prepare all necessary data for the current trading step.
        This includes fetching the latest price, calculating current indicator values,
        and populating `self.current_data`.
        """
        # Calls the abstract method to get the specific data structure
        self.current_data = self.create_current_data()
        # Increment candle counter if an order is pending or position exists
        if self.pending_order_type or self.position:
            self.candles_since_order += 1

    # --- Abstract Methods (MUST be implemented by subclasses) ---

    @abc.abstractmethod
    def create_current_data(self) -> Data:
        """
        Calculate and return an instance of the strategy's specific CurrentData subclass.
        This method should compute all required indicator values for the *current* step
        based on historical data (e.g., self.data.Close, self.close_series) and
        return them structured in the custom Data object.
        Example: Fetch self.data.Close[-1], calculate EMA[-1], RSI[-1], etc.
        """
        pass

    @abc.abstractmethod
    def has_position(self) -> bool:
        """Check if there is an active open position (long or short)."""
        pass

    @abc.abstractmethod
    def should_stop_trading(self) -> bool:
        """
        Implement risk management rules. Return True to halt further trading signals.
        Example: Check for max drawdown, consecutive losses, etc.
        """
        pass

    @abc.abstractmethod
    def trading(self) -> None:
        """
        Core strategy logic: decide whether to enter or exit positions.
        - Use self.current_data for decisions.
        - Use self.has_position() to check current state.
        - Use self.should_stop_trading() for risk checks.
        - Call self.trade_signal('LONG'/'SHORT') to enter positions.
        - Call self.close_trades() to exit positions.
        """
        pass

    @abc.abstractmethod
    def post_trading(self) -> None:
        """
        Perform actions after the trading logic for the current step.
        Examples: Update trailing stops, log metrics, update internal counters.
        Check if pending orders were filled and clear pending status.
        """
        # Example: Check if a pending order was filled (backtesting.py handles this implicitly,
        # but you might want custom logic or logging)
        # if self.pending_order_type and self.position: # Simplified check
        #    self.logger.info(f"{self.pending_order_type} order filled/position confirmed", data=self.current_data.__dict__)
        #    self.pending_order_type = None
        #    self.pending_order_data = None
        pass

    # --- Helper Methods (Provided) ---

    def trade_signal(self, type: Literal["LONG", "SHORT"]) -> None:
        """Initiates a trade entry."""
        self.logger.info(
            f"{type} Signal",
            data={
                "type": type,
                "last_price": (
                    float(self.last_order_data.price) if self.last_order_data else None
                ),
                **self.current_data.__dict__,
            },
            tags=["strategy", "signal", type.lower()],
        )
        if type == "LONG":
            self.do_long()
        elif type == "SHORT":
            self.do_short()

    def do_short(self) -> None:
        """Place a short order."""
        try:
            # Consider adding stop loss / take profit logic here or via backtesting.py parameters
            self.sell(
                size=self.size, limit=self.current_data.price
            )  # Using limit order at current price
            self.pending_order_type = "SHORT"
            self.pending_order_data = self.current_data
            self.candles_since_order = 0
            self.last_order_data = self.current_data
            self.logger.info(
                "SHORT order placed",
                data={**self.current_data.__dict__},
                tags=["strategy", "order", "short"],
            )
        except Exception as e:
            self.logger.error(
                "Failed to place SHORT order",
                exception=e,
                tags=["strategy", "error", "short"],
            )

    def do_long(self) -> None:
        """Place a long order."""
        try:
            # Consider adding stop loss / take profit logic here or via backtesting.py parameters
            self.buy(
                size=self.size, limit=self.current_data.price
            )  # Using limit order at current price
            self.pending_order_type = "LONG"
            self.pending_order_data = self.current_data
            self.candles_since_order = 0
            self.last_order_data = self.current_data
            self.logger.info(
                "LONG order placed",
                data={**self.current_data.__dict__},
                tags=["strategy", "order", "long"],
            )
        except Exception as e:
            self.logger.error(
                "Failed to place LONG order",
                exception=e,
                tags=["strategy", "error", "long"],
            )

    def close_trades(self) -> None:
        """Close the current open position."""
        if not self.position:
            self.logger.warning("Attempted to close trades but no position is open.")
            return

        is_short = self.position.is_short
        position_type = "SHORT" if is_short else "LONG"
        self.logger.info(
            f"CLOSE {position_type} Signal",
            data={
                **self.current_data.__dict__,
                "candles_in_position": self.candles_since_order,
                # Add P/L if available/calculable here
            },
            tags=["strategy", "signal", "close", position_type.lower()],
        )
        try:
            self.position.close()
            self.pending_order_type = "CLOSE"
            self.pending_order_data = self.current_data
            self.candles_since_order = 0
            # self.last_order_data = self.current_data # debatable if close constitutes a new "last order"
            self.logger.info(
                f"CLOSE {position_type} order placed",
                data=self.current_data.__dict__,
                tags=["strategy", "order", "close", position_type.lower()],
            )
        except Exception as e:
            self.logger.error(
                f"Failed to place CLOSE {position_type} order",
                exception=e,
                tags=["strategy", "error", "close", position_type.lower()],
            )
