import abc
from dataclasses import dataclass
import uuid
from backtesting import Strategy
from typing import Literal, Optional
import pandas as pd
from .logger import ConsoleLogger, TradingLogger


@dataclass
class CurrentData:
    price: float
    time: str
    # other indicators and important data


class BaseStrategy(Strategy, metaclass=abc.ABCMeta):
    """Base class for trading strategies."""

    current_data: CurrentData
    last_order_data: Optional[CurrentData]
    size: float
    pending_order_type: Optional[Literal["LONG", "SHORT", "CLOSE"]]
    pending_order_data: Optional[CurrentData]
    candles_since_order: int

    def _create_logger(self, session_id: str) -> TradingLogger:
        """Get the logger for the strategy."""
        return TradingLogger(
            session_id=session_id,
            loggers=[ConsoleLogger(session_id=session_id)],
        )

    def init(self):
        # Initialize logger
        self.logger = self._create_logger(session_id=str(uuid.uuid4()))

        # Initialize data series
        close_series = pd.Series(self.data.Close)
        # Initialize indicators
        # Initialize other variables

    def next(self):
        self.pre_trading()
        self.trading()
        self.post_trading()

    def pre_trading(self):
        """Get current data, indicators, signals, market context, everything that is needed for trading signal creation"""
        self.current_data = CurrentData(
            price=self.data.Close[-1],
            time=str(self.data.index[-1]),
            # other indicators and important data
        )

    @abc.abstractmethod
    def has_position(self):
        """Check if there is a position"""
        pass

    @abc.abstractmethod
    def should_stop_trading(self):
        """Check if the trading needs to be stopped
        - Use for risk management protection
        """
        pass

    @abc.abstractmethod
    def trading(self):
        """Trading logic for enter and exit positions
        - Mathetmatical expression to get trading signal
        - Usage of trade_signal method to place orders
        - Use of close_trades method to close positions
        - Use of should_stop_trading method to stop trading
        """
        pass

    @abc.abstractmethod
    def post_trading(self):
        """Update trade counters, other variables and log position confirmations"""
        pass

    def trade_signal(self, type: Literal["LONG", "SHORT"]):
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

    def do_short(self):
        try:
            self.sell(size=self.size, limit=self.current_data.price)
            self.pending_order_type = "SHORT"
            self.pending_order_data = self.current_data
            self.candles_since_order = 0
            self.last_order_data = self.current_data
            self.logger.info(
                "SHORT order placed",
                data={
                    **self.current_data.__dict__,
                },
                tags=["strategy", "order", "short"],
            )
        except Exception as e:
            self.logger.error(
                "Failed to place SHORT order",
                exception=e,
                tags=["strategy", "error", "short"],
            )

    def do_long(self):
        try:
            self.buy(size=self.size, limit=self.current_data.price)
            self.pending_order_type = "LONG"
            self.pending_order_data = self.current_data
            self.candles_since_order = 0
            self.last_order_data = self.current_data
            self.logger.info(
                "LONG order placed",
                data={
                    **self.current_data.__dict__,
                },
                tags=["strategy", "order", "long"],
            )
        except Exception as e:
            self.logger.error(
                "Failed to place LONG order",
                exception=e,
                tags=["strategy", "error", "long"],
            )

    def close_trades(self):
        is_short = self.position.is_short
        type = "SHORT" if is_short else "LONG"
        self.logger.info(
            f"CLOSE {type} Position",
            data={
                **self.current_data.__dict__,
                "candles_in_position": self.candles_since_order,
            },
            tags=["strategy", "signal", "close", type.lower()],
        )
        try:
            self.position.close()
            self.pending_order_type = "CLOSE"
            self.pending_order_data = self.current_data
            self.candles_since_order = 0
            self.last_order_data = self.current_data
            self.logger.info(
                f"CLOSE {type} order placed",
                data=self.current_data.__dict__,
                tags=["strategy", "order", "close", type.lower()],
            )
        except Exception as e:
            self.logger.error(
                f"Failed to place CLOSE {type} order",
                exception=e,
                tags=["strategy", "error", "close", type.lower()],
            )
