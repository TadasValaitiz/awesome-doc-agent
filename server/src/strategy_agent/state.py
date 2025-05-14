"""Define the state structures for the agent."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Dict, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep, RemainingSteps
import pandas as pd
from typing_extensions import Annotated
from pydantic import BaseModel, Field
class BacktestResults(TypedDict):
    timestamp: str
    best_parameters: dict[str, float]
    best_results: OptimizerResults
    static_params: dict[str, float]

class OptimizerResults(TypedDict):
    total_return: float  # Return [%]
    buy_hold_return: float  # Buy & Hold Return [%]
    sharpe_ratio: float  # Sharpe Ratio
    sortino_ratio: float  # Sortino Ratio
    max_drawdown: float  # Max. Drawdown [%]
    avg_drawdown: float  # Avg. Drawdown [%]
    trades: int  # # Trades
    win_rate: float  # Win Rate [%]
    avg_trade: float  # Avg. Trade [%]
    best_trade: float  # Best Trade [%]
    worst_trade: float  # Worst Trade [%]
    exposure_time: float  # Exposure Time [%]
    equity_final: float  # Equity Final [$]
    equity_peak: float  # Equity Peak [$]
    volatility_ann: float  # Volatility (Ann.) [%]
    cagr: float  # CAGR [%]
    calmar_ratio: float  # Calmar Ratio
    profit_factor: float  # Profit Factor
    expectancy: float  # Expectancy [%]
    sqn: float  # SQN
    kelly_criterion: float  # Kelly Criterion




class SerializableDataFrame(BaseModel):
    """A wrapper class for pandas DataFrames that provides serialization/deserialization.

    This allows pandas DataFrames to be properly serialized/deserialized in LangGraph state.
    """

    cached_df: Optional[pd.DataFrame] = Field(default=None, exclude=True)
    serialized_data: Optional[str] = None

    model_config = {
        "arbitrary_types_allowed": True,
    }

    def __init__(self, df: Optional[pd.DataFrame] = None, **data: Any):
        super().__init__(**data)
        if df is not None:
            self.df = df

    @property
    def df(self) -> Optional[pd.DataFrame]:
        """Get the DataFrame."""
        if self.cached_df is not None:
            return self.cached_df
        if self.serialized_data:
            self.cached_df = self._deserialize_dataframe(self.serialized_data)
            return self.cached_df
        return None

    @df.setter
    def df(self, df: pd.DataFrame) -> None:
        """Set the DataFrame and update the serialized data."""
        self.cached_df = df
        self.serialized_data = self._serialize_dataframe(df)

    def _serialize_dataframe(self, df: pd.DataFrame) -> str:
        """Serialize a pandas DataFrame to a base64-encoded string."""
        buffer = io.BytesIO()
        df.to_parquet(buffer, compression="gzip")
        return base64.b64encode(buffer.getvalue()).decode()

    def _deserialize_dataframe(self, serialized: str) -> pd.DataFrame:
        """Deserialize a base64-encoded string to a pandas DataFrame."""
        buffer = io.BytesIO(base64.b64decode(serialized))
        return pd.read_parquet(buffer)

    async def serialize_async(self, df: pd.DataFrame) -> str:
        """Asynchronously serialize a pandas DataFrame to a base64-encoded string."""
        # We'll run the CPU-bound serialization in a thread pool
        import asyncio

        return await asyncio.to_thread(self._serialize_dataframe, df)

    async def deserialize_async(self, serialized: str) -> pd.DataFrame:
        """Asynchronously deserialize a base64-encoded string to a pandas DataFrame."""
        # We'll run the CPU-bound deserialization in a thread pool
        import asyncio

        return await asyncio.to_thread(self._deserialize_dataframe, serialized)

    @classmethod
    async def create_async(cls, df: pd.DataFrame) -> "SerializableDataFrame":
        """Asynchronously create a SerializableDataFrame from a pandas DataFrame."""
        instance = cls()
        instance.cached_df = df
        instance.serialized_data = await instance.serialize_async(df)
        return instance

    async def get_df_async(self) -> Optional[pd.DataFrame]:
        """Asynchronously get the DataFrame."""
        if self.cached_df is not None:
            return self.cached_df
        if self.serialized_data:
            self.cached_df = await self.deserialize_async(self.serialized_data)
            return self.cached_df
        return None

    def model_dump(self, **kwargs):
        """Convert to dictionary for serialization."""
        return {"serialized_data": self.serialized_data}


@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )


@dataclass
class OptimizationArtifacts:
    """Represents the optimization artifacts for a strategy."""

    results: Optional[BacktestResults] = field(default=None)
    plots: Optional[dict[str, str]] = field(default=None)
    logs: Optional[SerializableDataFrame] = field(default=None)

@dataclass
class StrategyAgentState(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """

    is_last_step: IsLastStep = field(default=False)
    remaining_steps: RemainingSteps = field(default=0)
    strategy_feedback: Optional[str] = field(default=None)
    strategy_approved: bool = field(default=False)
    strategy_message_index: int = field(default=0)
    code_feedback: Optional[str] = field(default=None)
    code_approved: bool = field(default=False)
    code_output: Optional[str] = field(default=None)
    judge_skip: bool = field(default=False)
    optimization_symbols: list[str] = field(default_factory=lambda: ["BTCUSD"])
    optimization_timeframes: list[str] = field(default_factory=lambda: ["5"])
    optimization_months: int = field(default=6)
    optimization_feedback: Optional[str] = field(default=None)
    optimization_step_message: Optional[str] = field(default=None)
    optimization_artifacts: Optional[OptimizationArtifacts] = field(default=None)
    optimization_step: Literal[
        "none",
        "loading_data",
        "data_loaded",
        "backtesting",
        "backtesting_finished",
        "backtesting_failed",
        "backtesting_poor_results",
    ] = field(default="none")
    optimization_data: Dict[str, Dict[str, SerializableDataFrame]] = field(
        default_factory=dict
    )
