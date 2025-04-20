import sqlite3
import json
from typing import List, Optional, Dict, Any, Union, TypedDict, cast
from datetime import datetime, timezone
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, validator


class TradingStrategyDefinition(BaseModel):
    is_strategy: bool = Field(
        default=False,
        description="Whether the user is describing a trading strategy in a way, that it's clear what kinf of strategy is this, and majority of fields are filled",
    )
    strategy_name: Optional[str] = Field(
        default=None,
        description="Concise and descriptive name for the automated trading strategy (2-5 words).",
    )
    strategy_type: Optional[
        Literal[
            "Trend-following",
            "Momentum-based",
            "Reversal",
            "Breakout",
            "Mean Reversion",
            "Volatility-based",
            "Event-driven",
            "Time-based",
            "Hybrid",
        ]
    ] = Field(
        default=None,
        description="The core type of automated trading strategy chosen by the user.",
    )
    assistant_response_summary: Optional[str] = Field(
        description="Detailed summary explaining the assistant's responses and decisions, suitable for the user."
    )
    assistant_reasoning: Optional[List[str]] = Field(
        default_factory=list,
        description="Chronological list of reasoning steps by the assistant, updated with each interaction.",
    )
    trading_idea: Optional[str] = Field(
        description="Detailed explanation of the user's automated trading idea, including rationale and expected edge."
    )
    indicators_and_signals: Optional[List[str]] = Field(
        description="Detailed list of indicators or signals, their calculation, logic, parameters, and their role in the strategy."
    )
    entry_conditions: Optional[List[str]] = Field(
        description="Comprehensive description of entry conditions including indicator thresholds, patterns, and exact logic for automation."
    )
    exit_conditions: Optional[List[str]] = Field(
        description="Detailed explanation of automated exit conditions (take-profit, stop-loss, trailing stop) with risk management logic."
    )
    position_sizing: Optional[str] = Field(
        description="Explicit description of position sizing or money management strategy (fixed size, percentage, volatility-based, etc.)."
    )
    risk_management_rules: Optional[List[str]] = Field(
        description="Specific rules and procedures for managing risk, drawdowns, maximum daily losses, etc."
    )
    markets_and_timeframes: Optional[List[str]] = Field(
        description="Clearly defined markets (stocks, crypto, forex), symbols, exchanges, and precise timeframes or sessions targeted."
    )
    order_types: Optional[List[str]] = Field(
        description="Types of orders used in the strategy (Market, Limit, Stop, Stop-Limit, etc.) and the reasoning behind each type."
    )
    additional_info: Optional[str] = Field(
        description="Additional information about the strategy that is not covered by the other fields."
    )
    questions_about_strategy: Optional[List[str]] = Field(
        description="AI assistant questions about the strategy"
    )
    search_queries: Optional[List[str]] = Field(
        description="Search queries for the strategy"
    )
    source_urls: Optional[List[str]] = Field(description="Source URLs for the strategy")

    # Add validators for list fields to handle string-to-list conversion
    @validator(
        "assistant_reasoning",
        "indicators_and_signals",
        "entry_conditions",
        "exit_conditions",
        "risk_management_rules",
        "markets_and_timeframes",
        "order_types",
        "search_queries",
        "source_urls",
        "questions_about_strategy",
        pre=True,
    )
    def convert_string_to_list(cls, value):
        """Convert a string to a list of strings if provided as a string."""
        if isinstance(value, str):
            # If the string is empty, return an empty list
            if not value.strip():
                return []

            # If the string already looks like a list representation, try to parse it
            if value.startswith("[") and value.endswith("]"):
                try:
                    import ast

                    parsed_value = ast.literal_eval(value)
                    if isinstance(parsed_value, list):
                        return parsed_value
                except (SyntaxError, ValueError):
                    pass

            # Otherwise, treat the string as a single item
            return [value]
        return value

    class Config:
        """Configuration for the Pydantic model."""

        validate_assignment = True
        extra = "ignore"  # Ignore extra fields

    def context_str(self):
        markdown = ""

        if self.strategy_name:
            markdown += f"{self.strategy_name}Strategy type: ({self.strategy_type})\n\n"

        if self.trading_idea:
            markdown += f"Trading Idea\n{self.trading_idea}\n\n"

        if self.indicators_and_signals and len(self.indicators_and_signals) > 0:
            markdown += "Trading Indicators and Signals\n"
            for indicator in self.indicators_and_signals:
                markdown += f"- {indicator}\n"
            markdown += "\n"

        if self.entry_conditions and len(self.entry_conditions) > 0:
            markdown += "Entry Conditions\n"
            for condition in self.entry_conditions:
                markdown += f"* {condition}\n"
            markdown += "\n"

        if self.exit_conditions and len(self.exit_conditions) > 0:
            markdown += "Exit Conditions\n"
            for condition in self.exit_conditions:
                markdown += f"* {condition}\n"
            markdown += "\n"

        if self.markets_and_timeframes and len(self.markets_and_timeframes) > 0:
            markdown += "Target Markets\n"
            for market in self.markets_and_timeframes:
                markdown += f"- {market}\n"
            markdown += "\n"

        if self.order_types and len(self.order_types) > 0:
            markdown += "Order Types\n"
            for order_type in self.order_types:
                markdown += f"- {order_type}\n"
            markdown += "\n"

        if self.risk_management_rules and len(self.risk_management_rules) > 0:
            markdown += "Risk Management Rules\n"
            for risk_management_rule in self.risk_management_rules:
                markdown += f"- {risk_management_rule}\n"
            markdown += "\n"

        if self.position_sizing and len(self.position_sizing) > 0:
            markdown += "Position Sizing\n"
            markdown += f"{self.position_sizing}\n"
            markdown += "\n"

        return markdown


class StrategyType(TypedDict):
    id: Optional[int]
    message_id: str
    timestamp: str
    flags: int
    reactions: int
    content: str
    strategy: TradingStrategyDefinition


class StrategyDb:
    def __init__(self, db_path: str = "data/messages.db"):
        self.conn = sqlite3.connect(db_path)

    def get_strategy(self, strategy_id: int) -> Optional[StrategyType]:
        """
        Get a specific strategy by ID.
        Returns None if not found.
        """
        cursor = self.conn.execute(
            """
            SELECT id, message_id, timestamp, flags, reactions, content, strategy_json 
            FROM discord_strategies
            WHERE id = ?
            """,
            (strategy_id,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return {
            "id": row[0],
            "message_id": row[1],
            "timestamp": row[2],
            "flags": row[3],
            "reactions": row[4],
            "content": row[5],
            "strategy": TradingStrategyDefinition(**json.loads(row[6])),
        }

    def list_strategies(self, limit: int = 100, offset: int = 0) -> List[StrategyType]:
        """
        List strategies with pagination.
        """
        cursor = self.conn.execute(
            """
            SELECT id, message_id, timestamp, flags, reactions, content, strategy_json 
            FROM discord_strategies
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )

        strategies = []
        for row in cursor:
            strategies.append(
                {
                    "id": row[0],
                    "message_id": row[1],
                    "timestamp": row[2],
                    "flags": row[3],
                    "reactions": row[4],
                    "content": row[5],
                    "strategy": TradingStrategyDefinition(**json.loads(row[6])),
                }
            )

        return strategies

    def list_strategies_by_ids(self, ids: List[int]) -> List[StrategyType]:
        """
        List strategies that match the given IDs.

        Args:
            ids: List of strategy IDs to retrieve

        Returns:
            List of matching strategies
        """
        if not ids:
            return []

        placeholders = ",".join("?" * len(ids))
        query = f"""
            SELECT id, message_id, timestamp, flags, reactions, content, strategy_json 
            FROM discord_strategies 
            WHERE id IN ({placeholders})
            ORDER BY timestamp DESC
        """

        cursor = self.conn.execute(query, ids)

        return [
            {
                "id": row[0],
                "message_id": row[1],
                "timestamp": row[2],
                "flags": row[3],
                "reactions": row[4],
                "content": row[5],
                "strategy": TradingStrategyDefinition(**json.loads(row[6])),
            }
            for row in cursor
        ]

    def get_strag(self, strat_id: int) -> Optional[StrategyType]:
        """
        Shorthand method for get_strategy.

        Args:
            strat_id: ID of the strategy to retrieve

        Returns:
            Strategy if found, None otherwise
        """
        return self.get_strategy(strat_id)

    def get_strag_by_msg(self, message_id: str) -> Optional[StrategyType]:
        """
        Get a strategy by its associated message ID.

        Args:
            message_id: The message ID to look up

        Returns:
            Strategy if found, None otherwise
        """
        cursor = self.conn.execute(
            """
            SELECT id, message_id, timestamp, flags, reactions, content, strategy_json 
            FROM discord_strategies
            WHERE message_id = ?
            LIMIT 1
            """,
            (message_id,),
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return {
            "id": row[0],
            "message_id": row[1],
            "timestamp": row[2],
            "flags": row[3],
            "reactions": row[4],
            "content": row[5],
            "strategy": TradingStrategyDefinition(**json.loads(row[6])),
        }

    def strategy_count(self) -> int:
        """
        Return the total number of strategies in the database.
        """
        cursor = self.conn.execute("SELECT COUNT(*) FROM discord_strategies")
        return cursor.fetchone()[0]

    def delete_strategy(self, strategy_id: int) -> bool:
        """
        Delete a strategy from the database.

        Args:
            strategy_id: ID of the strategy to delete

        Returns:
            True if strategy was deleted, False otherwise
        """
        with self.conn:
            cursor = self.conn.execute(
                "DELETE FROM discord_strategies WHERE id = ?", (strategy_id,)
            )
            return cursor.rowcount > 0

    def close(self):
        self.conn.close()
