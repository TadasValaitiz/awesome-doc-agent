"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep, RemainingSteps
from typing_extensions import Annotated


@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )


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
