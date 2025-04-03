"""State definitions.

State is the interface between the graph and end user as well as the
data model used internally by the graph.
"""

from dataclasses import field
import operator
from typing import Annotated, Any, Dict, List, Optional, Callable

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pandas import DataFrame
from pydantic import BaseModel, Field


class FieldPosition(BaseModel):
    row_index: str = Field(description="Row Index of the field")
    column_index: str = Field(description="Column Index of the field")


class MissingValues(BaseModel):
    missing_values: Optional[List[tuple[FieldPosition, str]]] = Field(
        default=None,
        description="List of missing values where first element is FieldPosition object, second element is reason for missing values",
    )


class Outliers(BaseModel):
    outliers_reasoning: Optional[str] = Field(
        default=None,
        description="Reasoning for outliers, calculation logic, that is used to detect outliers",
    )
    outliers_boundaries: Optional[dict] = Field(
        default=None,
        description="Outliers boundaries found in dataset, that will be used to detect outliers",
    )
    outliers: Optional[List[tuple[FieldPosition, str]]] = Field(
        default=None,
        description="List of outliers where first element is FieldPosition object, second element is description for outliers",
    )


class Duplicates(BaseModel):
    duplicates: Optional[List[tuple[List[FieldPosition], str]]] = Field(
        default=None,
        description="List of duplicates where first element is List of FieldPosition objects, second element is description ",
    )


class Inconsistencies(BaseModel):
    inconsistent_format_reasoning: Optional[str] = Field(
        default=None,
        description="Reasoning for inconsistent format, calculation logic, that is used to detect inconsistent format",
    )
    warnings: Optional[List[tuple[FieldPosition, str]]] = Field(
        default=None,
        description="List of warnings where first element is FieldPosition object, second element is warning message",
    )


class InputState(BaseModel):
    """Input state defines the interface between the graph and the user (external API)."""

    model_config = {"arbitrary_types_allowed": True}

    original: DataFrame


class State(InputState):
    current_node: str = Field(default="init")
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    loop_step: Annotated[int, operator.add] = Field(default=0)
    missing_values: Optional[MissingValues] = Field(default=None)
    outliers: Optional[Outliers] = Field(default=None)
    duplicates: Optional[Duplicates] = Field(default=None)
    inconsistencies: Optional[Inconsistencies] = Field(default=None)

    def with_current_node(self, node_name: str) -> "State":
        return State(
            original=self.original,
            current_node=node_name,
            messages=self.messages,
            loop_step=self.loop_step,
        )


class OutputState(BaseModel):
    info: dict[str, Any]
