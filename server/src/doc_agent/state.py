"""State definitions.

State is the interface between the graph and end user as well as the
data model used internally by the graph.
"""

from dataclasses import field
import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, Callable, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pandas import DataFrame
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


class FieldPosition(TypedDict):
    row_index: int
    column_index: int


class BaseValidatedModel(BaseModel):
    """Base class for models that need validation from LLM output."""

    validation_error: Optional[str] = Field(
        default=None,
        description="Internal field to track validation errors",
    )

    @classmethod
    def from_llm_output(cls, output_str: str) -> "BaseValidatedModel":
        """Creates a model instance from LLM output, catching validation errors."""
        parser = PydanticOutputParser(pydantic_object=cls)
        try:
            return parser.parse(output_str)
        except Exception as e:
            instance = cls()
            instance.validation_error = str(e)
            return instance

    def to_markdown(self) -> str:
        """Convert the model to a markdown representation for LLM consumption."""
        if self.validation_error:
            return f"**Error**: {self.validation_error}"

        lines = [f"## {self.__class__.__name__}"]

        for field_name, field_value in self.model_dump().items():
            if field_name == "validation_error" or field_value is None:
                continue

            if isinstance(field_value, list) and field_value:
                lines.append(f"\n### {field_name}")
                for i, item in enumerate(field_value):
                    if isinstance(item, dict):
                        lines.append(f"\n#### Item {i+1}")
                        for key, val in item.items():
                            lines.append(f"- **{key}**: {val}")
                    else:
                        lines.append(f"- {item}")
            else:
                lines.append(f"\n**{field_name}**: {field_value}")

        return "\n".join(lines)


class MissingValues(BaseValidatedModel):
    missing_values: Optional[List[tuple[FieldPosition, str]]] = Field(
        default=None,
        description="List of missing values where first element is FieldPosition object, second element is reason for missing values",
    )


class Outliers(BaseValidatedModel):
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


class Duplicates(BaseValidatedModel):
    duplicates: Optional[List[tuple[List[FieldPosition], str]]] = Field(
        default=None,
        description="List of duplicates where first element is List of FieldPosition objects, second element is description ",
    )


class Inconsistencies(BaseValidatedModel):
    inconsistent_format_reasoning: Optional[str] = Field(
        default=None,
        description="Reasoning for inconsistent format, calculation logic, that is used to detect inconsistent format",
    )
    warnings: Optional[List[tuple[FieldPosition, str]]] = Field(
        default=None,
        description="List of warnings where first element is FieldPosition object, second element is warning message",
    )


class ValueFix(TypedDict):
    row_index: int
    column_index: int
    original_value: Any
    fixed_value: Any
    reason: str

class MissingValueFixes(BaseValidatedModel):
    fixes: Optional[List[ValueFix]] = Field(
        default=None,
        description="List of fixes for missing values including the position, original and new values",
    )


class OutlierFixes(BaseValidatedModel):
    fixes: Optional[List[ValueFix]] = Field(
        default=None,
        description="List of fixes for outliers including the position, original and new values",
    )


class DuplicateFixes(BaseValidatedModel):
    rows_to_drop: Optional[List[int]] = Field(
        default=None, description="List of row indices to drop as they are duplicates"
    )
    reason: Optional[str] = Field(
        default=None, description="Explanation for which rows were kept vs dropped"
    )


class InconsistencyFixes(BaseValidatedModel):
    fixes: Optional[List[ValueFix]] = Field(
        default=None,
        description="List of fixes for inconsistent values including the position, original and new values",
    )


class Summary(BaseValidatedModel):
    """Summary of data quality analysis results."""

    summary: str = Field(
        description="Comprehensive summary of data quality issues found in the dataset"
    )



class InputState(BaseModel):
    """Input state defines the interface between the graph and the user (external API)."""

    model_config = {"arbitrary_types_allowed": True}
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    current_node: str = Field(default="__start__")
    user_intent: Literal[
        "analyze_all",
        "fix_duplicates",
        "fix_missing_values",
        "fix_outliers",
        "fix_inconsistencies",
        "chat",
    ] = Field(default="analyze_all")


class State(InputState):
    file_name: str = Field(default="")
    supervisor_node: str = Field(default="init")
    next_node: str = Field(default="init")
    original: Optional[DataFrame] = Field(default=None)
    loop_step: Annotated[int, operator.add] = Field(default=0)
    missing_values: Optional[MissingValues] = Field(default=None)
    outliers: Optional[Outliers] = Field(default=None)
    duplicates: Optional[Duplicates] = Field(default=None)
    inconsistencies: Optional[Inconsistencies] = Field(default=None)
    missing_value_fixes: Optional[MissingValueFixes] = Field(default=None)
    outlier_fixes: Optional[OutlierFixes] = Field(default=None)
    duplicate_fixes: Optional[DuplicateFixes] = Field(default=None)
    inconsistency_fixes: Optional[InconsistencyFixes] = Field(default=None)
    analysis_summary: Optional[Summary] = Field(default=None)
    fixes_summary: Optional[Summary] = Field(default=None)

