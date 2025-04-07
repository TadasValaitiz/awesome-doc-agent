from dataclasses import dataclass
from pandas import DataFrame
import base64
import io
from io import StringIO
from typing import Dict, Any, List, Literal, Optional, TypedDict
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage


@dataclass
class DocumentMetadata:

    def __init__(self, file_name: str, df: DataFrame):
        self.file_name = file_name
        self.df = df
        self.columns = df.columns
        self.num_rows = len(df)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DocumentMetadata to a dictionary format."""
        serialized_df = serialize_dataframe(self.df)
        return {
            "file_name": self.file_name,
            "serialized_df": serialized_df,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        if not all(key in data for key in ["file_name", "serialized_df"]):
            raise ValueError("Missing required fields in data dictionary")

        try:
            return cls(
                file_name=data["file_name"],
                df=deserialize_dataframe(data["serialized_df"]),
            )
        except Exception as e:
            raise ValueError(f"Error deserializing DataFrame: {str(e)}")


def serialize_dataframe(df):
    buffer = io.BytesIO()
    df.to_parquet(buffer, compression="gzip")
    return base64.b64encode(buffer.getvalue()).decode()


def deserialize_dataframe(serialized):
    buffer = io.BytesIO(base64.b64decode(serialized))
    return pd.read_parquet(buffer)


class FieldPosition(TypedDict):
    row_index: int
    column_index: int


class BaseValidatedModel(BaseModel):
    validation_error: Optional[str] = Field(
        default=None,
        exclude=True,
    )


class MissingValues(BaseValidatedModel):
    missing_values: Optional[List[tuple[FieldPosition, str]]] = Field(
        default=None,
    )


class Outliers(BaseValidatedModel):
    outliers_reasoning: Optional[str] = Field(
        default=None,
    )
    outliers_boundaries: Optional[dict] = Field(
        default=None,
    )
    outliers: Optional[List[tuple[FieldPosition, str]]] = Field(
        default=None,
    )


class Duplicates(BaseValidatedModel):
    duplicates: Optional[List[tuple[List[FieldPosition], str]]] = Field(
        default=None,
    )


class Inconsistencies(BaseValidatedModel):
    inconsistent_format_reasoning: Optional[str] = Field(
        default=None,
    )
    warnings: Optional[List[tuple[FieldPosition, str]]] = Field(
        default=None,
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
    )


class OutlierFixes(BaseValidatedModel):
    fixes: Optional[List[ValueFix]] = Field(
        default=None,
    )


class DuplicateFixes(BaseValidatedModel):
    rows_to_drop: Optional[List[int]] = Field(
        default=None,
    )
    reason: Optional[str] = Field(default=None)


class InconsistencyFixes(BaseValidatedModel):
    fixes: Optional[List[ValueFix]] = Field(default=None)


class Summary(BaseValidatedModel):
    summary: str


class State(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    file_name: str = Field(default="")
    current_node: str = Field(default="init")
    supervisor_node: str = Field(default="init")
    next_node: str = Field(default="init")
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
    user_intent: Literal[
        "analyze_all",
        "fix_duplicates",
        "fix_missing_values",
        "fix_outliers",
        "fix_inconsistencies",
        "chat",
    ] = Field(default="analyze_all")

    def current_node_updates(
        self,
    ) -> Optional[
        Inconsistencies
        | Duplicates
        | Outliers
        | MissingValues
        | MissingValueFixes
        | OutlierFixes
        | DuplicateFixes
        | InconsistencyFixes
        | Summary
    ]:
        match self.current_node:
            case "analyze_inconsistencies":
                return self.inconsistencies
            case "analyze_duplicates":
                return self.duplicates
            case "analyze_outliers":
                return self.outliers
            case "analyze_missing_values":
                return self.missing_values
            case "fix_missing_values":
                return self.missing_value_fixes
            case "fix_outliers":
                return self.outlier_fixes
            case "fix_duplicates":
                return self.duplicate_fixes
            case "fix_inconsistencies":
                return self.inconsistency_fixes
            case "summarize_analysis":
                return self.analysis_summary
            case "summarize_fixes":
                return self.fixes_summary
            case _:
                return None


class StateUpdates(BaseModel):
    load_document: Optional[State] = Field(default=None)
    start: Optional[State] = Field(default=None, alias="__start__")
    end: Optional[State] = Field(default=None, alias="__end__")
    planner: Optional[State] = Field(default=None)
    analyzer: Optional[State] = Field(default=None)
    fixer: Optional[State] = Field(default=None)
    chat_message: Optional[State] = Field(default=None)
    analyze_missing_values: Optional[State] = Field(default=None)
    analyze_outliers: Optional[State] = Field(default=None)
    analyze_duplicates: Optional[State] = Field(default=None)
    analyze_inconsistencies: Optional[State] = Field(default=None)
    fix_missing_values: Optional[State] = Field(default=None)
    fix_outliers: Optional[State] = Field(default=None)
    fix_duplicates: Optional[State] = Field(default=None)
    fix_inconsistencies: Optional[State] = Field(default=None)
    summarize_analysis: Optional[State] = Field(default=None)
    summarize_fixes: Optional[State] = Field(default=None)
    current_node: str = Field(default="init")

    def current_node_state(self) -> Optional[State]:
        state = getattr(self, self.current_node)
        if state is not None and isinstance(state, State):
            return state
        return None

    def current_node_state_updates(self):
        state = self.current_node_state()
        if state is not None:
            return state.current_node_updates()
        return None

    @classmethod
    def from_update(cls, update_dict: Dict[str, Any]) -> "StateUpdates":
        """Create a StateUpdates instance from a dictionary update.

        Args:
            update_dict: Dictionary containing update information

        Returns:
            StateUpdates instance with current_node set based on the update
        """
        # Initialize with default values
        updates = cls()

        # Map special names to their attribute equivalents
        field_mapping = {"__start__": "start", "__end__": "end"}

        # Find the first key that has an update and set it as current_node
        for key, value in update_dict.items():
            # Skip current_node and handle special names
            if key == "current_node":
                continue

            # Map special names to their normal field names
            field_name = field_mapping.get(key, key)

            updates.current_node = field_name
            if field_name in cls.model_fields:
                setattr(updates, field_name, State(**value) if value else None)

        return updates
