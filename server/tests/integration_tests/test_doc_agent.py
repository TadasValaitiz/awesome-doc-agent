import os
import pytest
from typing import Any, Dict
from src.doc_agent import graph
from src.doc_agent.state import InputState
from src.doc_agent.utils import panda_sample
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@pytest.fixture(scope="function")
def sample_input() -> InputState:
    """Fixture that provides a sample input state for testing."""
    return InputState(original=panda_sample())


@pytest.mark.asyncio
async def test_doc_agent_simple_runthrough(sample_input: InputState) -> None:
    """Test basic doc agent graph execution with sample input."""
    # Run the graph with sample input
    events = []
    async for type, event in graph.astream(
        input=sample_input,
        config={"configurable": {"model": "gpt-4"}},
        stream_mode=["values"],
    ):
        events.append((type, event))

    # Basic assertions
    assert len(events) > 0, "Should have received at least one event"

    # Check that we have the expected event types
    event_types = [t for t, _ in events]
    assert "values" in event_types, "Should have values in the events"

    # Check that we have the expected analysis results
    has_missing_values = any(
        "missing_values" in event.get("update", {}) for _, event in events
    )
    has_outliers = any("outliers" in event.get("update", {}) for _, event in events)
    has_inconsistencies = any(
        "inconsistencies" in event.get("update", {}) for _, event in events
    )
    has_duplicates = any("duplicates" in event.get("update", {}) for _, event in events)

    assert (
        has_missing_values or has_outliers or has_inconsistencies or has_duplicates
    ), "Should have at least one type of analysis result"


@pytest.mark.asyncio
async def test_doc_agent_invoke(sample_input: InputState) -> None:
    """Test doc agent graph using ainvoke method."""
    result = await graph.ainvoke(
        input=sample_input,
        config={"configurable": {"model": "gpt-4"}},
    )

    # Basic assertions
    assert result is not None, "Result should not be None"
    assert isinstance(result, dict), "Result should be a dictionary"

    # Check that we have the expected analysis results
    has_missing_values = "missing_values" in result
    has_outliers = "outliers" in result
    has_inconsistencies = "inconsistencies" in result
    has_duplicates = "duplicates" in result

    assert (
        has_missing_values or has_outliers or has_inconsistencies or has_duplicates
    ), "Should have at least one type of analysis result"
