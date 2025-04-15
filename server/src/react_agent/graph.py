"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
import functools
from typing import Dict, List, Literal, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    BaseMessageChunk,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model
from langgraph.types import Command, interrupt
from langgraph.graph import StateGraph, START, END

# Define the function that calls the model


async def call_model(state: State, config: RunnableConfig):
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_runnable_config(config)

    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=timezone.utc).isoformat()
    )

    chunks = []
    async for chunk in model.astream(
        [{"role": "system", "content": system_message}, *state.messages], config
    ):
        chunks.append(chunk)
        yield Command(update={"messages": [chunk]})

    final_message = functools.reduce(
        lambda acc, chunk: acc.__add__(chunk),
        chunks,
        BaseMessageChunk(content="", type=""),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if (
        state.is_last_step
        and final_message.additional_kwargs.get("tool_calls") is not None
    ):
        yield Command(
            update={
                "messages": [
                    AIMessage(
                        id=final_message.id,
                        content="Sorry, I could not find an answer to your question in the specified number of steps.",
                    )
                ]
            }
        )
    else:
        yield Command(
            update={
                "messages": [
                    AIMessage(
                        id=final_message.id,
                        content=final_message.content,
                    )
                ]
            }
        )


# Define a new graph

builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the two nodes we will cycle between
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge(START, "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
# You can customize this by adding interrupt points for state updates
graph = builder.compile(
    interrupt_before=[],  # Add node names here to update state before they're called
    interrupt_after=[],  # Add node names here to update state after they're called
)
graph.name = "ReActAgent"  # This customizes the name in LangSmith
