"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
import functools
from typing import Dict, List, Literal, Optional, Type, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    BaseMessageChunk,
)
from langchain_core.runnables import RunnableConfig, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
import pandas as pd

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from strategy_agent.configuration import Configuration
from strategy_agent.state import InputState, State
from strategy_agent.tools import TOOLS
from strategy_agent.utils import init_model


class StrategyAssistantNode:
    """Abstract base class for analysis nodes that follow a common pattern."""

    def __init__(self, node_name: str):
        self.node_name = node_name

    def get_prompt_template(self) -> str:
        """Return the prompt template to use for this analysis."""
        raise NotImplementedError

    def get_output_model(self) -> Type[BaseModel]:
        """Return the Pydantic model to parse the output into.

        The returned model must have a `from_llm_output` class method.
        """
        raise NotImplementedError

    def get_state_key(self) -> str:
        """Return the key in the state to update with the result."""
        raise NotImplementedError

    async def __call__(self, state: State, *, config: Optional[RunnableConfig] = None):
        """Execute the analysis node with the common pattern."""
        configuration = Configuration.from_runnable_config(config)

        model = init_model(config).bind_tools(TOOLS)

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

create_strategy_node = StrategyAssistantNode("strategy_assistant")

builder = StateGraph(
    state_schema=State,
    input=InputState,
    config_schema=Configuration,
)

builder.add_node(create_strategy_node)
builder.add_node("tools", ToolNode(TOOLS))


builder.add_edge(START, "strategy_assistant")
builder.add_conditional_edges(
    "strategy_assistant",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)
graph = builder.compile(
    interrupt_before=[],
    interrupt_after=[],
)
graph.name = "StrategyAgent"
