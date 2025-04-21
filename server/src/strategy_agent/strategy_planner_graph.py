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
from strategy_agent.state import InputState, StrategyAgentState
from strategy_agent.tools import TOOLS
from strategy_agent.utils import init_model


class StrategyAssistantNode:
    """Abstract base class for analysis nodes that follow a common pattern."""

    def __init__(self, node_name: str):
        self.node_name = node_name

    async def __call__(self, state: StrategyAgentState, *, config: Optional[RunnableConfig] = None):
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
            chunks[1:],
            chunks[0],
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
                            additional_kwargs=final_message.additional_kwargs,
                        )
                    ]
                }
            )


def route_model_output(state: StrategyAgentState) -> Literal["__end__", "tools"]:
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

    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


def create_strategy_planner():
    strategy_assistant = StrategyAssistantNode("strategy_assistant")

    builder = StateGraph(
        state_schema=StrategyAgentState,
        input=InputState,
        config_schema=Configuration,
    )

    builder.add_node(strategy_assistant.node_name, strategy_assistant)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, strategy_assistant.node_name)
    builder.add_conditional_edges(
        strategy_assistant.node_name,
        route_model_output,
    )
    builder.add_edge("tools", strategy_assistant.node_name)

    return builder.compile(
        interrupt_before=[], interrupt_after=[], name="StrategyPlannerAgent"
    )
