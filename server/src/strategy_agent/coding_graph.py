from typing import Optional

from e2b_code_interpreter import Sandbox, AsyncSandbox
import asyncio

from strategy_agent.state import StrategyAgentState

from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model

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
from langchain_core.output_parsers import StrOutputParser
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
from strategy_agent.coding_judge_graph import create_code_judge_graph
from strategy_agent.reflection_graph import create_reflection_graph
from strategy_agent.logger import server_logger


class StrategyCodeNode:
    """Abstract base class for analysis nodes that follow a common pattern."""

    def __init__(self, node_name: str):
        self.node_name = node_name

    async def __call__(
        self, state: StrategyAgentState, *, config: Optional[RunnableConfig] = None
    ):
        """Execute the analysis node with the common pattern."""
        configuration = Configuration.from_runnable_config(config)

        def extract_prompt(s: StrategyAgentState):
            history = ChatPromptTemplate.from_messages(s.messages).format()
            return {
                "code_example": "",
                "history": history,
            }

        prompt = ChatPromptTemplate.from_template(configuration.code_system_prompt)
        llm = init_model(configuration.code_model)

        chain = extract_prompt | prompt | llm

        chunks = []
        async for chunk in chain.astream(state, config):
            chunks.append(chunk)
            yield Command(update={"messages": [chunk]})

        final_message = functools.reduce(
            lambda acc, chunk: acc.__add__(chunk),
            chunks[1:],
            chunks[0],
        )

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


def create_strategy_coder():
    strategy_code = StrategyCodeNode("strategy_coder")

    builder = StateGraph(
        state_schema=StrategyAgentState,
        input=InputState,
        config_schema=Configuration,
    )

    builder.add_node(strategy_code.node_name, strategy_code)
    builder.add_edge(START, strategy_code.node_name)
    builder.add_edge(strategy_code.node_name, END)

    return builder.compile(
        interrupt_before=[], interrupt_after=[], name="StrategyCoderAgent"
    )


_GLOBAL_SANDBOX = None


async def async_get_or_create_sandbox():
    global _GLOBAL_SANDBOX
    if _GLOBAL_SANDBOX is None:
        _GLOBAL_SANDBOX = await AsyncSandbox.create("OpenEvalsPython")
    return _GLOBAL_SANDBOX


async def create_coding_with_reflection_graph(
    config: RunnableConfig,
):
    configurable = config.get("configurable", {})
    sandbox = configurable.get("sandbox", None)
    if sandbox is None:
        sandbox = await async_get_or_create_sandbox()

    return create_reflection_graph(
        create_strategy_coder(), create_code_judge_graph(sandbox), StrategyAgentState
    ).compile(interrupt_before=[], interrupt_after=[], name="StrategyCoderReflection")
