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

from strategy_agent.strategy_planner_graph import create_strategy_planner
from strategy_agent.configuration import Configuration
from strategy_agent.state import InputState, StrategyAgentState
from strategy_agent.tools import TOOLS
from strategy_agent.utils import init_model
from strategy_agent.reflection_graph import create_reflection_graph
from strategy_agent.trading_strategy_judge import trading_strategy_judge_graph
from strategy_agent.strategy_planner_graph import (
    create_strategy_planner_with_reflection_graph,
)
from strategy_agent.coding_graph import create_coding_with_reflection_graph


def graph(config: RunnableConfig):
    graph = StateGraph(StrategyAgentState, config_schema=Configuration)
    graph.add_node(
        "strategy_planner", create_strategy_planner_with_reflection_graph(config)
    )
    graph.add_node("strategy_coder", create_coding_with_reflection_graph(config))
    graph.add_edge(START, "strategy_planner")
    graph.add_edge("strategy_planner", "strategy_coder")
    graph.add_edge("strategy_coder", END)
    return graph.compile(name="StrategyAgent")
