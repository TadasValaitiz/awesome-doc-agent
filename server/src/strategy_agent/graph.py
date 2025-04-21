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
from strategy_agent.state import InputState, State
from strategy_agent.tools import TOOLS
from strategy_agent.utils import init_model


builder = StateGraph(
    state_schema=State,
    input=InputState,
    config_schema=Configuration,
)

builder.add_node("strategy_planner", create_strategy_planner())

builder.add_edge(START, "strategy_planner")
builder.add_edge("strategy_planner", END)

graph = builder.compile(interrupt_before=[], interrupt_after=[], name="StrategyAgent")
