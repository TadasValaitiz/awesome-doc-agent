"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

import asyncio
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from strategy_agent.planner_graph import (
    create_planner_with_reflection_graph,
)
from strategy_agent.coding_graph import create_coding_with_reflection_graph
from strategy_agent.configuration import Configuration
from strategy_agent.state import StrategyAgentState


async def graph(config: RunnableConfig):
    graph = StateGraph(StrategyAgentState, config_schema=Configuration)
    graph.add_node("strategy_planner", create_planner_with_reflection_graph(config))

    # Await the async function directly
    strategy_coder = await create_coding_with_reflection_graph(config)
    graph.add_node("strategy_coder", strategy_coder)

    graph.add_edge(START, "strategy_planner")
    graph.add_edge("strategy_planner", "strategy_coder")
    graph.add_edge("strategy_coder", END)
    return graph.compile(name="StrategyAgent")
