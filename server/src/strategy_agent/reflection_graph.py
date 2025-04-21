from typing import Optional, Type, Any, Literal, get_type_hints
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage
from strategy_agent.state import StrategyAgentState
from dataclasses import fields


def end_or_reflect(state: StrategyAgentState) -> Literal[END, "graph"]:
    if state.remaining_steps < 2:
        return END
    if len(state.messages) == 0:
        return END
    last_message = state.messages[-1]
    if isinstance(last_message, HumanMessage):
        return "graph"
    else:
        return END


def create_reflection_graph(
    graph: CompiledStateGraph,
    reflection: CompiledStateGraph,
    state_schema: Optional[Type[Any]] = None,
    config_schema: Optional[Type[Any]] = None,
) -> StateGraph:
    _state_schema = state_schema or graph.builder.schema

    # Get field names from the dataclass
    field_names = [f.name for f in fields(_state_schema)]

    if "remaining_steps" not in field_names:
        raise ValueError("Missing required field 'remaining_steps' in state_schema")

    if "messages" not in field_names:
        raise ValueError("Missing required field 'messages' in state_schema")

    if "strategy_feedback" not in field_names:
        raise ValueError("Missing required field 'strategy_feedback' in state_schema")

    if "strategy_approved" not in field_names:
        raise ValueError("Missing required field 'strategy_approved' in state_schema")

    if "code_feedback" not in field_names:
        raise ValueError("Missing required field 'code_feedback' in state_schema")

    if "code_approved" not in field_names:
        raise ValueError("Missing required field 'code_approved' in state_schema")


    rgraph = StateGraph(state_schema, config_schema=config_schema)
    rgraph.add_node("graph", graph)
    rgraph.add_node("reflection", reflection)
    rgraph.add_edge(START, "graph")
    rgraph.add_edge("graph", "reflection")
    rgraph.add_conditional_edges("reflection", end_or_reflect)
    return rgraph
