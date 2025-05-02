from typing import Optional, Type, Any, Literal, get_type_hints
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage
from strategy_agent.state import StrategyAgentState
from dataclasses import fields
from strategy_agent.logger import server_logger


def skip_or_reflect(state: StrategyAgentState) -> Literal[END, "reflection"]:
    server_logger.info(f"[Reflection Node] Skip or Reflect : {state.judge_skip}")

    if state.judge_skip:
        return END
    else:
        return "reflection"


def end_or_reflect(state: StrategyAgentState) -> Literal[END, "graph"]:
    server_logger.info(f"[Reflection Node] Code plan: {state.code_output}")
    server_logger.info(
        f"[Reflection Node] Strategy feedback: {state.strategy_feedback}"
    )
    server_logger.info(f"[Reflection Node] Code feedback: {state.code_feedback}")
    server_logger.info(
        f"[Reflection Node] Strategy approved: {state.strategy_approved}"
    )
    server_logger.info(f"[Reflection Node] Code approved: {state.code_approved}")
    server_logger.info(f"[Reflection Node] Remaining steps: {state.remaining_steps}")

    if state.remaining_steps < 2:
        server_logger.info(
            f"[Reflection Node] Ending reflection due to no remaining steps"
        )
        return END
    if len(state.messages) == 0:
        server_logger.info(f"[Reflection Node] Ending reflection due to no messages")
        return END
    last_message = state.messages[-1]
    if isinstance(last_message, HumanMessage):
        server_logger.info(f"[Reflection Node] Routing to main graph")
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

    if "strategy_message_index" not in field_names:
        raise ValueError("Missing required field 'strategy_message_index' in state_schema")

    if "code_output" not in field_names:
        raise ValueError("Missing required field 'code_output' in state_schema")

    if "judge_skip" not in field_names:
        raise ValueError("Missing required field 'judge_skip' in state_schema")

    rgraph = StateGraph(state_schema, config_schema=config_schema)
    rgraph.add_node("graph", graph)
    rgraph.add_node("reflection", reflection)
    rgraph.add_edge(START, "graph")
    rgraph.add_conditional_edges("graph", skip_or_reflect)
    rgraph.add_conditional_edges("reflection", end_or_reflect)
    return rgraph
