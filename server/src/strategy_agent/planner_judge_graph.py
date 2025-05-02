"""Trading strategy creation plan judge.

This module implements a judge for evaluating trading strategy creation plans.
"""

from langgraph.graph import StateGraph, START, END
from openevals.llm import create_llm_as_judge
from strategy_agent.state import StrategyAgentState
from strategy_agent.logger import server_logger
from strategy_agent.prompts import planner_judge_prompt


def judge_planner(state, config):
    """Evaluate the assistant's trading strategy plan using a separate judge model."""
    evaluator = create_llm_as_judge(
        prompt=planner_judge_prompt,
        model="openai:o3-mini",
        feedback_key="pass",
    )
    eval_result = evaluator(
        outputs=state.messages[-1].content, inputs=None, reference_outputs=None
    )

    server_logger.info(f"Eval result: {eval_result}")

    if not isinstance(eval_result, dict):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": "The evaluation system encountered an error. Please try again.",
                }
            ]
        }
    if eval_result["score"]:
        return {
            "strategy_feedback": eval_result["comment"],
            "strategy_approved": True,
            "strategy_message_index": len(state.messages) - 1,
        }
    else:
        return {
            "strategy_feedback": eval_result["comment"],
            "strategy_approved": False,
            "strategy_message_index": len(state.messages) - 1,
            "messages": [{"role": "user", "content": eval_result["comment"]}],
        }


# Define the judge graph
def create_planner_judge_graph():
    graph = StateGraph(StrategyAgentState)
    graph.add_node(
        "judge_planner",
        judge_planner,
    )
    graph.add_edge(START, "judge_planner")
    graph.add_edge("judge_planner", END)
    return graph.compile()
