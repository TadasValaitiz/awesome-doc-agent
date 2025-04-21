"""Trading strategy creation plan judge.

This module implements a judge for evaluating trading strategy creation plans.
"""

from langgraph.graph import StateGraph, START, END
from openevals.llm import create_llm_as_judge
from strategy_agent.state import StrategyAgentState


# Define a detailed critique prompt for trading strategy creation plans
trading_strategy_critique_prompt = """You are an expert quantitative finance judge evaluating trading strategy creation plans. Your task is to critique the AI assistant's latest trading strategy plan in the conversation below.

Evaluate the strategy plan based on these criteria:
1. Feasibility - Is the strategy realistic and implementable with the available resources?
2. Market Understanding - Does it demonstrate proper understanding of market dynamics and asset behavior?
3. Risk Management - Does it include robust risk assessment and management techniques?
4. Backtesting Approach - Is there a clear methodology for historical validation?
5. Performance Metrics - Are appropriate evaluation metrics defined?
6. Adaptability - Does the strategy account for changing market conditions?
7. Data Requirements - Are the data needs clearly specified and obtainable?
8. Implementation Path - Is there a clear roadmap for development and deployment?

If the strategy plan meets ALL criteria satisfactorily, set pass to True.

If you find ANY issues with the plan, do NOT set pass to True. Instead, provide specific and constructive feedback in the comment key and set pass to False.

Be detailed in your critique so the assistant can understand exactly how to improve the trading strategy plan.

<response>
{outputs}
</response>"""


# Define the judge function with a robust evaluation approach
def judge_trading_strategy(state, config):
    """Evaluate the assistant's trading strategy plan using a separate judge model."""
    evaluator = create_llm_as_judge(
        prompt=trading_strategy_critique_prompt,
        model="openai:o3-mini",
        feedback_key="pass",
    )
    eval_result = evaluator(
        outputs=state.messages[-1].content, inputs=None, reference_outputs=None
    )

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
        return {"judge_feedback": eval_result["comment"], "judge_approved": True}
    else:
        return {
            "judge_feedback": eval_result["comment"],
            "judge_approved": False,
            "messages": [{"role": "user", "content": eval_result["comment"]}],
        }


# Define the judge graph
trading_strategy_judge_graph = (
    StateGraph(StrategyAgentState)
    .add_node(judge_trading_strategy)
    .add_edge(START, "judge_trading_strategy")
    .add_edge("judge_trading_strategy", END)
    .compile()
)
