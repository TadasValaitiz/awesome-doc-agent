from langgraph.graph import StateGraph
from strategy_agent.sandbox.pyright import analize_code_with_pyright
from strategy_agent.coding_extractors import extract_code_from_markdown_code_blocks
from strategy_agent.state import StrategyAgentState
from strategy_agent.logger import server_logger
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage


async def run_reflection(state: StrategyAgentState):
    server_logger.info(
        f"[Code Judge] Run reflection remaining steps: {state.remaining_steps}, message_index: {len(state.messages)-1}"
    )
    code_markdown = str(state.messages[-1].content)
    server_logger.info(f"[Code Judge] Code Markdown:\n {code_markdown}")

    code = extract_code_from_markdown_code_blocks(code_markdown)

    result = await analize_code_with_pyright(code)

    server_logger.info(f"Code Evaluation Result {result}")

    # After code execution, download and log any output files

    if result and not result[0]:
        errors = [error["errorWithContext"] for error in result[1]]
        prettyErrors = "\n\n".join(errors)

        content = f"""
I ran pyright and found some problems with the code you generated:

```python
{code}
```

Errors:
{prettyErrors}

Instructions:
Try to fix it. Make sure to regenerate the entire code snippet.
"""

        message = HumanMessage(
            content=content,
        )
        return {
            "messages": [message],
            "code_approved": False,
            "code_feedback": errors,
            "code_output": code,
        }

    return {
        "code_approved": True,
        "code_feedback": None,
        "code_output": code,
    }


async def create_code_judge_graph(config: RunnableConfig):
    return (
        StateGraph(StrategyAgentState)
        .add_node("run_reflection", run_reflection)
        .add_edge("__start__", "run_reflection")
        .compile(name="CodingJudge")
    )
