import base64
import json
import io
import asyncio
import aiofiles
from pathlib import Path
from typing import Literal
from langgraph.graph import StateGraph
import pandas as pd
from strategy_agent.sandbox.pyright import analize_code_with_pyright, run_code
from strategy_agent.coding_extractors import extract_code_from_markdown_code_blocks
from strategy_agent.state import (
    BacktestResults,
    OptimizationArtifacts,
    StrategyAgentState,
    SerializableDataFrame,
)
from strategy_agent.logger import server_logger
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from strategy_agent.market_data.preload import preload_crypto_data_incremental
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.types import Command, interrupt
from strategy_agent.coding_judge_feedback_msg import (
    pyright_error_message,
    execution_failure_message,
    optimization_failure_message,
    poor_results_message,
    missing_data_message,
    backtesting_message,
    success_message,
    print_logs,
    print_params,
    print_results,
)


async def run_codecheck(state: StrategyAgentState):
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
        content = pyright_error_message(code, errors)

        message = HumanMessage(content=content)
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


async def run_data_preload(state: StrategyAgentState):
    yield Command(
        update={
            "optimization_step": "loading_data",
        }
    )
    data_dict = await preload_crypto_data_incremental(
        symbols=state.optimization_symbols,
        timeframes=state.optimization_timeframes,
        months=state.optimization_months,
        fresh_data=False,
    )

    # Convert pandas DataFrames to SerializableDataFrame objects asynchronously
    serializable_data = {}
    for symbol, timeframes in data_dict.items():
        serializable_data[symbol] = {}
        for timeframe, df in timeframes.items():
            # Use the async creation method to avoid blocking
            serializable_data[symbol][timeframe] = (
                await SerializableDataFrame.create_async(df)
            )

    yield Command(
        update={
            "optimization_step": "data_loaded",
            "optimization_data": serializable_data,
        }
    )


async def collect_optimization_artifacts(
    optimization_dir: Path,
) -> OptimizationArtifacts:
    # Get paths to result files
    results_path = optimization_dir / "results.json"
    plots_path = optimization_dir
    logs_path = optimization_dir / "logs.log"

    plots = None
    results = None
    logs_df = None

    # Read JSON results if exists
    results_exists = await asyncio.to_thread(results_path.exists)
    if results_exists:
        async with aiofiles.open(results_path, "r") as f:
            content = await f.read()
            results = json.loads(content)

    # Convert PNG plots to base64 if they exist
    plots = {}
    # Use asyncio.to_thread to make the blocking glob call non-blocking
    plot_files = await asyncio.to_thread(lambda: list(plots_path.glob("*.png")))

    async def process_plot_file(png_file):
        async with aiofiles.open(png_file, "rb") as f:
            plot_data = base64.b64encode(await f.read()).decode()
            return png_file.name, plot_data

    if plot_files:
        plot_results = await asyncio.gather(
            *[process_plot_file(png_file) for png_file in plot_files]
        )
        plots = dict(plot_results)

    # Convert logs to dataframe if exists
    logs_exists = await asyncio.to_thread(logs_path.exists)
    if logs_exists:
        async with aiofiles.open(logs_path, "r") as f:
            log_content = await f.read()
            # Split the log content by lines and create a DataFrame
            lines = log_content.splitlines()
            # Make pandas read_csv operation non-blocking
            logs_df = await asyncio.to_thread(
                lambda: pd.DataFrame(lines, columns=["log_entry"])
            )

    return OptimizationArtifacts(
        results=results,
        plots=plots,
        logs=SerializableDataFrame(logs_df),
    )


async def run_backtest(state: StrategyAgentState):
    symbol = state.optimization_symbols[0]
    timeframe = state.optimization_timeframes[0]
    data = state.optimization_data[symbol][timeframe]

    if data.df is None:
        error_message = missing_data_message(symbol, timeframe)
        yield Command(
            update={
                "optimization_step": "backtesting_failed",
                "optimization_step_message": error_message,
            }
        )
    else:
        status_message = backtesting_message(symbol, timeframe)
        yield Command(
            update={
                "optimization_step": "backtesting",
                "optimization_step_message": status_message,
            }
        )

        success, result = await run_code(code_content=state.code_output, data=data.df)

        server_logger.info(f"Code Execution success: {success} Result:{result}")

        optimization_dir = Path(result["optimization_dir"])
        optimization_artifacts = await collect_optimization_artifacts(optimization_dir)

        if not success:
            content = execution_failure_message(result["returncode"], result["stderr"])
            yield Command(
                update={
                    "messages": [HumanMessage(content=content)],
                    "optimization_step": "backtesting_failed",
                    "optimization_artifacts": optimization_artifacts,
                    "optimization_step_message": f"Execution failed with code {result['returncode']}: {result['stderr']}",
                }
            )
        else:
            opt = (
                optimization_artifacts.results.get("best_results", None)
                if optimization_artifacts.results
                else None
            )
            if opt is None:
                logs_df = (
                    optimization_artifacts.logs.df
                    if optimization_artifacts.logs
                    else None
                )
                content = optimization_failure_message(print_logs(logs_df))
                yield Command(
                    update={
                        "messages": [HumanMessage(content=content)],
                        "optimization_step": "backtesting_failed",
                        "optimization_artifacts": optimization_artifacts,
                        "optimization_step_message": "Optimization failed to finish. Results are missing.",
                    }
                )

            elif (
                opt["trades"] < 5
                or opt["sharpe_ratio"] < 1.0
                or opt["max_drawdown"] > 15
            ):
                content = poor_results_message(
                    print_results(optimization_artifacts.results),
                    print_params(optimization_artifacts.results),
                )

                yield Command(
                    update={
                        "messages": [HumanMessage(content=content)],
                        "optimization_step": "backtesting_poor_results",
                        "optimization_artifacts": optimization_artifacts,
                        "optimization_step_message": "Optimization results are not good enough. Please improve the strategy and try again.",
                    }
                )

            else:
                success_msg = success_message()
                yield Command(
                    update={
                        "optimization_step": "backtesting_finished",
                        "optimization_artifacts": optimization_artifacts,
                        "optimization_step_message": success_msg,
                    }
                )


def route_after_codecheck(
    state: StrategyAgentState,
) -> Literal["__end__", "run_data_preload"]:
    if state.code_approved:
        return "run_data_preload"
    else:
        return "__end__"


async def create_code_judge_graph(config: RunnableConfig):
    return (
        StateGraph(StrategyAgentState)
        .add_node("run_codecheck", run_codecheck)
        .add_node("run_data_preload", run_data_preload)
        .add_node("run_backtest", run_backtest)
        .add_edge(START, "run_codecheck")
        .add_conditional_edges(
            "run_codecheck",
            route_after_codecheck,
        )
        .add_edge("run_data_preload", "run_backtest")
        .add_edge("run_backtest", END)
        .compile(name="CodingJudge")
    )
