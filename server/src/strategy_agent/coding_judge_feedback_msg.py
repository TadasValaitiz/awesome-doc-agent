"""
This module contains the feedback message templates used by the coding judge.
These templates are used to provide feedback to the user when evaluating code.
"""

from typing import List, Dict, Any, Optional, Union, TypeVar, cast
import pandas as pd
from strategy_agent.state import BacktestResults

# TypeVar for handling both Dict and BacktestResults
T = TypeVar("T", Dict[str, Any], BacktestResults)


def print_logs(logs_df: Optional[pd.DataFrame]) -> str:
    """
    Format logs dataframe as a string.

    Args:
        logs_df: The logs dataframe

    Returns:
        Formatted logs string
    """
    if logs_df is None:
        return ""
    return logs_df.tail(20).to_string()


def print_params(results: Optional[T]) -> str:
    """
    Format best parameters as a string.

    Args:
        results: The results dictionary or BacktestResults

    Returns:
        Formatted parameters string
    """
    if results is None:
        return ""
    # Use dict-like access which works for both dict and TypedDict
    params = results.get("best_parameters", None)
    if params is None:
        return ""

    markdown = "### Strategy Parameters\n"
    for key, value in params.items():
        markdown += f"- {key}: {value}\n"
    return markdown


def print_results(results: Optional[T]) -> str:
    """
    Format optimization results as a markdown string.

    Args:
        results: The results dictionary or BacktestResults

    Returns:
        Formatted results markdown
    """
    opt = results.get("best_results", None) if results else None
    if opt is None:
        return ""

    markdown = "## Backtest Results\n\n"
    # Format metrics into sections
    markdown += "### Returns & Risk Metrics\n"
    markdown += f"- Total Return: {opt['total_return']:.2f}%\n"
    markdown += f"- Buy & Hold Return: {opt['buy_hold_return']:.2f}%\n"
    markdown += f"- CAGR: {opt['cagr']:.2f}%\n"
    markdown += f"- Sharpe Ratio: {opt['sharpe_ratio']:.2f}\n"
    markdown += f"- Sortino Ratio: {opt['sortino_ratio']:.2f}\n"
    markdown += f"- Calmar Ratio: {opt['calmar_ratio']:.2f}\n"
    markdown += f"- Volatility (Ann.): {opt['volatility_ann']:.2f}%\n"
    markdown += f"- Max Drawdown: {opt['max_drawdown']:.2f}%\n"
    markdown += f"- Avg Drawdown: {opt['avg_drawdown']:.2f}%\n\n"

    markdown += "### Trade Statistics\n"
    markdown += f"- Number of Trades: {opt['trades']}\n"
    markdown += f"- Win Rate: {opt['win_rate']:.2f}%\n"
    markdown += f"- Average Trade: {opt['avg_trade']:.2f}%\n"
    markdown += f"- Best Trade: {opt['best_trade']:.2f}%\n"
    markdown += f"- Worst Trade: {opt['worst_trade']:.2f}%\n"
    markdown += f"- Profit Factor: {opt['profit_factor']:.2f}\n"
    markdown += f"- Expectancy: {opt['expectancy']:.2f}%\n"
    markdown += f"- SQN: {opt['sqn']:.2f}\n"
    markdown += f"- Kelly Criterion: {opt['kelly_criterion']:.2f}\n"
    markdown += f"- Exposure Time: {opt['exposure_time']:.2f}%\n\n"

    markdown += "### Account Statistics\n"
    markdown += f"- Final Equity: ${opt['equity_final']:.2f}\n"
    markdown += f"- Peak Equity: ${opt['equity_peak']:.2f}\n"

    return markdown


def pyright_error_message(code: Optional[str], errors: List[str]) -> str:
    """
    Generate a message for when Pyright finds errors in the code.

    Args:
        code: The code that was analyzed
        errors: List of error messages from Pyright

    Returns:
        Formatted error message
    """
    pretty_errors = "\n\n".join(errors)
    safe_code = code or ""

    return f"""
I ran pyright and found some problems with the code you generated:

```python
{safe_code}
```

Errors:
{pretty_errors}

Instructions:
Try to fix it. Make sure to regenerate the entire code snippet.
"""


def execution_failure_message(returncode: Union[int, str], stderr: str) -> str:
    """
    Generate a message for when code execution fails.

    Args:
        returncode: The return code from the execution
        stderr: The standard error output

    Returns:
        Formatted error message
    """
    return f"""
Execution failed with code {returncode}: {stderr}

Please fix the code and try again.
"""


def optimization_failure_message(logs: str) -> str:
    """
    Generate a message for when optimization fails to finish.

    Args:
        logs: The optimization logs

    Returns:
        Formatted error message
    """
    return f"""
Optimization failed to finish. Please check the logs for more information and fix the code.

Logs:
{logs}
"""


def poor_results_message(results: Any, params: Any) -> str:
    """
    Generate a message for when optimization results are not good enough.

    Args:
        results: The optimization results
        params: The optimization parameters

    Returns:
        Formatted error message
    """
    return f"""
Optimization results are not good enough. Please improve the strategy and try again.

Results:
{results}

Parameters:
{params}
"""


def missing_data_message(symbol: str, timeframe: str) -> str:
    """
    Generate a message for when market data is missing.

    Args:
        symbol: The trading symbol
        timeframe: The timeframe

    Returns:
        Formatted error message
    """
    return f"Market data is missing for {symbol} {timeframe}"


def backtesting_message(symbol: str, timeframe: str) -> str:
    """
    Generate a message for when backtesting is running.

    Args:
        symbol: The trading symbol
        timeframe: The timeframe

    Returns:
        Formatted status message
    """
    return f"Running backtest for {symbol} {timeframe}"


def success_message() -> str:
    """
    Generate a message for when optimization results are successful.

    Returns:
        Formatted success message
    """
    return "Optimization results are good enough."
