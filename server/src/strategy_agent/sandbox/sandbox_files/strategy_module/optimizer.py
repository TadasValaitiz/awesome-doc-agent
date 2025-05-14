from dataclasses import dataclass
from typing import Dict, List, Tuple, Type, TypedDict, Union, Optional, Any, Callable
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Integer, Real, Dimension
from backtesting import Backtest
from scipy.optimize._optimize import OptimizeResult

from .logger import ConsoleLogger, FileLogger
from .trading_logger import TradingLogger
from .base_strategy import BaseStrategy


@dataclass
class StaticParams:
    """Parameters for backtesting that remain constant during optimization.

    Attributes:
        data: DataFrame containing OHLCV data for backtesting
        strategy: Strategy class to optimize
        cash: Initial cash amount
        spread: Spread in price units
        commission: Commission as percentage or fixed amount
        margin: Margin requirement
        trade_on_close: Whether to execute trades on close price
        hedging: Whether to allow hedging
        exclusive_orders: Whether orders are mutually exclusive
        finalize_trades: Whether to wait for confirmation of closed trades
        cv_splits: Number of cross-validation splits
        metric: Metric to optimize ("sharpe", "returns", "sortino", or "custom")
    """

    cash: float = 10_000
    spread: float = 0.0
    commission: Union[float, Tuple[float, float]] = 0.0
    margin: float = 1.0
    trade_on_close: bool = False
    hedging: bool = False
    exclusive_orders: bool = False
    finalize_trades: bool = False
    cv_splits: int = 1
    metric: str = "sharpe"


class BacktestResults(TypedDict):
    """Backtest results for Any strategy.

    Contains all performance metrics returned by backtesting framework
    in a serializable format.
    """

    total_return: float  # Return [%]
    buy_hold_return: float  # Buy & Hold Return [%]
    sharpe_ratio: float  # Sharpe Ratio
    sortino_ratio: float  # Sortino Ratio
    max_drawdown: float  # Max. Drawdown [%]
    avg_drawdown: float  # Avg. Drawdown [%]
    trades: int  # # Trades
    win_rate: float  # Win Rate [%]
    avg_trade: float  # Avg. Trade [%]
    best_trade: float  # Best Trade [%]
    worst_trade: float  # Worst Trade [%]
    exposure_time: float  # Exposure Time [%]
    equity_final: float  # Equity Final [$]
    equity_peak: float  # Equity Peak [$]
    volatility_ann: float  # Volatility (Ann.) [%]
    cagr: float  # CAGR [%]
    calmar_ratio: float  # Calmar Ratio
    profit_factor: float  # Profit Factor
    expectancy: float  # Expectancy [%]
    sqn: float  # SQN
    kelly_criterion: float  # Kelly Criterion


class StrategyOptimizer:
    """Optimizer for trading strategies using Bayesian optimization.

    This class implements Bayesian optimization for finding optimal parameters
    for a trading strategy, using cross-validation to prevent overfitting.

    Attributes:
        best_params: Best parameters found during optimization
        optimize_results: Full optimization results
        best_results: Backtest results with the best parameters
    """

    best_params: Optional[Dict[str, float | int]] = None
    optimize_results: Optional[OptimizeResult] = None
    best_results: Optional[BacktestResults] = None

    def __init__(
        self,
        session_id: str,
        data: pd.DataFrame,
        strategy: Type[BaseStrategy],
        params: StaticParams,
        space_tuples: List[Tuple[str, Union[Integer, Real]]],
    ) -> None:
        """Initialize the optimizer.

        Args:
            session_id: Unique identifier for the optimization session
            params: Static parameters for the backtesting
            space_tuples: List of tuples (parameter_name, parameter_space)
                where parameter_space is a skopt space object
        """
        self.data = data
        self.strategy = strategy
        self.params = params
        self.logger = TradingLogger(
            session_id=session_id,
            loggers=[FileLogger(session_id=session_id)],
        )
        self.space_tuples = space_tuples
        self.space = [space[1] for space in space_tuples]
        self.space_names = [space[0] for space in space_tuples]
        self.optimization_history = []

    def optimize(
        self, n_calls: int = 50, n_random_starts: int = 10
    ) -> Optional[Tuple[OptimizeResult, Dict[str, float], BacktestResults]]:
        """Run Bayesian optimization to find optimal strategy parameters.

        Uses Gaussian Process optimization to find parameters that maximize
        the chosen metric in cross-validation.

        Args:
            n_calls: Total number of evaluations
            n_random_starts: Number of random evaluations before using GP model

        Returns:
            Tuple of (optimization_results, best_parameters, backtest_results)
            or None if optimization failed
        """
        try:
            self.logger.info(
                "Starting optimization",
                data={
                    "n_calls": n_calls,
                    "n_random_starts": n_random_starts,
                    "parameter_space": {
                        dim[0]: [dim[1].low, dim[1].high] for dim in self.space_tuples
                    },
                    "initial_cash": self.params.cash,
                    "commission_rate": self.params.commission,
                },
                tags=["strategy_optimizer", "start"],
            )

            self.optimize_results = gp_minimize(
                func=self._objective,
                dimensions=self.space,
                n_calls=n_calls,
                n_random_starts=n_random_starts,
                noise="gaussian",
                verbose=False,
            )
            if self.optimize_results is not None:
                self.best_params = dict(zip(self.space_names, self.optimize_results.x))
                bt = Backtest(
                    self.data,
                    self.strategy,
                    cash=self.params.cash,
                    commission=self.params.commission,
                    exclusive_orders=self.params.exclusive_orders,
                    trade_on_close=self.params.trade_on_close,
                    margin=self.params.margin,
                )

                self.stats = bt.run(**self.best_params)
                self.best_results = to_backtest_results(self.stats)
                self.logger.info(
                    "Ending optimization",
                    data={
                        "params": serialize_params(self.best_params),
                        "results": self.best_results,
                    },
                    tags=["strategy_optimizer", "end"],
                )
                # Only return the tuple if all values are not None
                if self.best_params is not None and self.best_results is not None:
                    return (self.optimize_results, self.best_params, self.best_results)

            # If any component is None, return None
            return None

        except Exception as e:
            import traceback

            self.logger.error(
                "Optimization failed",
                data={
                    "optimization_state": {
                        "results": str(self.optimize_results),
                    },
                },
                exception=e,
                tags=["strategy_optimizer", "error"],
            )
            return None

    def singe_run(self, params: Dict[str, float | int]):
        self.best_params = params
        bt = Backtest(
            self.data,
            self.strategy,
            cash=self.params.cash,
            commission=self.params.commission,
            exclusive_orders=self.params.exclusive_orders,
            trade_on_close=self.params.trade_on_close,
            margin=self.params.margin,
        )

        self.stats = bt.run(**params)
        self.best_results = to_backtest_results(self.stats)
        self.logger.info(
            "Single Backtest results",
            data={
                "params": serialize_params(params),
                "results": self.best_results,
            },
            tags=["strategy_optimizer", "end"],
        )

        return self.stats, self.best_results

    def plot_optimization_results(
        self,
        save_path: Optional[str] = None,
        show_plots: bool = False,
        figsize: Tuple[int, int] = (12, 8),
    ) -> None:
        """Plot optimization results using skopt's plotting utilities.

        Creates visualizations of the optimization process including:
        1. Convergence plot showing optimization progress
        2. Parameter importance plot showing the effect of each parameter
        3. Parameter relationships plot (if at least 2 parameters)

        Args:
            save_path: Directory path to save plots (None = don't save)
            show_plots: Whether to display plots interactively
            figsize: Size of the figures (width, height) in inches

        Requires the optimize method to have been called first.
        """
        if self.optimize_results is None:
            self.logger.warning(
                "No optimization results to plot",
                data={"message": "Run optimize() first"},
                tags=["strategy_optimizer", "plot"],
            )
            return

        from skopt.plots import plot_convergence, plot_objective, plot_evaluations
        import matplotlib.pyplot as plt
        import os

        # 1. Convergence plot (optimization history)
        plt.figure(figsize=figsize)
        plot_convergence(self.optimize_results)
        plt.title("Optimization Convergence", fontsize=14)
        plt.xlabel("Number of Calls", fontsize=12)
        plt.ylabel(f"Negative {self.params.metric.capitalize()} Ratio", fontsize=12)
        plt.grid(True, alpha=0.3)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(
                os.path.join(save_path, "convergence_plot.png"),
                dpi=300,
                bbox_inches="tight",
            )

        if show_plots:
            plt.show()
        else:
            plt.close()

        # 2. Parameter importance and relationships
        if len(self.space) >= 1:
            plt.figure(figsize=figsize)
            plot_objective(self.optimize_results)
            relationships = [f"{i} - {name}" for i, name in enumerate(self.space_names)]
            plt.suptitle(
                t=f"Parameter Importance and Relationships ({relationships})",
                fontsize=14,
            )
            # Add parameter names to the plot

            if save_path is not None:
                plt.savefig(
                    os.path.join(save_path, "parameter_importance.png"),
                    dpi=300,
                    bbox_inches="tight",
                )

            if show_plots:
                plt.show()
            else:
                plt.close()

        # 3. Parameter evaluations scatter plot
        if len(self.space) >= 2:
            plt.figure(figsize=figsize)
            plot_evaluations(self.optimize_results)
            relationships = [f"{i} - {name}" for i, name in enumerate(self.space_names)]

            plt.suptitle(
                t=f"Parameter Evaluations ({relationships})",
                fontsize=14,
            )

            if save_path is not None:
                plt.savefig(
                    os.path.join(save_path, "parameter_evaluations.png"),
                    dpi=300,
                    bbox_inches="tight",
                )

            if show_plots:
                plt.show()
            else:
                plt.close()

        # 4. Plot optimization history as a custom visualization
        if self.optimization_history:
            plt.figure(figsize=figsize)

            # Extract scores for plotting (convert from negative values used in optimization)
            scores = [
                -h["score"] for h in self.optimization_history if h["score"] > -1e9
            ]
            iterations = list(range(1, len(scores) + 1))

            plt.plot(iterations, scores, "bo-", linewidth=1.5)

            # Add best score marker
            if scores:
                best_score = max(scores)
                best_iter = scores.index(best_score) + 1
                plt.plot(
                    best_iter,
                    best_score,
                    "ro",
                    markersize=10,
                    label=f"Best Score: {best_score:.4f} (iteration {best_iter})",
                )

            plt.grid(True, alpha=0.3)
            plt.title(
                f"Optimization Progress - {self.params.metric.capitalize()}",
                fontsize=14,
            )
            plt.xlabel("Iteration", fontsize=12)
            plt.ylabel(f"{self.params.metric.capitalize()} Score", fontsize=12)
            plt.legend()

            if save_path is not None:
                plt.savefig(
                    os.path.join(save_path, "optimization_history.png"),
                    dpi=300,
                    bbox_inches="tight",
                )

            if show_plots:
                plt.show()
            else:
                plt.close()

    def plot_best_backtest_results(
        self,
        save_path: Optional[str] = None,
        show_plots: bool = False,
        figsize: Tuple[int, int] = (12, 8),
    ) -> None:
        """Plot the backtest results using the best parameters found.

        Creates visualizations showing the performance of the best strategy:
        1. Performance metrics summary
        2. Comparison with top metrics across optimization

        Args:
            save_path: Directory path to save plots (None = don't save)
            show_plots: Whether to display plots interactively
            figsize: Size of the figures (width, height) in inches

        Requires the optimize method to have been called first.
        """
        if self.best_results is None or self.best_params is None:
            self.logger.warning(
                "No optimization results to plot",
                data={"message": "Run optimize() first"},
                tags=["strategy_optimizer", "plot"],
            )
            return

        import matplotlib.pyplot as plt
        import numpy as np
        import os

        # 1. Plot key metrics as a bar chart
        plt.figure(figsize=figsize)

        # Select key metrics to display
        key_metrics = {
            "Total Return (%)": self.best_results["total_return"],
            "Sharpe Ratio": self.best_results["sharpe_ratio"],
            "Max Drawdown (%)": self.best_results["max_drawdown"],
            "Win Rate (%)": self.best_results["win_rate"],
            "# Trades": self.best_results["trades"],
        }

        # Create bar chart
        colors = ["#2C7BB6", "#2C7BB6", "#D7191C", "#2C7BB6", "#2C7BB6"]
        bars = plt.bar(
            range(len(key_metrics)), list(key_metrics.values()), color=colors
        )
        plt.xticks(range(len(key_metrics)), list(key_metrics.keys()), rotation=45)
        plt.title("Key Performance Metrics with Optimal Parameters", fontsize=14)
        plt.grid(axis="y", alpha=0.3)

        # Add data labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(
                os.path.join(save_path, "performance_metrics.png"),
                dpi=300,
                bbox_inches="tight",
            )

        if show_plots:
            plt.show()
        else:
            plt.close()

        # 2. Plot parameters table
        plt.figure(figsize=(figsize[0], figsize[1] // 2))
        plt.axis("off")

        param_names = list(self.best_params.keys())
        param_values = [f"{self.best_params[p]:.4f}" for p in param_names]

        table_data = [param_values]

        table = plt.table(
            cellText=table_data,
            rowLabels=["Value"],
            colLabels=param_names,
            loc="center",
            cellLoc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)

        plt.title("Optimal Parameters", fontsize=14, pad=50)

        if save_path is not None:
            plt.savefig(
                os.path.join(save_path, "optimal_parameters.png"),
                dpi=300,
                bbox_inches="tight",
            )

        if show_plots:
            plt.show()
        else:
            plt.close()

    def _objective(self, dimen_params: List[float]) -> float:
        """Objective function for Bayesian optimization.

        Args:
            dimen_params: List of parameter values to evaluate

        Returns:
            Negative score (for minimization) of the parameters
        """
        try:
            # Convert parameters to Python native types to avoid serialization issues
            params = {}
            for name, value in zip(self.space_names, dimen_params):
                if isinstance(value, np.integer):
                    params[name] = int(value)
                elif isinstance(value, np.floating):
                    params[name] = float(value)
                else:
                    params[name] = value

            mean_score, score_std = self._calculate_cv_score(params)

            # Store result in history with native Python types
            self.optimization_history.append(
                {"params": params, "score": float(mean_score), "std": float(score_std)}
            )

            # Handle infinite values
            if np.isinf(mean_score) or np.isnan(mean_score):
                return -1e10  # Return a large negative but finite number

            # We minimize the negative score (maximizing the actual score)
            return float(-mean_score)  # Ensure we return a float

        except Exception as e:
            self.logger.error(
                "Error in objective function",
                exception=e,
                tags=["strategy_optimizer", "error"],
            )
            return -1e10  # Return a large negative but finite number

    def _calculate_cv_score(self, params: Dict[str, float]) -> Tuple[float, float]:
        """Run cross-validation and return mean score and standard deviation.

        Args:
            params: Strategy parameters to evaluate

        Returns:
            Tuple of (mean_score, score_std) across CV folds
        """
        # Split data into CV periods
        split_size = len(self.data) // self.params.cv_splits
        scores = []
        for i in range(self.params.cv_splits):
            try:
                start_idx = i * split_size
                end_idx = start_idx + split_size
                cv_data = self.data.iloc[start_idx:end_idx]

                # Run backtest with params
                bt = Backtest(
                    cv_data,
                    self.strategy,
                    cash=self.params.cash,
                    commission=self.params.commission,
                    exclusive_orders=self.params.exclusive_orders,
                    trade_on_close=self.params.trade_on_close,
                    margin=self.params.margin,
                )

                stats = bt.run(**params)

                backtest_results = to_backtest_results(stats)
                if backtest_results is None:
                    self.logger.warning(
                        "Stats are None",
                        tags=["strategy_optimizer", "stats"],
                    )
                    score = -1e10
                elif self.params.metric == "sharpe":
                    score = backtest_results.get("sharpe_ratio", -1e10)
                elif self.params.metric == "returns":
                    score = backtest_results.get("total_return", -1e10)
                elif self.params.metric == "sortino":
                    score = backtest_results.get("sortino_ratio", -1e10)
                else:
                    # Custom metric combining returns and drawdown
                    returns = backtest_results.get("total_return", -1e10)
                    drawdown = min(backtest_results.get("max_drawdown", 100), 100)
                    score = returns * (1 - abs(drawdown) / 100)

                # Handle NaN/inf values
                if np.isnan(score) or np.isinf(score):
                    self.logger.warning(
                        "Score is NaN/inf",
                        data={"score": score, "stats": backtest_results},
                        tags=["strategy_optimizer", "scores"],
                    )
                    score = -1e10  # Use a large negative but finite number

                scores.append(float(score))  # Ensure we append a float

            except Exception as e:
                self.logger.error(
                    f"Error in fold {i}",
                    exception=e,
                    tags=["strategy_optimizer", "error"],
                )
                scores.append(-1e10)  # Use a large negative but finite number

        if not scores:
            self.logger.warning(
                "No scores",
                tags=["strategy_optimizer", "scores"],
            )
            return (
                -1e10,
                0.0,
            )  # Return large negative but finite number if all folds failed

        # Filter out extremely negative scores for mean/std calculation
        valid_scores = [s for s in scores if s > -1e10]
        if not valid_scores:
            self.logger.warning(
                "Scores are not valid",
                data={"scores": scores},
                tags=["strategy_optimizer", "scores"],
            )
            return -1e10, 0.0

        return float(np.mean(valid_scores)), float(np.std(valid_scores))

    def save_optimization_results(self, file_path: str) -> bool:
        """Save optimization results to a file for later analysis or reference.

        Saves a dictionary containing the best parameters, best backtest results,
        and optimization history to a JSON file.

        Args:
            file_path: Path to save the results file

        Returns:
            True if saving was successful, False otherwise

        Requires the optimize method to have been called first.
        """
        if self.best_params is None or self.best_results is None:
            self.logger.warning(
                "No optimization results to save",
                data={"message": "Run optimize() first"},
                tags=["strategy_optimizer", "save"],
            )
            return False

        try:
            import json
            import os
            from datetime import datetime

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

            # Format history entries for serialization
            history = []
            for entry in self.optimization_history:
                # Convert params to string representation if needed
                if "params" in entry and isinstance(entry["params"], dict):
                    history.append(
                        {
                            "params": {k: str(v) for k, v in entry["params"].items()},
                            "score": float(entry["score"]),
                            "std": float(entry["std"]) if "std" in entry else 0.0,
                        }
                    )

            # Create results dictionary using serialize_params to handle numpy types
            results = {
                "timestamp": datetime.now().isoformat(),
                "best_parameters": serialize_params(self.best_params),
                "best_results": self.best_results,  # Already converted by to_backtest_results
                "static_params": {
                    "cash": float(self.params.cash),
                    "commission": (
                        float(self.params.commission)
                        if isinstance(self.params.commission, (int, float))
                        else self.params.commission
                    ),
                    "margin": float(self.params.margin),
                    "metric": str(self.params.metric),
                    "cv_splits": int(self.params.cv_splits),
                },
            }

            # Use Python's default encoder with a custom encoder
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer, int)):
                        return int(obj)
                    if isinstance(obj, (np.floating, float)):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, np.bool_):
                        return bool(obj)
                    return super().default(obj)

            # Save to file with custom encoder for any remaining numpy types
            with open(file_path, "w") as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)

            self.logger.info(
                "Optimization results saved",
                data={"file_path": file_path},
                tags=["strategy_optimizer", "save"],
            )

            return True

        except Exception as e:
            self.logger.error(
                "Failed to save optimization results",
                exception=e,
                data={"file_path": file_path},
                tags=["strategy_optimizer", "save", "error"],
            )
            return False

    def save_full_stats(self, file_path: str) -> bool:
        """Save full backtesting stats to a file.

        This dumps the entire stats pandas Series object to a CSV file.

        Args:
            file_path: Path to save the stats file

        Returns:
            True if saving was successful, False otherwise
        """
        if not hasattr(self, "stats"):
            self.logger.warning(
                "No stats to save",
                data={"message": "Run single_run() or optimize() first"},
                tags=["strategy_optimizer", "save"],
            )
            return False

        try:
            import os

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

            # Save stats to CSV
            self.stats.to_csv(file_path)

            self.logger.info(
                "Full stats saved to CSV",
                data={"file_path": file_path},
                tags=["strategy_optimizer", "save"],
            )

            return True

        except Exception as e:
            self.logger.error(
                "Failed to save full stats",
                exception=e,
                data={"file_path": file_path},
                tags=["strategy_optimizer", "save", "error"],
            )
            return False

    @classmethod
    def load_optimization_results(cls, file_path: str) -> Dict[str, Any]:
        """Load previously saved optimization results.

        Args:
            file_path: Path to the saved results file

        Returns:
            Dictionary containing the optimization results

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file doesn't contain valid optimization results
        """
        import json

        with open(file_path, "r") as f:
            results = json.load(f)

        if not all(k in results for k in ["best_parameters", "best_results"]):
            raise ValueError("Invalid optimization results file")

        return results


def to_backtest_results(series: pd.Series) -> Optional[BacktestResults]:
    """Convert pandas Series to a serializable dictionary.

    Args:
        series: Series containing backtest results from Backtest.run()

    Returns:
        Dictionary of results in BacktestResults format or None if empty
    """
    if series is None or series.empty:
        return None

    # Helper function to handle NaN values
    def clean_value(value) -> float:
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            # For JSON serialization, we'll return 0.0 instead of None for numeric fields
            # This avoids type compatibility issues with BacktestResults
            return 0.0
        return value

    # Create the results dictionary with proper type handling
    results: BacktestResults = {
        "total_return": clean_value(float(series["Return [%]"])),
        "buy_hold_return": clean_value(float(series["Buy & Hold Return [%]"])),
        "sharpe_ratio": clean_value(float(series["Sharpe Ratio"])),
        "sortino_ratio": clean_value(float(series["Sortino Ratio"])),
        "max_drawdown": clean_value(float(series["Max. Drawdown [%]"])),
        "avg_drawdown": clean_value(float(series["Avg. Drawdown [%]"])),
        "trades": int(series["# Trades"]),
        "win_rate": clean_value(float(series["Win Rate [%]"])),
        "avg_trade": clean_value(float(series["Avg. Trade [%]"])),
        "best_trade": clean_value(float(series["Best Trade [%]"])),
        "worst_trade": clean_value(float(series["Worst Trade [%]"])),
        "exposure_time": clean_value(float(series["Exposure Time [%]"])),
        "equity_final": clean_value(float(series["Equity Final [$]"])),
        "equity_peak": clean_value(float(series["Equity Peak [$]"])),
        "volatility_ann": clean_value(float(series["Volatility (Ann.) [%]"])),
        "cagr": clean_value(float(series["CAGR [%]"])),
        "calmar_ratio": clean_value(float(series["Calmar Ratio"])),
        "profit_factor": clean_value(float(series["Profit Factor"])),
        "expectancy": clean_value(float(series["Expectancy [%]"])),
        "sqn": clean_value(float(series["SQN"])),
        "kelly_criterion": clean_value(float(series["Kelly Criterion"])),
    }
    return results


def serialize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert NumPy values to serializable Python types.

    Args:
        params: Dictionary containing parameter values which may include NumPy types

    Returns:
        Dictionary with all values converted to serializable Python types
    """
    if params is None:
        return {}

    serialized = {}
    for key, value in params.items():
        # Check for NaN or inf values
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            serialized[key] = 0.0  # Use 0.0 for NaN/inf values for consistent typing
        # Check for NumPy integer types
        elif np.issubdtype(type(value), np.integer):
            serialized[key] = int(value)
        # Check for NumPy float types
        elif np.issubdtype(type(value), np.floating):
            serialized[key] = float(value)
        # Check for NumPy boolean
        elif isinstance(value, np.bool_):
            serialized[key] = bool(value)
        # Check for NumPy arrays
        elif isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        else:
            serialized[key] = value
    return serialized
