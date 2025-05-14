import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

from strategy_agent.logger import server_logger as logger
from .data_loader import load_data
from .data_loader_incremental import load_incremental_data, get_available_dates


def preload_crypto_data(
    symbols: List[str] = ["BTCUSD", "ETHUSD"],
    timeframes: List[str] = ["1", "5", "15", "60"],
    months: int = 3,
    fresh_data: bool = False,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Preloads historical data for specified cryptocurrency symbols for the last N months.

    Args:
        session_id: Unique session identifier
        symbols: List of symbols to preload (default: ["BTCUSD", "ETHUSD"])
        timeframes: List of timeframes in minutes (default: ["1", "5", "15", "60"])
        months: Number of months of historical data to download (default: 3)
        fresh_data: If True, bypass cache and load fresh data (default: False)

    Returns:
        Dictionary with structure {symbol: {timeframe: dataframe}}
    """
    logger.info(
        "Starting cryptocurrency data preloading",
        data={
            "symbols": symbols,
            "timeframes": timeframes,
            "months": months,
            "fresh_data": fresh_data,
        },
        tags=["preload", "start"],
    )

    # Calculate the start date (N months ago)
    since_date = (datetime.now() - timedelta(days=30 * months)).isoformat()
    days = 30 * months  # Approximate number of days in N months

    # Dictionary to store the results
    result_data = {}

    for symbol in symbols:
        logger.info(
            f"Preloading data for {symbol}",
            data={"symbol": symbol, "since": since_date, "days": days},
            tags=["preload", "symbol", symbol],
        )

        result_data[symbol] = {}

        for timeframe in timeframes:
            logger.info(
                f"Preloading {symbol} data for {timeframe} minute timeframe",
                data={"symbol": symbol, "timeframe": timeframe},
                tags=["preload", "timeframe", timeframe],
            )

            try:
                # Load the data
                df = load_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since_date,
                    days=days,
                    fresh_data=fresh_data,
                )

                if df is not None and not df.empty:
                    result_data[symbol][timeframe] = df
                    logger.info(
                        f"Successfully preloaded {symbol} {timeframe}m data",
                        data={
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "rows": len(df),
                            "start_date": (
                                df.index.min().isoformat() if not df.empty else None
                            ),
                            "end_date": (
                                df.index.max().isoformat() if not df.empty else None
                            ),
                        },
                        tags=["preload", "success", symbol, timeframe],
                    )
                else:
                    logger.warning(
                        f"No data returned for {symbol} {timeframe}m",
                        data={"symbol": symbol, "timeframe": timeframe},
                        tags=["preload", "warning", "empty_data", symbol, timeframe],
                    )
            except Exception as e:
                logger.error(
                    f"Failed to preload {symbol} {timeframe}m data",
                    exception=e,
                    data={"symbol": symbol, "timeframe": timeframe},
                    tags=["preload", "error", symbol, timeframe],
                )

    # Log summary of preloaded data
    preloaded_summary = {
        symbol: list(timeframes.keys()) for symbol, timeframes in result_data.items()
    }

    logger.info(
        "Cryptocurrency data preloading completed",
        data={
            "preloaded_data": preloaded_summary,
            "total_symbols": len(result_data),
            "total_timeframes": sum(
                len(timeframes) for timeframes in result_data.values()
            ),
        },
        tags=["preload", "complete"],
    )

    return result_data


async def preload_crypto_data_incremental(
    symbols: List[str] = ["BTCUSD", "ETHUSD"],
    timeframes: List[str] = ["5"],
    months: int = 3,
    fresh_data: bool = False,
):
    """
    Preloads historical data for specified cryptocurrency symbols for the last N months.

    Args:
        symbols: List of symbols to preload (default: ["BTCUSD", "ETHUSD"])
        timeframes: List of timeframes to preload (default: ["5"])
        months: Number of months of historical data to preload
        fresh_data: If True, bypass cache and load fresh data
    """
    # Calculate since date based on months
    since = (datetime.now() - timedelta(days=months * 30)).isoformat()
    days = months * 30  # Approximation of months to days

    successful = 0
    failed = 0
    data_dict: Dict[str, Dict[str, pd.DataFrame]] = {}
    for symbol in symbols:
        data_dict[symbol] = {}
        for timeframe in timeframes:
            try:
                logger.info(
                    f"Preloading {symbol} data at {timeframe}min timeframe for past {months} months",
                    data={"symbol": symbol, "timeframe": timeframe, "months": months},
                    tags=["preload", "start"],
                )

                # Check what data we already have
                if not fresh_data:
                    existing_dates = await get_available_dates(symbol, timeframe)
                    if existing_dates:
                        logger.info(
                            f"Found existing data for {symbol} at {timeframe}min",
                            data={
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "dates": len(existing_dates),
                            },
                            tags=["preload", "existing"],
                        )

                # Load data incrementally
                data = await load_incremental_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    days=days,
                    fresh_data=fresh_data,
                )

                if data is not None and not data.empty:
                    data_dict[symbol][timeframe] = data
                    logger.info(
                        f"Successfully preloaded {symbol} data at {timeframe}min timeframe",
                        data={
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "rows": len(data),
                            "date_range": f"{data.index.min()} to {data.index.max()}",
                        },
                        tags=["preload", "success"],
                    )
                    successful += 1
                else:
                    logger.warning(
                        f"Failed to preload {symbol} data at {timeframe}min timeframe: Empty dataset",
                        data={"symbol": symbol, "timeframe": timeframe},
                        tags=["preload", "warning"],
                    )
                    failed += 1

            except Exception as e:
                logger.error(
                    f"Error preloading {symbol} data at {timeframe}min timeframe",
                    exception=e,
                    data={"symbol": symbol, "timeframe": timeframe},
                    tags=["preload", "error"],
                )
                failed += 1

    logger.info(
        "Preloading complete",
        data={"successful": successful, "failed": failed},
        tags=["preload", "complete"],
    )

    return data_dict


if __name__ == "__main__":
    asyncio.run(
        preload_crypto_data_incremental(
            symbols=["BTCUSD", "ETHUSD"],
            timeframes=["5"],
            months=6,
            fresh_data=False,
        )
    )
