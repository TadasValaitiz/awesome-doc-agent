from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import os
import asyncio
import aiofiles
from typing import Optional, List, Dict, Tuple, Any
from strategy_agent.logger import server_logger as logger
from .cache import cache
import ccxt.async_support as ccxt_async  # Use async version of ccxt
import ccxt  # Keep sync version for type hints
import glob

# Global semaphore to limit concurrent API calls to Binance
# Adjust max_concurrent_requests based on Binance rate limits
MAX_CONCURRENT_REQUESTS = 2
binance_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Exchange instance pool to reuse connections
exchange_pool: Dict[str, Any] = {}
exchange_pool_lock = asyncio.Lock()


async def load_incremental_data(
    symbol: str,
    timeframe: str = "1",
    since: Optional[str] = None,
    days: int = 1,
    fresh_data: bool = False,
) -> Optional[pd.DataFrame]:
    """Load data incrementally, storing each day separately.

    Args:
        symbol: Symbol to trade (e.g., BTCUSD)
        timeframe: Timeframe in minutes for data fetching
        since: Start time in ISO format (e.g., 2024-02-07T00:00:00)
        days: Number of days of historical data to download
        fresh_data: If True, bypass cache and load fresh data from Binance

    Returns:
        DataFrame with OHLCV data or None if loading fails
    """
    if not symbol:
        logger.error(
            "No symbol provided",
            data={"message": "Symbol must be provided for data loading"},
            tags=["data", "error"],
        )
        return None

    if not since:
        since = (datetime.now() - timedelta(days=days)).isoformat()

    # Parse since date and create list of days to fetch
    start_date = datetime.fromisoformat(since)
    date_ranges = _get_date_ranges(start_date, days)

    # Initialize exchange for the whole operation
    exchange = await get_exchange_instance()
    try:
        # Initialize empty DataFrame to store all data
        all_data = []

        # Process days sequentially to avoid overwhelming the API
        for day_start, day_end in date_ranges:
            day_data = await _process_day(
                symbol=symbol,
                timeframe=timeframe,
                day_start=day_start,
                day_end=day_end,
                fresh_data=fresh_data,
                exchange=exchange,
            )

            if day_data is not None:
                all_data.append(day_data)

   
        combined_data = pd.concat(all_data)
        combined_data.sort_index(inplace=True)
        logger.info(
            "Successfully loaded all data",
            data={"symbol": symbol, "timeframe": timeframe, "days": days},
            tags=["data", "success"],
        )
        return combined_data
    finally:
        # Always release the exchange back to the pool
        await release_exchange_instance(exchange)


async def get_exchange_instance():
    """Get a Binance exchange instance from the pool or create a new one."""
    async with exchange_pool_lock:
        if "binance" not in exchange_pool:
            # Initialize with default settings
            try:
                exchange = ccxt_async.binance(
                    {
                        "enableRateLimit": True,  # Enable the built-in rate limiter
                        "timeout": 30000,  # Increase timeout to 30 seconds
                    }
                )

                # Test the connection with retries
                max_retries = 3
                for attempt in range(1, max_retries + 1):
                    try:
                        await exchange.load_markets()
                        break
                    except Exception as e:
                        if attempt == max_retries:
                            logger.error(
                                f"Failed to connect to Binance after {max_retries} attempts: {str(e)}"
                            )
                            raise
                        logger.warning(
                            f"Connection attempt {attempt} failed, retrying in 2 seconds: {str(e)}"
                        )
                        await asyncio.sleep(2)

                exchange_pool["binance"] = exchange
                logger.info("Successfully initialized Binance exchange instance")
            except Exception as e:
                logger.error(f"Failed to initialize Binance exchange: {str(e)}")
                raise

        return exchange_pool["binance"]


async def release_exchange_instance(exchange):
    """Return the exchange instance to the pool."""
    # Do not close, just return to pool
    pass


async def close_all_exchanges():
    """Close all exchange connections in the pool."""
    async with exchange_pool_lock:
        for exchange_name, exchange in exchange_pool.items():
            try:
                await exchange.close()
                logger.info(f"Closed {exchange_name} connection")
            except Exception as e:
                logger.error(f"Error closing {exchange_name} connection: {str(e)}")
        exchange_pool.clear()


async def _process_day(
    symbol: str,
    timeframe: str,
    day_start: datetime,
    day_end: datetime,
    fresh_data: bool,
    exchange=None,
) -> Optional[pd.DataFrame]:
    """Process a single day of data, either from cache or from Binance."""
    day_data = None

    # Try to load from cache first if fresh_data is False
    if not fresh_data:
        day_cache_file = await _get_daily_cache_file(symbol, timeframe, day_start)

        # Check if file exists - use async to avoid blocking
        loop = asyncio.get_event_loop()
        file_exists = await loop.run_in_executor(
            None, lambda: os.path.exists(day_cache_file)
        )

        if day_cache_file and file_exists:
            logger.info(
                "Loading daily data from cache",
                data={
                    "file": str(day_cache_file),
                    "date": day_start.date().isoformat(),
                },
                tags=["data", "cache", "load"],
            )
            try:
                day_data = await _read_cache_file(day_cache_file)
            except Exception as e:
                logger.warning(
                    "Failed to load day from cache, falling back to Binance",
                    data={"error": str(e), "date": day_start.date().isoformat()},
                    tags=["data", "cache", "error"],
                )

    # If not loaded from cache, fetch from Binance
    if day_data is None:
        # Use the semaphore to limit concurrent API calls
        async with binance_semaphore:
            day_data = await _load_binance_day(
                symbol=symbol,
                timeframe=timeframe,
                start_date=day_start,
                end_date=day_end,
                exchange=exchange,
            )

        # Add a delay between days to respect rate limits
        await asyncio.sleep(0.5)

        # Save to daily cache if data was loaded successfully
        if day_data is not None:
            try:
                cache_file = await _save_daily_cache(
                    day_data, symbol, timeframe, day_start
                )
                logger.info(
                    "Daily data saved to cache",
                    data={
                        "file": str(cache_file),
                        "date": day_start.date().isoformat(),
                    },
                    tags=["data", "cache", "save"],
                )
            except Exception as e:
                logger.error(
                    "Failed to save day to cache",
                    exception=e,
                    data={"date": day_start.date().isoformat()},
                    tags=["data", "cache", "error"],
                )

    return day_data


async def _read_cache_file(cache_file: str) -> pd.DataFrame:
    """Read a cache file asynchronously."""
    # For pandas read_csv, we need to use a file-like object or a string
    # Since pandas doesn't have a direct async read_csv, we'll load the file content
    # asynchronously and then use pandas to parse it
    async with aiofiles.open(cache_file, "r") as f:
        content = await f.read()

    # Use pandas to parse the CSV content from memory
    # We use StringIO to create a file-like object from the string
    from io import StringIO

    df = pd.read_csv(StringIO(content))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df


def _get_date_ranges(
    start_date: datetime, days: int
) -> List[Tuple[datetime, datetime]]:
    """Generate a list of (start, end) date tuples for each day in the range."""
    date_ranges = []
    for i in range(days):
        day_start = start_date + timedelta(days=i)
        day_start = day_start.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1) - timedelta(seconds=1)
        date_ranges.append((day_start, day_end))
    return date_ranges


async def _get_daily_cache_file(symbol: str, timeframe: str, day: datetime) -> str:
    """Generate a cache filename for a specific day (async version)."""
    cache_dir = cache.cache_dir
    # Format: symbol_timeframe_YYYY-MM-DD.csv
    filename = f"{symbol}_{timeframe}_{day.strftime('%Y-%m-%d')}.csv"
    return str(cache_dir / filename)


async def _save_daily_cache(
    df: pd.DataFrame, symbol: str, timeframe: str, day: datetime
) -> str:
    """Save daily data to the cache asynchronously."""
    cache_dir = cache.cache_dir

    # Use asyncio.to_thread to make blocking makedirs call non-blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: os.makedirs(cache_dir, exist_ok=True))

    # Format: symbol_timeframe_YYYY-MM-DD.csv
    filename = f"{symbol}_{timeframe}_{day.strftime('%Y-%m-%d')}.csv"
    cache_file = cache_dir / filename

    # Reset index to include timestamp as a column
    df_to_save = df.reset_index()

    # Save to a string buffer first (this is CPU-bound so no need to make async)
    from io import StringIO

    buffer = StringIO()
    df_to_save.to_csv(buffer, index=False)
    buffer.seek(0)

    # Write the buffer content to file asynchronously
    async with aiofiles.open(str(cache_file), "w") as f:
        await f.write(buffer.getvalue())

    return str(cache_file)


async def _load_binance_day(
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    exchange=None,
) -> Optional[pd.DataFrame]:
    """Load data for a single day from Binance exchange asynchronously.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSD')
        timeframe: Timeframe in minutes (e.g., '1', '5', '15')
        start_date: Start date/time
        end_date: End date/time
        exchange: Optional exchange instance to reuse

    Returns:
        pd.DataFrame with OHLCV data for the specified day
    """
    # Convert symbol to Binance format
    symbol_map = {
        "BTCUSD": "BTC/USDT",
        "ETHUSD": "ETH/USDT",
        "ADAUSD": "ADA/USDT",
        "DOGEUSD": "DOGE/USDT",
        "XRPUSD": "XRP/USDT",
        "SOLUSD": "SOL/USDT",
        "DOTUSD": "DOT/USDT",
        "LINKUSD": "LINK/USDT",
        "AVAXUSD": "AVAX/USDT",
        "MATICUSD": "MATIC/USDT",
    }

    if symbol not in symbol_map:
        raise ValueError(
            f"Unsupported symbol: {symbol}. Must be one of {list(symbol_map.keys())}"
        )

    binance_symbol = symbol_map[symbol]

    # Convert timeframe from minutes to valid ccxt timeframe format
    timeframe_map = {
        "1": "1m",
        "5": "5m",
        "15": "15m",
        "30": "30m",
        "60": "1h",
        "240": "4h",
        "1440": "1d",
    }

    if timeframe not in timeframe_map:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}. Must be one of {list(timeframe_map.keys())}"
        )

    ccxt_timeframe = timeframe_map[timeframe]

    # Convert dates to Unix timestamp (milliseconds)
    since_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)

    logger.info(
        f"Fetching {symbol} data for {start_date.date().isoformat()} from Binance...",
        data={
            "symbol": symbol,
            "binance_symbol": binance_symbol,
            "timeframe": timeframe,
            "ccxt_timeframe": ccxt_timeframe,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        tags=["data", "binance", "data"],
    )

    need_to_close = False
    try:
        # Use provided exchange instance or get one from the pool
        if exchange is None:
            exchange = await get_exchange_instance()
            need_to_close = False  # Don't close the shared instance

        # Initialize empty list to store all candles
        all_candles = []
        current_since = since_ts
        candles_per_request = 1000  # Binance limit per request

        # Calculate timeframe in milliseconds
        timeframe_minutes = int(timeframe)
        timeframe_ms = timeframe_minutes * 60 * 1000

        # Keep fetching until we reach the end date
        while current_since < end_ts:
            try:
                # Fetch a chunk of candles
                candles = await exchange.fetch_ohlcv(
                    symbol=binance_symbol,
                    timeframe=ccxt_timeframe,
                    since=current_since,
                    limit=candles_per_request,
                )

                if not candles:
                    break  # No more data available

                # Filter candles to only include those within our date range
                filtered_candles = [c for c in candles if c[0] <= end_ts]
                all_candles.extend(filtered_candles)

                # Update the since timestamp for the next iteration
                # Use the timestamp of the last candle plus one timeframe interval
                if candles:
                    last_timestamp = candles[-1][0]
                    current_since = last_timestamp + timeframe_ms
                else:
                    break

                logger.info(
                    "Fetched data chunk",
                    data={
                        "chunk_size": len(filtered_candles),
                        "total_fetched": len(all_candles),
                        "current_since": datetime.fromtimestamp(
                            current_since / 1000
                        ).isoformat(),
                    },
                    tags=["data", "binance", "pagination"],
                )

                # Add a larger delay to avoid rate limits (adjust based on Binance's limits)
                await asyncio.sleep(1.5)  # Increased from ~1s to 1.5s

            except ccxt_async.NetworkError as e:
                logger.warning(
                    f"Network error when fetching data, retrying in 5 seconds: {str(e)}",
                    tags=["data", "binance", "error", "retry"],
                )
                await asyncio.sleep(5)  # Longer delay on network error
                continue

            except ccxt_async.ExchangeError as e:
                if "rate limit" in str(e).lower():
                    logger.warning(
                        f"Rate limit hit, waiting longer: {str(e)}",
                        tags=["data", "binance", "rate_limit"],
                    )
                    await asyncio.sleep(10)  # Much longer delay on rate limit
                    continue
                else:
                    # Re-raise other exchange errors
                    raise

        if not all_candles:
            logger.warning(
                "No data received from Binance for this day",
                data={"date": start_date.date().isoformat()},
                tags=["data", "binance", "warning"],
            )
            return None

        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        # Convert timestamp to datetime and set as index
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # Sort by timestamp
        df.sort_index(inplace=True)

        # Rename columns to match format
        df.columns = ["Open", "High", "Low", "Close", "Volume"]

        logger.info(
            "Successfully loaded Binance data for day",
            data={
                "symbol": symbol,
                "date": start_date.date().isoformat(),
                "points": len(df),
            },
            tags=["binance", "data", "success"],
        )

        return df

    except Exception as e:
        logger.error(
            "Failed to fetch daily data from Binance",
            exception=e,
            data={
                "symbol": symbol,
                "binance_symbol": binance_symbol,
                "date": start_date.date().isoformat(),
            },
            tags=["data", "binance", "error"],
        )
        return None
    finally:
        # Only close if we created our own instance
        if need_to_close and exchange:
            await exchange.close()


async def get_available_dates(symbol: str, timeframe: str) -> List[str]:
    """Get a list of dates for which data is available in the cache.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSD')
        timeframe: Timeframe in minutes (e.g., '1', '5', '15')

    Returns:
        List of date strings in ISO format (YYYY-MM-DD)
    """
    cache_dir = cache.cache_dir
    pattern = f"{symbol}_{timeframe}_*.csv"

    # Use asyncio.to_thread for non-blocking file operations
    loop = asyncio.get_event_loop()
    files = await loop.run_in_executor(
        None, lambda: glob.glob(str(cache_dir / pattern))
    )

    # Extract dates from filenames - this is CPU-bound so no need for async
    dates = []
    for file in files:
        filename = os.path.basename(file)
        # Extract the date part (after the second underscore)
        parts = filename.split("_")
        if len(parts) >= 3:
            date_str = parts[2].replace(".csv", "")
            dates.append(date_str)

    # Sort dates
    dates.sort()
    return dates


# Synchronous wrapper for backward compatibility
def load_incremental_data_sync(
    symbol: str,
    timeframe: str = "1",
    since: Optional[str] = None,
    days: int = 1,
    fresh_data: bool = False,
) -> Optional[pd.DataFrame]:
    """Synchronous wrapper around the async load_incremental_data function."""
    try:
        return asyncio.run(
            load_incremental_data(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                days=days,
                fresh_data=fresh_data,
            )
        )
    except Exception as e:
        logger.error(
            "Error in synchronous wrapper for load_incremental_data",
            exception=e,
            data={"symbol": symbol, "timeframe": timeframe, "days": days},
            tags=["data", "sync_wrapper", "error"],
        )
        return None
    finally:
        # Make sure we close any remaining exchange connections
        try:
            asyncio.run(close_all_exchanges())
        except Exception:
            # Ignore errors in cleanup
            pass


# Synchronous wrapper for backward compatibility
def get_available_dates_sync(symbol: str, timeframe: str) -> List[str]:
    """Synchronous wrapper around the async get_available_dates function."""
    try:
        return asyncio.run(get_available_dates(symbol, timeframe))
    except Exception as e:
        logger.error(
            "Error in synchronous wrapper for get_available_dates",
            exception=e,
            data={"symbol": symbol, "timeframe": timeframe},
            tags=["data", "sync_wrapper", "error"],
        )
        return []
