from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import krakenex
from typing import Optional, Union
from strategy_agent.logger import server_logger as logger
from .cache import cache
import ccxt


def load_data(
    symbol: Optional[str] = None,
    timeframe: str = "1",
    since: Optional[str] = None,
    days: int = 1,
    fresh_data: bool = False,
) -> Optional[pd.DataFrame]:
    """Load data from either cache, CSV file or Kraken API.

    Args:
        symbol: Symbol to trade (for live data)
        timeframe: Timeframe in minutes for data fetching
        since: Start time in ISO format (e.g., 2024-02-07T00:00:00)
        days: Number of days of historical data to download
        fresh_data: If True, bypass cache and load fresh data from Kraken

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

    # Try to load from cache first if fresh_data is False
    if not fresh_data:
        cached_file = cache.get_cached_file(symbol, timeframe, since, days)
        if cached_file:
            logger.info(
                "Loading data from cache",
                data={"file": str(cached_file)},
                tags=["data", "cache", "load"],
            )
            try:
                return cache.load_from_cache(cached_file)
            except Exception as e:
                logger.warning(
                    "Failed to load from cache, falling back to Kraken",
                    data={"error": str(e)},
                    tags=["data", "cache", "error"],
                )

    # Load from Kraken
    data = load_binance_data( symbol, timeframe, since, days)

    # Save to cache if data was loaded successfully
    if data is not None:
        try:
            cache_file = cache.save_to_cache(data, symbol, timeframe, since, days)
            logger.info(
                "Data saved to cache",
                data={"file": str(cache_file)},
                tags=["data", "cache", "save"],
            )
        except Exception as e:
            logger.error(
                "Failed to save to cache",
                exception=e,
                tags=["data", "cache", "error"],
            )

    return data


def load_kraken_data(
    session_id: str,
    symbol: str,
    timeframe: str = "1",
    since: Optional[str] = None,
    days: int = 1,
) -> Optional[pd.DataFrame]:
    """Load data from Kraken exchange.

    Args:
        symbol: Symbol to trade (e.g., BTCUSD)
        timeframe: Timeframe in minutes for data fetching
        since: Start time in ISO format (e.g., 2024-02-07T00:00:00)
        days: Number of days of historical data to download

    Returns:
        DataFrame with OHLCV data or None if loading fails
    """
    try:
        k = krakenex.API()

        # Convert symbol to Kraken format (e.g., BTCUSD -> XBTUSD)
        if symbol.startswith("BTC"):
            symbol = "XBT" + symbol[3:]

        # Set default time range if not specified
        if not since:
            since = (datetime.now() - timedelta(days=days)).isoformat()

        # Convert since to Unix timestamp
        since_ts = int(datetime.fromisoformat(since).timestamp())

        logger.info(
            f"Fetching {symbol} data from Kraken...",
            data={"timeframe": timeframe, "since": since, "since_ts": since_ts},
            tags=["data", "kraken", "data"],
        )

        # Fetch OHLCV data
        ohlcv = k.query_public(
            "OHLC", data={"pair": symbol, "interval": int(timeframe), "since": since_ts}
        )

        if "error" in ohlcv and ohlcv["error"]:
            logger.error(
                f"Kraken API error",
                data={"error": ohlcv["error"]},
                tags=["data", "kraken", "api_error"],
            )
            return None

        if "result" not in ohlcv or not ohlcv["result"]:
            logger.error("No data received from Kraken", tags=["kraken", "api_error"])
            return None

        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv["result"][list(ohlcv["result"].keys())[0]],
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "vwap",
                "volume",
                "count",
            ],
        )

        if df.empty:
            logger.error(
                "Received empty dataset from Kraken", tags=["kraken", "data_error"]
            )
            return None

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)

        # Convert string values to float
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        # Rename columns to match expected format
        df.columns = ["Open", "High", "Low", "Close", "VWAP", "Volume", "Count"]

        logger.info(
            f"Successfully loaded Kraken data",
            data={"symbol": symbol, "points": len(df)},
            tags=["kraken", "data", "success"],
        )
        return df[["Open", "High", "Low", "Close", "Volume"]]

    except Exception as e:
        logger.error(
            "Error fetching data from Kraken",
            data={"error": str(e)},
            tags=["kraken", "api_error"],
        )
        return None


def load_binance_data(
    symbol: str,
    timeframe: str,
    since: Optional[str] = None,
    days: int = 1,
) -> Optional[pd.DataFrame]:
    """Load historical OHLCV data from Binance using ccxt.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSD')
        timeframe: Candle timeframe in minutes (e.g., '1', '5', '15')
        since: Start time in ISO format (e.g., '2024-02-02T00:00:00')
        days: Number of days of historical data to download

    Returns:
        pd.DataFrame with columns: Open, High, Low, Close, Volume
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

    # Set default time range if not specified
    if not since:
        since = (datetime.now() - timedelta(days=days)).isoformat()

    # Convert since to Unix timestamp
    since_ts = int(datetime.fromisoformat(since).timestamp())

    logger.info(
        f"Fetching {symbol} data from Binance...",
        data={
            "symbol": symbol,
            "binance_symbol": binance_symbol,
            "timeframe": timeframe,
            "ccxt_timeframe": ccxt_timeframe,
            "since": since,
            "since_ts": since_ts,
        },
        tags=["data", "binance", "data"],
    )

    # Calculate the maximum number of candles that can be fetched
    timeframe_minutes = int(timeframe)
    time_diff = datetime.now() - datetime.fromisoformat(since)
    total_minutes = time_diff.total_seconds() / 60
    max_candles = int(total_minutes / timeframe_minutes)

    logger.info(
        "Calculated candles needed",
        data={
            "timeframe_minutes": timeframe_minutes,
            "total_minutes": total_minutes,
            "max_candles": max_candles,
        },
        tags=["data", "binance", "calculation"],
    )

    try:
        # Initialize Binance exchange
        exchange = ccxt.binance()

        # Initialize empty list to store all candles
        all_candles = []
        current_since = (
            since_ts * 1000
        )  # Start from the initial timestamp (in milliseconds)
        candles_per_request = 1000  # Binance limit per request

        while len(all_candles) < max_candles:
            # Fetch a chunk of candles
            candles = exchange.fetch_ohlcv(
                symbol=binance_symbol,
                timeframe=ccxt_timeframe,
                since=current_since,
                limit=candles_per_request,
            )

            if not candles:
                break  # No more data available

            all_candles.extend(candles)

            # Update the since timestamp for the next iteration
            # Use the timestamp of the last candle plus one timeframe interval
            last_timestamp = candles[-1][0]  # First element is timestamp
            current_since = last_timestamp + (
                timeframe_minutes * 60 * 1000
            )  # Convert minutes to milliseconds

            logger.info(
                "Fetched data chunk",
                data={
                    "chunk_size": len(candles),
                    "total_fetched": len(all_candles),
                    "max_candles": max_candles,
                    "current_since": current_since,
                },
                tags=["data", "binance", "pagination"],
            )

            # Add a small delay to avoid rate limits
            exchange.sleep(max(exchange.rateLimit, 1000))

        if not all_candles:
            logger.error(
                "No data received from Binance", tags=["data", "binance", "error"]
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
            "Successfully loaded Binance data",
            data={
                "symbol": symbol,
                "binance_symbol": binance_symbol,
                "points": len(df),
            },
            tags=["binance", "data", "success"],
        )

        return df

    except Exception as e:
        logger.error(
            "Failed to fetch data from Binance",
            exception=e,
            data={"symbol": symbol, "binance_symbol": binance_symbol},
            tags=["data", "binance", "error"],
        )
        raise
