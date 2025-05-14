from pathlib import Path
from datetime import datetime, date
import pandas as pd
from typing import Optional, Tuple


class DataCache:
    """Manages caching of trading data."""

    def __init__(self, cache_dir: Path = Path("data/cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_date_range(self, since: str, days: int) -> Tuple[date, date]:
        """Convert since and days to start and end dates."""
        start_date = datetime.fromisoformat(since).date()
        end_date = start_date + pd.Timedelta(days=days)
        return start_date, end_date

    def _generate_filename(
        self, symbol: str, timeframe: str, start_date: date, end_date: date
    ) -> str:
        """Generate cache filename based on parameters."""
        return f"{symbol}_{timeframe}min_{start_date:%Y%m%d}_{end_date:%Y%m%d}.csv"

    def get_cached_file(
        self, symbol: str, timeframe: str, since: str, days: int
    ) -> Optional[Path]:
        """Get cached file path if exists."""
        start_date, end_date = self._get_date_range(since, days)
        filename = self._generate_filename(symbol, timeframe, start_date, end_date)
        cache_file = self.cache_dir / filename
        return cache_file if cache_file.exists() else None

    def save_to_cache(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        since: str,
        days: int,
    ) -> Path:
        """Save data to cache file."""
        start_date, end_date = self._get_date_range(since, days)
        filename = self._generate_filename(symbol, timeframe, start_date, end_date)
        cache_file = self.cache_dir / filename
        data.to_csv(cache_file)
        return cache_file

    def load_from_cache(self, cache_file: Path) -> pd.DataFrame:
        """Load data from cache file."""
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)


# Create a default cache instance
cache = DataCache() 