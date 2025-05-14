from strategy_agent.market_data.data_loader_incremental import (
    load_incremental_data_sync,
)
from server.src.strategy_agent.sandbox.sandbox_files.base_example import run_backtest

if __name__ == "__main__":
    data = load_incremental_data_sync("BTCUSD", "5", since="2024-11-01", days=120)
    if data is not None:
        import os

        os.makedirs("data/optimization", exist_ok=True)
        data.to_parquet("data/optimization/backtest_data.pkl")
        run_backtest()
    else:
        print("No data found")
