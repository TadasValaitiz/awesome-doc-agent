from strategy_agent.logger import server_logger


async def base_strategy_example():
    import os
    import asyncio
    import aiofiles
    import pathlib

    async def read_file_content(file_path):
        try:
            async with aiofiles.open(file_path, "r") as file:
                return await file.read()
        except Exception as e:
            server_logger.error(f"Failed to read file {file_path}", exception=e)
            return f"# Error reading {file_path}: {str(e)}"

    # Use pathlib with asyncio.to_thread for file path operations
    async def get_file_paths():
        # Get the module path using __file__ which is already in memory
        file_path = pathlib.Path(__file__)
        base_dir = file_path.parent.parent

        # Construct correct paths with the base_strategy directory
        base_sandbox_dir = base_dir / "strategy_agent" / "sandbox" / "sandbox_files"
        strategy_module_dir = base_sandbox_dir / "strategy_module"

        logger_path = strategy_module_dir / "trading_logger.py"
        strategy_path = strategy_module_dir / "base_strategy.py"
        current_data_path = strategy_module_dir / "current_data.py"
        example_path = base_sandbox_dir / "base_example.py"
        return logger_path, strategy_path, current_data_path, example_path

    # Get file paths asynchronously
    logger_path, strategy_path, current_data_path, example_path = await get_file_paths()

    # Read file contents
    logger_content = await read_file_content(logger_path)
    base_strategy = await read_file_content(strategy_path)
    current_data_content = await read_file_content(current_data_path)
    example_content = await read_file_content(example_path)
    # Combine files with headers
    code_example = f"""
### Base Classes

```python
{logger_content}
```

```python
{current_data_content}
```

```python
{base_strategy}
```

### Example strategy blueprint

```python
{example_content}
```
"""

    return code_example


