#!/usr/bin/env python
import asyncio
import os
import pathlib
from openevals.code.e2b.pyright import create_async_e2b_pyright_evaluator
from strategy_agent.logger import server_logger
from strategy_agent.coding_sandbox import (
    async_get_or_create_sandbox,
    download_all_files_from_sandbox
)

async def test_sandbox():
    """
    Test function to initialize a sandbox and run the evaluator with the output.py content.
    
    This function:
    1. Initializes the sandbox
    2. Creates a pyright evaluator
    3. Reads output.py from the filesystem (not copied directly)
    4. Logs the sandbox output at each step
    """
    # Step 1: Initialize sandbox
    server_logger.info("Starting sandbox initialization")
    sandbox = await async_get_or_create_sandbox()

    
    # Step 2: Initialize evaluator
    server_logger.info("Creating pyright evaluator")
    evaluator = create_async_e2b_pyright_evaluator(
        sandbox=sandbox,
        sandbox_project_directory="/code",
        code_extraction_strategy="none",
    )
    server_logger.info(f"Evaluator created: {evaluator}")
    
    # Step 3: Read output.py using filesystem function
    output_file_path = pathlib.Path(__file__).parent / "sandbox_files" / "output.py"
    server_logger.info(f"Reading output file from: {output_file_path}")
    
    try:
        with open(output_file_path, "r") as file:
            code_content = file.read()
        server_logger.info(f"Successfully read output.py, content length: {len(code_content)}")
    except Exception as e:
        server_logger.error(f"Error reading output.py: {str(e)}")
        return
    
    # Define stdout and stderr handler callbacks
    async def stdout_handler(message):
        server_logger.info(f"Sandbox stdout: {message.line}")

    async def stderr_handler(message):
        server_logger.error(f"Sandbox stderr: {message.line}")
    
    # Step 4: Execute code in sandbox
    server_logger.info("Executing code in sandbox")
    
    try:
        result = await evaluator(
            outputs=code_content,
            reference_outputs=None,
            on_stdout=stdout_handler,
            on_stderr=stderr_handler,
        )
        server_logger.info(f"Evaluation result: {result}")
    except Exception as e:
        server_logger.error(f"Error during code evaluation: {str(e)}")
    
    # Step 5: Download files from sandbox after execution
    try:
        downloaded_files = await download_all_files_from_sandbox(
            sandbox, file_extensions=(".csv", ".json", ".txt", ".log", ".py")
        )
        server_logger.info(f"Downloaded {len(downloaded_files)} files from sandbox")
        for file_path, content in downloaded_files.items():
            # Create output directory if it doesn't exist
            output_dir = pathlib.Path(__file__).parent / "sandbox_output"
            output_dir.mkdir(exist_ok=True)
            
            # Write each file to the sandbox_output directory
            output_path = output_dir / pathlib.Path(file_path).name
            server_logger.info(f"Writing file to: {output_path}")
            with open(output_path, "w") as f:
                f.write(content)
            server_logger.info(f"Successfully wrote file: {output_path}")
    except Exception as e:
        server_logger.error(f"Error downloading files: {str(e)}")


def main():
    """Main entry point for the script."""
    server_logger.info("Starting test_sandbox script")
    asyncio.run(test_sandbox())
    server_logger.info("Completed test_sandbox script")


if __name__ == "__main__":
    main()
