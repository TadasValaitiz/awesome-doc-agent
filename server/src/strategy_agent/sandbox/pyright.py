import asyncio
import json
import subprocess
import tempfile
import os
import pathlib
from typing import Any, Literal
from strategy_agent.logger import server_logger
from pathlib import Path
import argparse


async def analize_code_with_pyright(code_content=None, use_sandbox_directory=True):
    """
    Test pyright analysis on Python code.

    Args:
        code_content (str, optional): Python code to analyze. If None, will read from output.py file
                                       unless use_sandbox_directory=True.
        use_sandbox_directory (bool): Whether to analyze the entire sandbox_files directory.
    """
    if code_content is None and not use_sandbox_directory:
        # Step 3: Read output.py using filesystem function
        output_file_path = pathlib.Path(__file__).parent / "sandbox_files" / "output.py"
        server_logger.info(f"Reading output file from: {output_file_path}")

        try:
            # Use asyncio.to_thread to read file asynchronously
            code_content = await asyncio.to_thread(read_file, output_file_path)
            server_logger.info(
                f"Successfully read output.py, content length: {len(code_content)}"
            )
        except Exception as e:
            server_logger.error(f"Error reading output.py: {str(e)}")
            return
    elif use_sandbox_directory:
        server_logger.info("Using entire sandbox_files directory for analysis")
    else:
        server_logger.info(
            f"Using provided code content, length: {len(code_content) if code_content else 0}"
        )

    # Use code_content in pyright
    result = await analyze_with_pyright(code_content, use_sandbox_directory)
    passed = result[0]
    server_logger.info(f"Pyright analysis result passed:{passed}")

    if not passed:
        server_logger.warning(f"Pyright analysis messages: {result[1]}")

    return result


def read_file(file_path):
    """Helper function to read a file in a separate thread."""
    with open(file_path, "r") as file:
        return file.read()


async def analyze_with_pyright(code_content=None, use_sandbox_directory=True):
    """
    Analyze Python code with pyright directly on the local machine.

    Args:
        code_content (str, optional): Python code as a string. If None and use_sandbox_directory
                                      is False, will only use the sandbox_files directory.
        use_sandbox_directory (bool): Whether to analyze the entire sandbox_files directory
                                      instead of just the provided code.

    Returns:
        tuple: (success, error_message_or_empty_string)
    """
    # Get path to the sandbox_files directory
    sandbox_files_path = pathlib.Path(__file__).parent / "sandbox_files"
    server_logger.info(f"Using sandbox_files path: {sandbox_files_path}")

    # Run the blocking operations in a separate thread
    return await asyncio.to_thread(
        _run_pyright_analysis, code_content, use_sandbox_directory, sandbox_files_path
    )


def _run_pyright_analysis(code_content, use_sandbox_directory, sandbox_files_path):
    """Helper function to run pyright analysis in a separate thread."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        if not use_sandbox_directory and code_content:
            # Create the Python file to analyze
            temp_file_path = temp_dir_path / "output.py"
            with open(temp_file_path, "w", encoding="utf-8") as temp_file:
                temp_file.write(code_content)

            # Create a symlink to the base_strategy directory
            base_strategy_link = temp_dir_path / "base_strategy"
            base_strategy_path = sandbox_files_path / "base_strategy"
            if os.path.exists(base_strategy_path):
                os.symlink(
                    str(base_strategy_path),
                    str(base_strategy_link),
                    target_is_directory=True,
                )
                server_logger.info(
                    f"Created symlink to base_strategy: {base_strategy_link}"
                )

            # Path to analyze
            analyze_path = temp_dir_path
        else:
            # Just analyze the entire sandbox_files directory
            analyze_path = sandbox_files_path

        # Create pyrightconfig.json in the appropriate directory
        config_dir = (
            temp_dir_path if not use_sandbox_directory else sandbox_files_path.parent
        )
        pyright_config = {
            "include": [str(analyze_path)],
            "typeCheckingMode": "basic",
            "useLibraryCodeForTypes": True,
            "reportMissingImports": True,
            "reportMissingTypeStubs": False,
            "pythonVersion": "3.10",
        }

        config_path = config_dir / "pyrightconfig.json"
        with open(config_path, "w", encoding="utf-8") as config_file:
            json.dump(pyright_config, config_file, indent=2)

        server_logger.info(f"Created pyrightconfig.json at: {config_path}")
        server_logger.info(f"Analyzing path: {analyze_path}")

        # Run pyright on the directory
        result = subprocess.run(
            [
                "pyright",
                "--outputjson",
                "--level",
                "error",  # Only report errors, not warnings
                str(analyze_path),
            ],
            capture_output=True,
            text=False,  # We'll handle the output as bytes
        )

        return parse_pyright_output(code_content, result.stdout)


def formatErrorWithContext(error, code_content):
    """Format the error with the context of the code."""

    try:
        if not code_content or error.get("line", -1) < 0:
            return error.get("message", "")

        line_number = error.get("line", 0)
        column = error.get("column", 0)
        filename = error.get("filename", "unknown_file")
        error_message = error.get("message", "")

        # Split the code content into lines
        code_lines = code_content.splitlines()

        # Determine the range of lines to show
        start_line = max(0, line_number - 2)
        end_line = min(len(code_lines), line_number + 2)

        # Build the context lines with line numbers
        context_lines = []
        for i in range(start_line, end_line + 1):
            if i >= len(code_lines):
                break

            line_prefix = f"{i+1}: "  # Python uses 0-based indexing
            code_line = code_lines[i]

            # Mark the error line
            if i == line_number:
                indicator = " " * (len(line_prefix) + column) + "^ Error reported here"
                context_lines.append(f"{line_prefix}{code_line}")
                context_lines.append(indicator)
            else:
                context_lines.append(f"{line_prefix}{code_line}")

        # Format the full error with context
        error_with_context = [
            f"{filename}:{line_number+1}:{column+1} - error: {error_message}",
            "",
            "Code context:",
            *context_lines,
        ]

        return "\n".join(error_with_context)
    except Exception as e:
        # Fallback if something goes wrong
        return f"Error formatting context: {str(e)}\nOriginal error: {error.get('message', '')}"


def parse_pyright_output(
    code_content: str,
    stdout: bytes,
) -> tuple[bool, list[dict[str, Any]]]:
    """
    Parse the JSON output from pyright.

    Args:
        stdout (bytes): The bytes output from the pyright process

    Returns:
        tuple: (success_bool, errors_json_or_error_message)
    """
    try:
        # Parse the JSON output
        output = json.loads(stdout)

        errors: list[dict[str, Any]] = []
        for error in output.get("generalDiagnostics", []):
            if error.get("severity", None) == "error":
                # Get the full path and extract just the filename
                full_path = error.get("file", "")
                filename = os.path.basename(full_path) if full_path else ""

                # Keep only relevant parts of the error
                errorType = {
                    "message": error.get("message", ""),
                    "line": error.get("range", {}).get("start", {}).get("line", -1),
                    "column": error.get("range", {}).get("start", {}).get("character", -1),
                    "filename": filename,
                    "path": full_path,
                }
                errorType["errorWithContext"] = formatErrorWithContext(
                    errorType, code_content
                )
                errors.append(errorType)

        success = len(errors) == 0
        return (success, errors)
    except json.JSONDecodeError:
        return (
            False,
            [{"message": f"Failed to parse Pyright output: {stdout.decode()}"}],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pyright analysis on Python code")
    parser.add_argument(
        "--dir", action="store_true", help="Analyze entire sandbox_files directory"
    )
    args = parser.parse_args()

    if args.dir:
        # Analyze the entire sandbox_files directory
        print("Analyzing entire sandbox_files directory...")
        asyncio.run(analize_code_with_pyright(use_sandbox_directory=True))

    else:
        # Default: analyze output.py file
        print("Testing with output.py file...")
        asyncio.run(analize_code_with_pyright())
