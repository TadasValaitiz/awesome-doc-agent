import asyncio
import json
import subprocess
import tempfile
import os
import pathlib
from typing import Any, Literal, Optional
import pandas as pd
from strategy_agent.logger import server_logger
from pathlib import Path
import argparse


def get_sandbox_files_path() -> Path:
    """Returns the path to the sandbox_files directory."""
    return pathlib.Path(__file__).parent / "sandbox_files"


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
        output_file_path = get_sandbox_files_path() / "output.py"
        server_logger.info(
            f"[analize_code_with_pyright] Reading output file from: {output_file_path}",
            tags=["pyright"],
        )

        try:
            # Use asyncio.to_thread to read file asynchronously
            code_content = await asyncio.to_thread(read_file, output_file_path)
            server_logger.info(
                f"[analize_code_with_pyright] Successfully read output.py, content length: {len(code_content)}",
                tags=["pyright"],
            )
        except Exception as e:
            server_logger.error(
                f"[analize_code_with_pyright] Error reading output.py: {str(e)}",
                tags=["pyright"],
            )
            return
    elif use_sandbox_directory:
        server_logger.info(
            "[analize_code_with_pyright] Using entire sandbox_files directory for analysis",
            tags=["pyright"],
        )
    else:
        server_logger.info(
            f"[analize_code_with_pyright] Using provided code content, length: {len(code_content) if code_content else 0}",
            tags=["pyright"],
        )

    # Use code_content in pyright
    result = await analyze_with_pyright(code_content, use_sandbox_directory)
    passed = result[0]
    server_logger.info(
        f"[analize_code_with_pyright] Pyright analysis result passed:{passed}",
        tags=["pyright"],
    )

    if not passed:
        server_logger.warning(
            f"[analize_code_with_pyright] Pyright analysis messages: {result[1]}",
            tags=["pyright"],
        )

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
        tuple: (success, list_of_formatted_errors)
    """
    # Get path to the sandbox_files directory
    sandbox_files_path = get_sandbox_files_path()
    server_logger.info(
        f"[analyze_with_pyright] Using sandbox_files path: {sandbox_files_path}",
        tags=["pyright"],
    )

    # Create a unique name for each temp directory based on timestamp to ensure uniqueness
    temp_dir_prefix = f"pyright_analysis_{asyncio.get_event_loop().time()}_"

    # Run the blocking operations in a separate thread
    return await asyncio.to_thread(
        _run_pyright_analysis,
        code_content,
        use_sandbox_directory,
        sandbox_files_path,
        temp_dir_prefix,
    )


def _run_pyright_analysis(
    code_content,
    use_sandbox_directory,
    sandbox_files_path,
    temp_dir_prefix="pyright_temp_",
):
    """Helper function to run pyright analysis in a separate thread."""
    # Create a temporary directory without using context manager so it persists
    temp_dir = tempfile.mkdtemp(prefix=temp_dir_prefix)
    temp_dir_path = Path(temp_dir)

    # Log the temporary directory path for debugging
    server_logger.info(
        f"[_run_pyright_analysis] Created persistent temporary directory at: {temp_dir_path}",
        tags=["pyright", "debug"],
    )

    try:
        analyze_path = temp_dir_path
        output_path = temp_dir_path
        config_dir = temp_dir_path

        # Create a symlink to the strategy_module directory if needed
        if use_sandbox_directory:
            symlink_dst = temp_dir_path / "strategy_module"
            symlink_src = sandbox_files_path / "strategy_module"
            if os.path.exists(symlink_src):
                os.symlink(
                    str(symlink_src),
                    str(symlink_dst),
                    target_is_directory=True,
                )
                server_logger.info(
                    f"[_run_pyright_analysis] Created symlink to strategy_module: {symlink_dst}",
                    tags=["pyright"],
                )

        # Write code content to file if provided
        if code_content:
            output_file_path = output_path / "output.py"

            # Check if file already exists and log it
            if output_file_path.exists():
                server_logger.info(
                    f"[_run_pyright_analysis] Overwriting existing output.py at: {output_file_path}",
                    tags=["pyright"],
                )

            with open(output_file_path, "w", encoding="utf-8") as output_file:
                output_file.write(code_content)

            server_logger.info(
                f"[_run_pyright_analysis] Created/updated output.py at: {output_file_path}",
                tags=["pyright"],
            )

        # Create pyrightconfig.json in the appropriate directory
        pyright_config = {
            "include": ["."],  # Use relative path instead of absolute
            "typeCheckingMode": "basic",
            "useLibraryCodeForTypes": True,
            "reportMissingImports": True,
            "reportMissingTypeStubs": False,
            "pythonVersion": "3.10",
            "extraPaths": ["."],  # Add the temp directory to Python path
        }

        config_path = config_dir / "pyrightconfig.json"
        with open(config_path, "w", encoding="utf-8") as config_file:
            json.dump(pyright_config, config_file, indent=2)

        server_logger.info(
            f"[_run_pyright_analysis] Created pyrightconfig.json at: {config_path}",
            tags=["pyright"],
        )
        server_logger.info(
            f"[_run_pyright_analysis] Analyzing path: {analyze_path}", tags=["pyright"]
        )

        # Run pyright on the directory
        try:
            # Create environment with PYTHONPATH set to the temp directory
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{str(temp_dir_path)}:{env.get('PYTHONPATH', '')}"

            result = subprocess.run(
                [
                    "pyright",
                    "--outputjson",
                    "--level",
                    "error",  # Only report errors, not warnings
                    ".",  # Use relative path instead of absolute path
                ],
                capture_output=True,
                text=False,  # We'll handle the output as bytes
                cwd=temp_dir_path,  # Set working directory explicitly
                env=env,  # Use modified environment with PYTHONPATH
            )

            return parse_pyright_output(analyze_path, result.stdout)
        except Exception as e:
            server_logger.error(
                f"[_run_pyright_analysis] Error running pyright: {str(e)}",
                tags=["pyright", "error"],
            )
            return (False, [{"message": f"Error running pyright: {str(e)}"}])
    except Exception as e:
        server_logger.error(
            f"[_run_pyright_analysis] Error in analysis: {str(e)}",
            tags=["pyright", "error"],
        )
        return (False, [{"message": f"Error in analysis: {str(e)}"}])

    # Note: We intentionally don't clean up the temp directory to allow for debugging
    # To clean up manually, you can remove this directory: {temp_dir_path}


def _run_execution(
    code_content,
    use_sandbox_directory,
    data: Optional[pd.DataFrame],
    sandbox_files_path,
    temp_dir_prefix="pyexec_temp_",
):
    """Helper function to execute Python code in a sandboxed environment.

    Args:
        code_content (str): The Python code to execute
        use_sandbox_directory (bool): Whether to include strategy_module directory in execution
        sandbox_files_path (Path): Path to the sandbox_files directory
        temp_dir_prefix (str): Prefix for the temporary directory name

    Returns:
        tuple: (success, output_or_error_message)
    """
    # Create a temporary directory without using context manager so it persists
    temp_dir = tempfile.mkdtemp(prefix=temp_dir_prefix)
    temp_dir_path = Path(temp_dir)

    # Log the temporary directory path for debugging
    server_logger.info(
        f"[_run_execution] Created persistent temporary directory at: {temp_dir_path}",
        tags=["execution", "debug"],
    )

    try:
        output_path = temp_dir_path
        optimization_dir = output_path / "data" / "optimization"

        # Create a symlink to the strategy_module directory if needed
        if use_sandbox_directory:
            symlink_dst = temp_dir_path / "strategy_module"
            symlink_src = sandbox_files_path / "strategy_module"
            if os.path.exists(symlink_src):
                os.symlink(
                    str(symlink_src),
                    str(symlink_dst),
                    target_is_directory=True,
                )
                server_logger.info(
                    f"[_run_execution] Created symlink to strategy_module: {symlink_dst}",
                    tags=["execution"],
                )

        # Write code content to file
        if data is not None:
            # Create data/optimization directory if it doesn't exist
            optimization_dir.mkdir(parents=True, exist_ok=True)
            server_logger.info(
                f"[_run_execution] Created optimization directory at: {optimization_dir}",
                tags=["execution"],
            )
            data.to_parquet(output_path / "data/optimization/backtest_data.pkl")

        output_file_path = output_path / "output.py"

        # Check if file already exists and log it
        if output_file_path.exists():
            server_logger.info(
                f"[_run_execution] Overwriting existing output.py at: {output_file_path}",
                tags=["execution"],
            )

        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(code_content)

        server_logger.info(
            f"[_run_execution] Created/updated output.py at: {output_file_path}",
            tags=["execution"],
        )

        # Execute the Python code
        try:
            # Create environment with PYTHONPATH set to the temp directory
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{str(temp_dir_path)}:{env.get('PYTHONPATH', '')}"

            server_logger.info(
                f"[_run_execution] Executing output.py in {temp_dir_path}",
                tags=["execution"],
            )

            # Run the Python code with stdout and stderr captured
            result = subprocess.run(
                ["python", "output.py"],
                capture_output=True,
                text=True,
                cwd=temp_dir_path,  # Set working directory explicitly
                env=env,  # Use modified environment with PYTHONPATH
                timeout=20 * 60,  # Add timeout to prevent infinite loops
            )

            # Check if execution was successful
            if result.returncode == 0:
                output = {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "optimization_dir": optimization_dir,
                }
                server_logger.info(
                    f"[_run_execution] Code executed successfully",
                    tags=["execution"],
                    data=output,
                )
                return (True, output)
            else:
                output = {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "message": f"Execution failed with code {result.returncode}",
                    "optimization_dir": optimization_dir,
                }
                server_logger.warning(
                    f"[_run_execution] Code execution failed with return code {result.returncode}",
                    tags=["execution", "error"],
                )
                return (False, output)

        except subprocess.TimeoutExpired:
            server_logger.error(
                f"[_run_execution] Execution timed out after 30 seconds",
                tags=["execution", "error"],
            )
            return (False, {"message": "Execution timed out after 30 seconds"})
        except Exception as e:
            server_logger.error(
                f"[_run_execution] Error executing Python code: {str(e)}",
                tags=["execution", "error"],
            )
            return (False, {"message": f"Error executing Python code: {str(e)}"})
    except Exception as e:
        server_logger.error(
            f"[_run_execution] Error setting up execution environment: {str(e)}",
            tags=["execution", "error"],
        )
        return (False, {"message": f"Error setting up execution environment: {str(e)}"})

    # Note: We intentionally don't clean up the temp directory to allow for debugging
    # To clean up manually, you can remove this directory: {temp_dir_path}


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
                # Create a cleaner error indicator with an arrow
                indicator = " " * (len(line_prefix) + column) + "^ Error here"
                context_lines.append(f"{line_prefix}{code_line}")
                context_lines.append(indicator)
            else:
                context_lines.append(f"{line_prefix}{code_line}")

        # Format the full error with context
        error_with_context = [
            f"{filename}:{line_number+1}:{column+1} error: {error_message}",
            "Code context:",
            *context_lines,
            "---",  # Separator for multiple errors
        ]

        return "\n".join(error_with_context)
    except Exception as e:
        # Fallback if something goes wrong
        return f"Error formatting context: {str(e)}\nOriginal error: {error.get('message', '')}"


def parse_pyright_output(
    path: Path,
    stdout: bytes,
) -> tuple[bool, list[dict[str, Any]]]:
    """
    Parse the JSON output from pyright.

    Args:
        path (Path): The path to the code being analyzed
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
                    "column": error.get("range", {})
                    .get("start", {})
                    .get("character", -1),
                    "filename": filename,
                    "path": full_path,
                }

                # Read file content from error path if available
                code_content = None
                if full_path and os.path.exists(full_path):
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            code_content = f.read()
                    except Exception as e:
                        server_logger.error(
                            f"[parse_pyright_output] Error reading file {full_path}: {str(e)}",
                            tags=["pyright", "error"],
                        )

                errorType["errorWithContext"] = formatErrorWithContext(
                    errorType, code_content
                )
                errors.append(errorType)

        success = len(errors) == 0
        return (success, errors)
    except json.JSONDecodeError:
        return (
            False,
            [
                {
                    "message": f"[parse_pyright_output] Failed to parse Pyright output: {stdout.decode()}",
                    "tags": ["pyright"],
                }
            ],
        )


async def execute_code(
    code_content=None, use_sandbox_directory=True, data: Optional[pd.DataFrame] = None
):
    """
    Execute Python code in a sandboxed environment.

    Args:
        code_content (str): Python code as a string to execute.
        use_sandbox_directory (bool): Whether to include the strategy_module directory
                                      in the execution environment.

    Returns:
        tuple: (success, execution_result)
    """
    # Get path to the sandbox_files directory
    sandbox_files_path = get_sandbox_files_path()
    server_logger.info(
        f"[execute_code] Using sandbox_files path: {sandbox_files_path}",
        tags=["execution"],
    )

    # Create a unique name for each temp directory based on timestamp to ensure uniqueness
    temp_dir_prefix = f"pyexec_{asyncio.get_event_loop().time()}_"

    # Run the blocking operations in a separate thread
    return await asyncio.to_thread(
        _run_execution,
        code_content,
        use_sandbox_directory,
        data,
        sandbox_files_path,
        temp_dir_prefix,
    )


async def run_code(
    code_content=None, use_sandbox_directory=True, data: Optional[pd.DataFrame] = None
):
    """
    Execute Python code in a sandbox and return the results.

    Args:
        code_content (str, optional): Python code to execute. If None, will read from output.py file
                                       in the sandbox_files directory.
        use_sandbox_directory (bool): Whether to include the strategy_module directory in execution.

    Returns:
        tuple: (success, execution_results)
    """
    if code_content is None:
        # Read output.py using filesystem function
        output_file_path = get_sandbox_files_path() / "output.py"
        server_logger.info(
            f"[run_code] Reading output file from: {output_file_path}",
            tags=["execution"],
        )

        try:
            # Use asyncio.to_thread to read file asynchronously
            code_content = await asyncio.to_thread(read_file, output_file_path)
            server_logger.info(
                f"[run_code] Successfully read output.py, content length: {len(code_content)}",
                tags=["execution"],
            )
        except Exception as e:
            server_logger.error(
                f"[run_code] Error reading output.py: {str(e)}",
                tags=["execution"],
            )
            return (False, {"message": f"Error reading output.py: {str(e)}"})

    # Execute the code
    result = await execute_code(code_content, use_sandbox_directory, data)
    success = result[0]

    server_logger.info(
        f"[run_code] Execution result success: {success}",
        tags=["execution"],
    )

    if not success:
        server_logger.warning(
            f"[run_code] Execution failed: {result[1].get('message', '')}",
            tags=["execution"],
        )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Python code analysis or execution in a sandbox"
    )
    parser.add_argument(
        "--dir", action="store_true", help="Analyze entire sandbox_files directory"
    )
    parser.add_argument(
        "--execute", action="store_true", help="Execute code instead of analyzing it"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to file to execute (defaults to sandbox_files/output.py)",
        default=None,
    )
    args = parser.parse_args()

    if args.execute:
        # Execute code mode
        code_content = None
        if args.file:
            print(f"Executing code from file: {args.file}")
            with open(args.file, "r") as file:
                code_content = file.read()
        else:
            # Default: execute output.py file content
            output_file_path = get_sandbox_files_path() / "output.py"
            print(f"Executing code from output.py file at {output_file_path}...")
            with open(output_file_path, "r") as file:
                code_content = file.read()

        result = asyncio.run(run_code(code_content, use_sandbox_directory=True))
        success, output = result

        print(f"\nExecution {'succeeded' if success else 'failed'}")
        if success:
            print("\nSTDOUT:")
            print(output.get("stdout", ""))
            if output.get("stderr"):
                print("\nSTDERR:")
                print(output.get("stderr"))
        else:
            print("\nError:", output.get("message", "Unknown error"))
            if "stdout" in output:
                print("\nSTDOUT:")
                print(output.get("stdout", ""))
            if "stderr" in output:
                print("\nSTDERR:")
                print(output.get("stderr", ""))

    elif args.dir:
        # Analyze the entire sandbox_files directory
        print("Analyzing entire sandbox_files directory...")
        asyncio.run(analize_code_with_pyright(use_sandbox_directory=True))
    else:
        # Default: analyze output.py file
        print("Testing with output.py file...")
        asyncio.run(analize_code_with_pyright())
