#!/usr/bin/env python3
import subprocess
import os
import sys
import json
import tempfile
import asyncio
from pathlib import Path

"""
Test program to run code execution in a subprocess and observe behavior.
"""


def run_command_direct(cmd_parts, **kwargs):
    """Run command directly without shell"""
    print(f"\n===== Running with shell=False: {' '.join(cmd_parts)} =====")
    print(f"Extra kwargs: {kwargs}")
    result = subprocess.run(
        cmd_parts, shell=False, capture_output=True, text=True, **kwargs
    )
    print(f"Exit code: {result.returncode}")
    print(f"Stdout:\n{result.stdout}")
    print(f"Stderr:\n{result.stderr}")
    return result


def print_env_info(target_dir):
    """Print relevant environment information"""
    print("\n===== Environment Information =====")
    print(f"Current working directory: {os.getcwd()}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"PATH: {os.environ.get('PATH', 'Not set')}")

    # Check if target directory exists
    print(f"Target directory exists: {os.path.exists(target_dir)}")
    if os.path.exists(target_dir):
        print(f"Contents of target directory: {os.listdir(target_dir)}")


async def execute_code(target_dir):
    """Execute Python code in the target directory"""
    print(f"\n===== Executing code in {target_dir} =====")

    # Create environment with PYTHONPATH set to the target directory
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{target_dir}:{env.get('PYTHONPATH', '')}"

    # Run the Python code with stdout and stderr captured using asyncio subprocess
    try:
        # Run the process asynchronously
        process = await asyncio.create_subprocess_exec(
            "python",
            "output.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=target_dir,  # Set working directory explicitly
            env=env,  # Use modified environment with PYTHONPATH
        )

        # Add a timeout using asyncio.wait_for
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=20 * 60  # 20 minutes timeout
            )

            # Decode the output
            stdout_str = stdout.decode("utf-8")
            stderr_str = stderr.decode("utf-8")

            print(f"Exit code: {process.returncode}")
            print(f"Stdout:\n{stdout_str}")
            print(f"Stderr:\n{stderr_str}")

            return {
                "returncode": process.returncode,
                "stdout": stdout_str,
                "stderr": stderr_str,
            }
        except asyncio.TimeoutError:
            # Cancel the process if it times out
            try:
                process.kill()
            except ProcessLookupError:
                pass
            print("Execution timed out after 20 minutes")
            return {
                "error": "Timeout",
                "message": "Execution timed out after 20 minutes",
            }

    except Exception as e:
        print(f"Error executing Python code: {str(e)}")
        return {"error": "Exception", "message": str(e)}


async def main():
    # Set target directory from command-line argument or use default
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = (
            "/var/folders/cd/_cr_2vnn6g9bmy70y43p4hlh0000gn/T/pyexec_457231.96_nxkqjllg"
        )

    # Get current environment info
    print_env_info(target_dir)

    # Execute the code
    result = await execute_code(target_dir)

    # Print a summary of execution
    print("\n===== Execution Summary =====")
    if "error" in result:
        print(f"Error: {result['error']}")
        print(f"Message: {result['message']}")
    else:
        status = "Success" if result["returncode"] == 0 else "Failed"
        print(f"Status: {status}")
        print(f"Return code: {result['returncode']}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
