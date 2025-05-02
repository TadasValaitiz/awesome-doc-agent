from e2b_code_interpreter import Sandbox, AsyncSandbox
from strategy_agent.logger import server_logger
import os
import pathlib
import asyncio
import aiofiles

_GLOBAL_SANDBOX = None


async def async_get_or_create_sandbox():
    global _GLOBAL_SANDBOX
    if _GLOBAL_SANDBOX is None:
        sandbox = await AsyncSandbox.create("OpenEvalsPython", timeout=30)
        _GLOBAL_SANDBOX = sandbox
        server_logger.info(f"Created sandbox sandboxId: {sandbox.sandbox_id}")

        sandbox_info = await sandbox.get_info()
        server_logger.info(f"Sandbox info: {sandbox_info}")
        await sandbox.commands.run(
            "pip install pandas numpy backtesting==0.6.1 ta==0.11.0 uuid pytz"
        )
        server_logger.info(
            "Installed dependencies pandas numpy backtesting==0.6.1 ta==0.11.0 uuid pytz"
        )
        # Upload helper files to the sandbox
        await upload_sandbox_helper_files(sandbox)
        server_logger.info(
            f"Uploaded sandbox helper files sandboxId: {sandbox.sandbox_id}"
        )

    return _GLOBAL_SANDBOX


async def upload_sandbox_helper_files(sandbox):
    """Upload helper files to the sandbox from the sandbox_files directory."""
    try:
        sandbox_files_dir = (
            pathlib.Path(__file__).parent / "sandbox_files/base_strategy"
        )

        # Use async version to list files in directory
        file_paths = await asyncio.to_thread(list, sandbox_files_dir.glob("*.py"))

        server_logger.info(f"Uploading sandbox files: {file_paths}")

        # Create a proper package structure in the sandbox
        try:
            await sandbox.files.make_dir("/code/base_strategy")
        except Exception as e:
            server_logger.info(f"Directory may already exist: {str(e)}")

        for file_path in file_paths:
            file_name = file_path.name

            # Use async file reading
            async with aiofiles.open(file_path, "r") as f:
                file_content = await f.read()

            # Upload to the sandbox /code/base_strategy directory
            await sandbox.files.write(f"/code/base_strategy/{file_name}", file_content)
            server_logger.info(
                f"Uploaded {file_name} to sandbox /code/base_strategy directory"
            )

        # Create setup.py file for the base_strategy package
        setup_py_content = """
from setuptools import setup

setup(
    name="base_strategy",
    version="0.1.0",
    packages=["base_strategy"],
    install_requires=[
        "pandas",
        "numpy",
        "backtesting==0.6.1",
        "ta==0.11.0",
        "uuid",
        "pytz"
    ],
)
"""
        await sandbox.files.write("/code/setup.py", setup_py_content)
        server_logger.info("Created setup.py file in sandbox /code directory")

        # Create pyrightconfig.json file in the root directory
        pyright_config_content = """
{
    "include": [
        "/code",
        "/code/base_strategy"
    ],
    "exclude": [
        "**/node_modules",
        "**/__pycache__"
    ],
    "ignore": [],
    "defineConstant": {},
    "typeCheckingMode": "basic",
    "useLibraryCodeForTypes": true,
    "reportMissingImports": true,
    "reportMissingTypeStubs": false,
    "pythonVersion": "3.10",
    "pythonPlatform": "Linux",
    "extraPaths": [
        "/code",
        "/code/base_strategy"
    ],
    "executionEnvironments": [
        {
            "root": "/code",
            "extraPaths": [
                "/code",
                "/code/base_strategy"
            ]
        }
    ]
}
"""
        # Place pyrightconfig.json in the root directory
        await sandbox.files.write("/pyrightconfig.json", pyright_config_content)
        server_logger.info("Created pyrightconfig.json file in sandbox root directory")

        # Install the package in development mode
        await sandbox.commands.run("cd /code && pip install -e .")
        server_logger.info("Installed base_strategy package in sandbox")

    except Exception as e:
        server_logger.error(f"Error uploading sandbox files: {str(e)}")


async def download_file_from_sandbox(sandbox, file_path):
    """Download a file from the sandbox."""
    try:
        # Use the correct API method as per e2b docs
        server_logger.info(f"Downloading file from sandbox: {file_path}")
        content = await sandbox.files.read(file_path)
        server_logger.info(f"Successfully downloaded {file_path} from sandbox")
        return content
    except Exception as e:
        server_logger.error(
            f"Error downloading file {file_path} from sandbox: {str(e)}"
        )
        return None


async def download_all_files_from_sandbox(sandbox, file_extensions=None):
    """Download all files from the sandbox with specified extensions.

    Args:
        sandbox: The e2b sandbox instance
        file_extensions: A tuple of file extensions to download.
                         If None, defaults to (.csv, .json, .txt, .log)

    Returns:
        A dictionary mapping file paths to their contents
    """
    if file_extensions is None:
        file_extensions = (".csv", ".json", ".txt", ".log", ".py")

    result = {}

    try:
        # Get list of files in sandbox /code directory
        root_dir_list = await sandbox.files.list("/")
        server_logger.debug(
            f"Files in sandbox root directory: {[entry for entry in root_dir_list]}"
        )
        files_list = await sandbox.files.list("/code")
        server_logger.debug(
            f"Files in sandbox /code directory: {[entry for entry in files_list]}"
        )
        base_list = await sandbox.files.list("/code/base_strategy")
        server_logger.debug(
            f"Files in sandbox /code/base_strategy directory: {[entry for entry in base_list]}"
        )

        # Look for output files generated during execution
        for entry in files_list:
            # Extract the path of each file
            if hasattr(entry, "path"):
                file_path = entry.path
            elif isinstance(entry, dict) and "path" in entry:
                file_path = entry["path"]
            else:
                continue

            if isinstance(file_path, str) and file_path.endswith(file_extensions):
                server_logger.info(f"Found file to download: {file_path}")
                file_content = await download_file_from_sandbox(sandbox, file_path)
                if file_content:
                    result[file_path] = file_content
                    server_logger.info(f"Contents of {file_path}:\n{len(file_content)}")

        return result
    except Exception as e:
        server_logger.error(f"Error downloading files from sandbox: {str(e)}")
        return result
