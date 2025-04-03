#!/usr/bin/env python3
"""
This script creates a new thread and run using LangGraph SDK.
It outputs the thread_id and run_id in a format suitable for piping to thread-stream.py.
"""

from langgraph_sdk import get_client, get_sync_client
import asyncio
from dotenv import load_dotenv
import os
import argparse
from rich.console import Console
from typing import Optional

# Load environment variables
load_dotenv()

# Initialize rich console
console = Console()


def create_thread_and_run(message: Optional[str] = None, verbose: bool = False):
    """Create a new thread and run, then output the IDs"""
    client = get_sync_client(
        url="https://awesome-doc-agent-4aea3ef58f0f58c9b577c1f02420ef02.us.langgraph.app",
        api_key=os.getenv("LANGSMITH_API_KEY"),
    )

    try:
        # 1. Create a new thread
        if verbose:
            console.print("\n[bold blue]🧵 Creating new thread...[/bold blue]")

        # Create thread
        thread = client.threads.create(
            graph_id="generic_chat_agent",
            metadata={"configurable": {"model": "openai/gpt-4o-mini"}},
        )
        thread_id = thread["thread_id"]

        if verbose:
            console.print(f"Thread created with ID: {thread_id}")

        # 2. Create a new run
        if verbose:
            console.print("\n[bold blue]🏃 Creating new run...[/bold blue]")

        # Prepare the input for the run
        if message:
            message = "Write me a long poem about a cat, 2 pages"

        input = {
            "messages": [{"role": "user", "content": message}],
        }

        run = client.runs.create(
            thread_id=thread_id,
            assistant_id="generic_chat_agent",
            input=input,
            config={"configurable": {"model": "openai/gpt-4o-mini"}},
        )
        run_id = run["run_id"]

        if verbose:
            console.print(f"Run created with ID: {run_id}")

        # 3. Output the IDs in a format suitable for piping
        # Format: thread_id run_id
        print(f"{thread_id} {run_id}")

        return thread_id, run_id

    except Exception as e:
        console.print(
            f"[red]Error occurred while creating thread and run: {str(e)}[/red]"
        )
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Create a new thread and run")
    parser.add_argument(
        "--message",
        default="Write me a long poem about a cat",
        type=str,
        help="Optional message to send with the run",
    )
    parser.add_argument(
        "--verbose", default=False, action="store_true", help="Show detailed output"
    )
    args = parser.parse_args()

    create_thread_and_run(args.message, args.verbose)


if __name__ == "__main__":
    main()
