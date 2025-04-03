"""
This script demonstrates the usage of LangGraph storage API
with examples of storing and retrieving states, managing checkpoints,
and working with different storage operations.
"""

from langgraph_sdk import get_client
import asyncio
from dotenv import load_dotenv
import os
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint
from datetime import datetime
import time

# Load environment variables
load_dotenv()

# Initialize rich console
console = Console()

def format_json(data):
    """Format JSON data with syntax highlighting"""
    return Syntax(json.dumps(data, indent=2), "json", theme="monokai")

def create_table(title, data, columns):
    """Create a rich table with the given data"""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    for col in columns:
        table.add_column(col)
    
    if isinstance(data, list):
        for row in data:
            table.add_row(*[str(row.get(col, "")) for col in columns])
    else:
        table.add_row(*[str(data.get(col, "")) for col in columns])
    
    return table

async def wait_for_run_completion(client, thread_id, run_id, timeout=30):
    """Wait for a run to complete with timeout"""
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            console.print("[red]Timeout waiting for run to complete[/red]")
            return False
            
        run_state = await client.runs.get(thread_id, run_id)
        if run_state["status"] in ["success", "error", "cancelled"]:
            return run_state["status"] == "success"
            
        await asyncio.sleep(1)

async def store_example_state():
    """Example of storing and managing states"""
    client = get_client(
        url="https://awesome-doc-agent-4aea3ef58f0f58c9b577c1f02420ef02.us.langgraph.app",
        api_key=os.getenv("LANGSMITH_API_KEY")
    )
    
    # 1. Create a new thread for our storage example
    console.print("\n[bold blue]🧵 Creating a new thread for storage example...[/bold blue]")
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    console.print(Panel(format_json(thread), title="New Thread"))

    # 2. Initialize thread with state
    console.print("\n[bold blue]💾 Initializing thread state...[/bold blue]")
    initial_state = {
        "conversation_context": {
            "topic": "Storage API Example",
            "started_at": datetime.now().isoformat(),
            "participants": ["user", "assistant"]
        },
        "message_count": 0,
        "last_update": datetime.now().isoformat()
    }
    
    # Update thread with initial state
    await client.threads.update(
        thread_id,
        metadata={"state": initial_state}
    )
    console.print(Panel(format_json(initial_state), title="Initial State Stored"))

    # 3. Get current thread
    console.print("\n[bold blue]📖 Reading current thread...[/bold blue]")
    current_thread = await client.threads.get(thread_id)
    console.print(Panel(format_json(current_thread), title="Current Thread"))

    # 4. Update thread metadata
    console.print("\n[bold blue]✏️ Updating thread metadata...[/bold blue]")
    updated_state = {
        **initial_state,
        "message_count": 1,
        "last_message": "Hello, this is a test message",
        "last_update": datetime.now().isoformat()
    }
    await client.threads.update(
        thread_id,
        metadata={"state": updated_state}
    )
    console.print(Panel(format_json(updated_state), title="Updated State"))

    # 5. Send a message
    console.print("\n[bold blue]💬 Sending a message...[/bold blue]")
    input_data = {
        "messages": [{
            "role": "user",
            "content": "This is a test message for storage example"
        }]
    }
    
    # Create a run with the message
    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id="generic_chat_agent",
        input=input_data
    )
    console.print(Panel(format_json(run), title="New Run"))
    
    # 6. Wait for run to complete
    console.print("\n[bold blue]⏳ Waiting for run to complete...[/bold blue]")
    success = await wait_for_run_completion(client, thread_id, run["run_id"])
    
    if success:
        console.print("[green]Run completed successfully[/green]")
    else:
        console.print("[red]Run failed or timed out[/red]")
    
    # 7. Get thread history
    console.print("\n[bold blue]📋 Getting thread history...[/bold blue]")
    history = await client.threads.get_history(thread_id)
    console.print(Panel(format_json(history), title="Thread History"))

    # 8. Get run state
    console.print("\n[bold blue]📊 Getting final run state...[/bold blue]")
    run_state = await client.runs.get(thread_id, run["run_id"])
    console.print(Panel(format_json(run_state), title="Final Run State"))

    # 9. Final thread state
    console.print("\n[bold blue]✅ Final thread state...[/bold blue]")
    final_thread = await client.threads.get(thread_id)
    console.print(Panel(format_json(final_thread), title="Final Thread State"))

if __name__ == "__main__":
    asyncio.run(store_example_state()) 