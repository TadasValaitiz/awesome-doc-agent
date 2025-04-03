"""
This is langgraph client test

Langgraph server is deployed on https://awesome-doc-agent-4aea3ef58f0f58c9b577c1f02420ef02.us.langgraph.app
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

async def main():
    # Initialize the client with the server URL and API key
    client = get_client(
        url="https://awesome-doc-agent-4aea3ef58f0f58c9b577c1f02420ef02.us.langgraph.app",
        api_key=os.getenv("LANGSMITH_API_KEY")
    )
    
    # 1. Search assistants
    console.print("\n[bold blue]🔍 Searching for assistants...[/bold blue]")
    assistants = await client.assistants.search()
    console.print(create_table("Available Assistants", assistants, ["assistant_id", "name", "description"]))
    
    if assistants:
        assistant_id = assistants[0]["assistant_id"]
        
        # 2. Get assistant details
        console.print("\n[bold blue]📋 Getting assistant details...[/bold blue]")
        assistant = await client.assistants.get(assistant_id)
        console.print(Panel(format_json(assistant), title="Assistant Details"))
        
        # 3. Get assistant graph
        console.print("\n[bold blue]📊 Getting assistant graph...[/bold blue]")
        graph = await client.assistants.get_graph(assistant_id)
        console.print(Panel(format_json(graph), title="Assistant Graph"))
        
        # 4. Get assistant versions
        console.print("\n[bold blue]🔄 Getting assistant versions...[/bold blue]")
        versions = await client.assistants.get_versions(assistant_id)
        console.print(create_table("Assistant Versions", versions, ["version_id", "created_at", "status"]))
        
        # 5. Search threads
        console.print("\n[bold blue]🔍 Searching for threads...[/bold blue]")
        threads = await client.threads.search()
        console.print(create_table("Available Threads", threads, ["thread_id", "created_at", "status"]))
        
        if threads:
            thread_id = threads[0]["thread_id"]
            
            # 6. Get thread state
            console.print("\n[bold blue]📋 Getting thread state...[/bold blue]")
            state = await client.threads.get_state(thread_id)
            console.print(Panel(format_json(state), title="Thread State"))
            
            # 7. Get thread history
            console.print("\n[bold blue]📜 Getting thread history...[/bold blue]")
            history = await client.threads.get_history(thread_id)
            console.print(Panel(format_json(history), title="Thread History"))
            
            # 8. List thread runs
            console.print("\n[bold blue]🔄 Listing thread runs...[/bold blue]")
            runs = await client.runs.list(thread_id)
            console.print(create_table("Thread Runs", runs, ["run_id", "status", "created_at"]))
            
            if runs:
                run_id = runs[0]["run_id"]
                
                # 9. Get run details
                console.print("\n[bold blue]📋 Getting run details...[/bold blue]")
                run = await client.runs.get(thread_id, run_id)
                console.print(Panel(format_json(run), title="Run Details"))
    
    # 10. Search crons
    console.print("\n[bold blue]⏰ Searching for cron jobs...[/bold blue]")
    crons = await client.crons.search()
    console.print(create_table("Cron Jobs", crons, ["cron_id", "name", "schedule", "status"]))

if __name__ == "__main__":
    asyncio.run(main())
