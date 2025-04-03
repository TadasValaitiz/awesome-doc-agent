#!/usr/bin/env python3

"""
This script streams the output of a LangGraph run in real-time.
It accepts a thread ID and run ID, then streams the output of that run
in a beautiful format as it arrives.
"""

from langgraph_sdk import get_client
import asyncio
from dotenv import load_dotenv
import os
import json
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
from rich import print as rprint
from datetime import datetime
from typing import Dict, Any, List

# Load environment variables
load_dotenv()

# Initialize rich console
console = Console()

def format_json(data):
    """Format JSON data with syntax highlighting"""
    return Syntax(json.dumps(data, indent=2), "json", theme="monokai")

def create_stream_table(stream_parts: List[Dict[str, Any]]) -> Table:
    """Create a table to display stream parts"""
    table = Table(title="Stream Parts", show_header=True, header_style="bold magenta")
    table.add_column("Type", style="cyan")
    table.add_column("Content", style="green")
    table.add_column("Timestamp", style="yellow")
    
    for part in stream_parts:
        # Convert StreamPart to dict for display
        part_dict = {
            "type": getattr(part, "type", "unknown"),
            "content": getattr(part, "content", ""),
            "timestamp": getattr(part, "timestamp", datetime.now().isoformat())
        }
        
        part_type = part_dict["type"]
        content = part_dict["content"]
        timestamp = part_dict["timestamp"]
        
        # Truncate content if too long
        if isinstance(content, str) and len(content) > 100:
            content = content[:97] + "..."
        
        table.add_row(part_type, str(content), timestamp)
    
    return table

async def stream_run(thread_id: str, run_id: str):
    """Stream a run in real-time"""
    client = get_client(
        url="https://awesome-doc-agent-4aea3ef58f0f58c9b577c1f02420ef02.us.langgraph.app",
        api_key=os.getenv("LANGSMITH_API_KEY")
    )

    try:
        # 1. Get Thread Information
        console.print("\n[bold blue]🧵 Fetching thread information...[/bold blue]")
        thread = await client.threads.get(thread_id)
        console.print(Panel(format_json(thread), title="Thread Information"))
        
        # 2. Get Run Information
        console.print("\n[bold blue]🏃 Fetching run information...[/bold blue]")
        run = await client.runs.get(thread_id, run_id)
        console.print(Panel(format_json(run), title="Run Information"))
        
        # 3. Stream the run
        console.print("\n[bold blue]🚀 Starting stream...[/bold blue]")
        
        # Create a layout for live updates
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3)
        )
        
        # Initialize stream parts list
        stream_parts = []
        
        # Use Live to update the display in real-time
        with Live(layout, refresh_per_second=4) as live:
            # Update header
            layout["header"].update(Panel("[bold blue]Streaming Run Output[/bold blue]"))
            
            # Get the stream - no await needed as it returns an AsyncIterator directly
            stream = client.runs.stream(thread_id, run_id)
            
            # Process each stream part as it arrives
            async for part in stream:
                # Add the part to our list
                stream_parts.append(part)
                
                # Update the body with the current table
                layout["body"].update(create_stream_table(stream_parts))
                
                # Update footer with the latest part
                part_type = getattr(part, "type", "unknown")
                part_content = getattr(part, "content", "")
                
                if part_type == "content":
                    layout["footer"].update(Panel(f"[bold green]Latest:[/bold green] {part_content}"))
                elif part_type == "error":
                    layout["footer"].update(Panel(f"[bold red]Error:[/bold red] {part_content}"))
                else:
                    layout["footer"].update(Panel(f"[bold yellow]Status:[/bold yellow] {part_type}"))
        
        # After streaming is complete, show a summary
        console.print("\n[bold green]✅ Streaming completed[/bold green]")
        console.print(f"Total stream parts: {len(stream_parts)}")
        
        # Get the final run state
        run_details = await client.runs.get(thread_id, run_id)
        console.print(Panel(format_json(run_details), title="Final Run Details"))

    except Exception as e:
        console.print(f"[red]Error occurred while streaming run: {str(e)}[/red]")

def main():
    parser = argparse.ArgumentParser(description='Stream a run in real-time')
    parser.add_argument('thread_id', type=str, help='Thread ID')
    parser.add_argument('run_id', type=str, help='Run ID to stream')
    args = parser.parse_args()

    asyncio.run(stream_run(args.thread_id, args.run_id))

if __name__ == "__main__":
    main() 