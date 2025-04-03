"""
This script helps debug thread information using LangGraph SDK.
It fetches comprehensive information about a thread including its runs,
storage, state, assistant info, and metadata.
"""

from langgraph_sdk import get_client
import asyncio
from dotenv import load_dotenv
import os
import json
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint
from datetime import datetime
from common.types import DocumentMetadata
from typing import Optional, Dict, Any, TypedDict, cast

# Load environment variables
load_dotenv()

# Initialize rich console
console = Console()

class ThreadMetadata(TypedDict, total=False):
    user_id: str
    document_metadata: Dict[str, Any]

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

def deserialize_document_metadata(metadata: Dict[str, Any]) -> Optional[DocumentMetadata]:
    """Deserialize document metadata from thread metadata"""
    if not metadata or "document_metadata" not in metadata:
        return None
        
    try:
        return DocumentMetadata.from_dict(metadata["document_metadata"])
    except Exception as e:
        console.print(f"[red]Error deserializing document metadata: {str(e)}[/red]")
        return None

async def debug_thread(thread_id: str):
    """Debug thread information using thread_id"""
    client = get_client(
        url="https://awesome-doc-agent-4aea3ef58f0f58c9b577c1f02420ef02.us.langgraph.app",
        api_key=os.getenv("LANGSMITH_API_KEY")
    )

    try:
        # 1. Get Thread Information
        console.print("\n[bold blue]🧵 Fetching thread information...[/bold blue]")
        thread = await client.threads.get(thread_id)
        console.print(Panel(format_json(thread), title="Thread Information"))

        # 2. Get Thread Metadata and State
        console.print("\n[bold blue]📊 Thread Metadata and State[/bold blue]")
        if thread.get("metadata"):
            metadata = cast(Dict[str, Any], thread["metadata"])
            console.print(Panel(format_json(metadata), title="Thread Metadata"))
            
            # Try to deserialize and verify DocumentMetadata
            doc_metadata = deserialize_document_metadata(metadata)
            if doc_metadata:
                console.print("\n[bold green]✅ Successfully deserialized DocumentMetadata[/bold green]")
                console.print(f"File name: {doc_metadata.file_name}")
                console.print(f"Number of rows: {doc_metadata.num_rows}")
                console.print(f"Columns: {', '.join(str(col) for col in doc_metadata.columns)}")
                
                # Show first few rows of the DataFrame
                console.print("\nFirst 5 rows of the DataFrame:")
                console.print(doc_metadata.df.head().to_string())
            else:
                console.print("[yellow]No DocumentMetadata found in thread metadata[/yellow]")
        else:
            console.print("[yellow]No metadata found for this thread[/yellow]")

        # 3. Get Thread History
        console.print("\n[bold blue]📋 Thread History[/bold blue]")
        history = await client.threads.get_history(thread_id)
        console.print(Panel(format_json(history), title="Thread History"))

        # 4. Get All Runs for the Thread
        console.print("\n[bold blue]🏃 Thread Runs[/bold blue]")
        runs = await client.runs.list(thread_id)
        if runs:
            for run in runs:
                console.print(Panel(format_json(run), title=f"Run {run.get('run_id')}"))
                
                # Get detailed run information
                run_details = await client.runs.get(thread_id, run["run_id"])
                console.print(Panel(format_json(run_details), title=f"Run Details {run['run_id']}"))
        else:
            console.print("[yellow]No runs found for this thread[/yellow]")

    except Exception as e:
        console.print(f"[red]Error occurred while debugging thread: {str(e)}[/red]")

def main():
    parser = argparse.ArgumentParser(description='Debug thread information using thread_id')
    parser.add_argument('thread_id', type=str, help='Thread ID to debug')
    args = parser.parse_args()

    asyncio.run(debug_thread(args.thread_id))

if __name__ == "__main__":
    main()
