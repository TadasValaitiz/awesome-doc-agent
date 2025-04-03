import asyncio
import json
from doc_agent import graph
from doc_agent.state import InputState
from doc_agent.utils import panda_sample
import os
from dotenv import load_dotenv
import pathlib


async def main():
    # Get the absolute path to the server directory
    server_dir = pathlib.Path(__file__).parent.parent.parent
    env_path = server_dir / ".env"

    # Load environment variables from the .env file
    load_dotenv(dotenv_path=env_path, override=True)

    # graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
    sample = panda_sample()
    async for type,event in graph.astream(
        input=InputState(original=sample),
        config={"configurable": {"model": "openai/gpt-4o-mini"}},
        stream_mode=["values"],
    ):
        print("--------------------------------")
        print(f"Event type: {type}", event)


if __name__ == "__main__":
    asyncio.run(main())
