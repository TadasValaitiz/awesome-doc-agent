"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from functools import partial
from typing import Any, Callable, List, Optional, cast

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated

from strategy_agent.state import StrategyAgentState
from strategy_agent.configuration import Configuration
from langgraph.types import Command
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph

from strategy_agent.sandbox.pyright import analize_code_with_pyright
from strategy_agent.coding_extractors import extract_code_from_markdown_code_blocks
from strategy_agent.state import StrategyAgentState
from strategy_agent.logger import server_logger
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

from strategy_agent import prompts
from strategy_agent.utils import init_model, reciprocal_rank_fusion, take_top_k
from strategy_agent.vector_db import AsyncVectorDB, VectorDB
from strategy_agent.database import StrategyDb, TradingStrategyDefinition
from strategy_agent.logger import server_logger


async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


@tool
async def search_trading_ideas(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    *,
    config: Annotated[RunnableConfig, InjectedToolArg],
):
    """Search for trading strategies, trading indicators, entry and exit conditions

    This function is particularly useful for trading ideas exploration, trading ideas brainstorming, trading indicators exploration.
    """

    configuration = Configuration.from_runnable_config(config)
    llm = init_model(configuration.chat_model)

    rag_fusion_prompt = ChatPromptTemplate.from_template(prompts.rag_fusion)

    def prepare_prompt(question: str):
        return {"question": question, "context": ""}

    vector_db = AsyncVectorDB()

    def load_strategies(ids: List[int]) -> List[str]:
        strategy_db = StrategyDb()
        return list(
            map(
                lambda x: cast(TradingStrategyDefinition, x["strategy"]).context_str(),
                strategy_db.list_strategies_by_ids(ids),
            )
        )

    rag_fusion_chain = (
        prepare_prompt
        | rag_fusion_prompt
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
        | (await vector_db.strategy_retriever()).map()
        | reciprocal_rank_fusion
        | partial(take_top_k, k=5)
        | load_strategies
    )

    result = await rag_fusion_chain.ainvoke(query)

    return Command(
        update={
            "messages": [
                ToolMessage(content="\n".join(result), tool_call_id=tool_call_id)
            ]
        }
    )


@tool
async def code_check_with_pyright(
    code_markdown: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    *,
    config: Annotated[RunnableConfig, InjectedToolArg],
):
    """Validate and check Python code for errors and quality issues

    Use this tool to ensure your Python code is error-free before executing it. It performs static analysis using pyright
    to catch type errors, undefined variables, import issues, and other common programming mistakes.

    Always run this tool when generating Python code to:
    - Find and fix bugs before execution
    - Ensure code follows best practices
    - Verify type correctness
    - Identify potential runtime errors

    Simply pass your Python code wrapped in markdown code blocks and receive detailed feedback on any issues found.
    """
    code = extract_code_from_markdown_code_blocks(code_markdown)

    result = await analize_code_with_pyright(code)

    # After code execution, download and log any output files

    if result and not result[0]:
        errors = [error["errorWithContext"] for error in result[1]]
        prettyErrors = "\n\n".join(errors)

        content = f"""
I ran pyright and found some problems with the code you generated:

```python
{code}
```

Errors:
{prettyErrors}

Instructions:
Try to fix it. Make sure to regenerate the entire code snippet.
"""

        message = ToolMessage(content=content, tool_call_id=tool_call_id)
        return Command(
            update={
                "messages": [message],
                "code_approved": False,
                "code_feedback": errors,
                "code_output": code,
            }
        )

    return Command(
        update={
            "code_approved": True,
            "code_feedback": None,
            "code_output": code,
        }
    )


PLANNER_TOOLS: List[Callable[..., Any]] = [search_trading_ideas]
CODER_TOOLS: List[Callable[..., Any]] = [code_check_with_pyright]
