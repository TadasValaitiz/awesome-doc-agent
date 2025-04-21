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

from strategy_agent import prompts
from strategy_agent.utils import init_model, reciprocal_rank_fusion, take_top_k
from strategy_agent.vector_db import AsyncVectorDB, VectorDB
from strategy_agent.database import StrategyDb, TradingStrategyDefinition


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
    config: Annotated[RunnableConfig, InjectedToolArg]
):
    """Search for trading strategies, trading indicators, entry and exit conditions

    This function is particularly useful for trading ideas exploration, trading ideas brainstorming, trading indicators exploration.
    """

    configuration = Configuration.from_runnable_config(config)
    llm = init_model(config)

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


TOOLS: List[Callable[..., Any]] = [search_trading_ideas]
