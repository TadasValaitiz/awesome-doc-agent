"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated

from strategy_agent.configuration import Configuration
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from strategy_agent import prompts
from strategy_agent.utils import init_model


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


async def search_trading_ideas(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
):
    """Search for trading strategies, trading indicators, entry and exit conditions

    This function is particularly useful for trading ideas exploration, trading ideas brainstorming, trading indicators exploration.
    """

    configuration = Configuration.from_runnable_config(config)
    llm = init_model(config)

    rag_fusion_prompt = ChatPromptTemplate.from_template(prompts.rag_fusion)

    def prepare_prompt(question: str):
        return {"question": question, "context": ""}

    rag_fusion_chain = prepare_prompt | rag_fusion_prompt | llm | StrOutputParser()

    # rag_fusion_chain = prepare_prompt
    #         | rag_fusion_prompt
    #         | RunnablePassthrough(
    #             lambda x: stream_handler.step_update(step="Querying Rag")
    #         )
    #         | llm
    #         | StrOutputParser()
    #         | (lambda x: x.split("\n"))
    #         | self.vector_db.strategy_retriever().map()
    #         | reciprocal_rank_fusion
    #         | partial(take_top_k, k=5)
    #         | self.load_strategies

    # )
    return await rag_fusion_chain.ainvoke(query)


TOOLS: List[Callable[..., Any]] = [search_trading_ideas]
