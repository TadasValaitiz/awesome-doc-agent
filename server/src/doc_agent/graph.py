"""Define a generic chat agent agent."""

import functools
from typing import Optional
import asyncio

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    BaseMessageChunk,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.output_parsers import PydanticOutputParser

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from pydantic import BaseModel, Field

from doc_agent import prompts
from doc_agent.configuration import Configuration
from doc_agent.state import (
    MissingValues,
    Outliers,
    Duplicates,
    Inconsistencies,
    State,
    InputState,
    OutputState,
)
from doc_agent.utils import init_model, pandas_to_markdown


async def analize_missing_values(
    state: State, *, config: Optional[RunnableConfig] = None
):
    configuration = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_template(prompts.MISSING_VALUES_PROMPT)
    output_parser = PydanticOutputParser(pydantic_object=MissingValues)

    raw_model = init_model(config)

    def extract_prompt(s: State):
        return {
            "data": pandas_to_markdown(s.original),
            "format": output_parser.get_format_instructions(),
        }

    chain = extract_prompt | prompt | raw_model

    yield Command(update={"current_node": "analize_missing_values"})

    chunks = []
    async for chunk in chain.astream(state):
        chunks.append(chunk)
        yield Command(
            update={"messages": [chunk], "current_node": "analize_missing_values"}
        )

    final_message = functools.reduce(
        lambda acc, chunk: acc.__add__(chunk),
        chunks,
        BaseMessageChunk(content="", type=""),
    )
    missing_values = output_parser.parse(str(final_message.content))
    yield Command(
        update={
            "current_node": "analize_missing_values",
            "missing_values": missing_values,
            "loop_step": 1,
        }
    )


async def analize_outliers(state: State, *, config: Optional[RunnableConfig] = None):
    configuration = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_template(prompts.OUTLIERS_PROMPT)
    output_parser = PydanticOutputParser(pydantic_object=Outliers)

    raw_model = init_model(config)

    def extract_prompt(s: State):
        return {
            "data": pandas_to_markdown(s.original),
            "format": output_parser.get_format_instructions(),
        }

    chain = extract_prompt | prompt | raw_model

    yield Command(update={"current_node": "analize_outliers"})

    chunks = []
    async for chunk in chain.astream(state):
        chunks.append(chunk)
        yield Command(update={"messages": [chunk], "current_node": "analize_outliers"})

    final_message = functools.reduce(
        lambda acc, chunk: acc.__add__(chunk),
        chunks,
        BaseMessageChunk(content="", type=""),
    )
    outliers = output_parser.parse(str(final_message.content))
    yield Command(
        update={
            "current_node": "analize_outliers",
            "outliers": outliers,
            "loop_step": 1,
        }
    )


async def analize_inconsistencies(
    state: State, *, config: Optional[RunnableConfig] = None
):
    configuration = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_template(prompts.INCONSISTENCIES_PROMPT)
    output_parser = PydanticOutputParser(pydantic_object=Inconsistencies)

    raw_model = init_model(config)

    def extract_prompt(s: State):
        return {
            "data": pandas_to_markdown(s.original),
            "format": output_parser.get_format_instructions(),
        }

    chain = extract_prompt | prompt | raw_model

    yield Command(update={"current_node": "analize_inconsistencies"})

    chunks = []
    async for chunk in chain.astream(state):
        chunks.append(chunk)
        yield Command(
            update={"messages": [chunk], "current_node": "analize_inconsistencies"}
        )

    final_message = functools.reduce(
        lambda acc, chunk: acc.__add__(chunk),
        chunks,
        BaseMessageChunk(content="", type=""),
    )
    inconsistencies = output_parser.parse(str(final_message.content))
    yield Command(
        update={
            "current_node": "analize_inconsistencies",
            "inconsistencies": inconsistencies,
            "loop_step": 1,
        }
    )


async def analize_duplicates(state: State, *, config: Optional[RunnableConfig] = None):
    configuration = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_template(prompts.DUPLICATES_PROMPT)
    output_parser = PydanticOutputParser(pydantic_object=Duplicates)

    raw_model = init_model(config)

    def extract_prompt(s: State):
        return {
            "data": pandas_to_markdown(s.original),
            "format": output_parser.get_format_instructions(),
        }

    chain = extract_prompt | prompt | raw_model

    yield Command(update={"current_node": "analize_duplicates"})

    chunks = []
    async for chunk in chain.astream(state):
        chunks.append(chunk)
        yield Command(
            update={"messages": [chunk], "current_node": "analize_duplicates"}
        )

    final_message = functools.reduce(
        lambda acc, chunk: acc.__add__(chunk),
        chunks,
        BaseMessageChunk(content="", type=""),
    )
    duplicates = output_parser.parse(str(final_message.content))
    yield Command(
        update={
            "current_node": "analize_duplicates",
            "duplicates": duplicates,
            "loop_step": 1,
        }
    )


workflow = StateGraph(
    state_schema=State,
    input=InputState,
    output=OutputState,
    config_schema=Configuration,
)
workflow.add_node("analize_duplicates", analize_duplicates)
workflow.add_node("analize_missing_values", analize_missing_values)
workflow.add_node("analize_outliers", analize_outliers)
workflow.add_node("analize_inconsistencies", analize_inconsistencies)

workflow.add_edge(START, "analize_duplicates")
workflow.add_edge("analize_duplicates", "analize_missing_values")
workflow.add_edge("analize_missing_values", "analize_outliers")
workflow.add_edge("analize_outliers", END)

graph = workflow.compile()
graph.name = "DocAgent"
