"""Define a generic chat agent agent."""

import functools
from typing import Literal, Optional, Type, Any, TypeVar, cast, List, Dict
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
    BaseValidatedModel,
    MissingValues,
    Outliers,
    Duplicates,
    Inconsistencies,
    MissingValueFixes,
    OutlierFixes,
    DuplicateFixes,
    InconsistencyFixes,
    Summary,
    State,
    InputState)
from doc_agent.utils import deserialize_dataframe, init_model, pandas_to_markdown
from langgraph.config import get_store


async def load_document(state: State, config: RunnableConfig):
    configuration = Configuration.from_runnable_config(config)

    if state.original is None:
        if not config:
            raise ValueError("Configuration is required to load document")

        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            raise ValueError("Thread ID is required to load document")

        store = get_store()
        if store is None:
            raise ValueError("Store not found in config")

        item = await store.aget(namespace=("documents",), key=thread_id)
        if item is None:
            raise ValueError(f"No document found for thread {thread_id}")

        file_name = item.value.get("file_name")
        serialized = item.value.get("serialized_df")
        if serialized is None:
            raise ValueError(f"No serialized dataframe found for thread {thread_id}")
        df = deserialize_dataframe(serialized)
        yield Command(
            update={
                "current_node": "load_document",
                "original": df,
                "file_name": file_name,
            }
        )
    else:
        yield Command(update={"current_node": "load_document"})


async def planner(state: State, *, config: Optional[RunnableConfig] = None):
    next_node = None

    if state.user_intent == "fix_duplicates":
        if state.duplicate_fixes is None:
            next_node = "fixer"
        else:
            next_node = "summarize_fixes"
    elif state.user_intent == "fix_missing_values":
        if state.missing_value_fixes is None:
            next_node = "fixer"
        else:
            next_node = "summarize_fixes"
    elif state.user_intent == "fix_outliers":
        if state.outlier_fixes is None:
            next_node = "fixer"
        else:
            next_node = "summarize_fixes"
    elif state.user_intent == "fix_inconsistencies":
        if state.inconsistency_fixes is None:
            next_node = "fixer"
        else:
            next_node = "summarize_fixes"
    elif state.user_intent == "analyze_all":
        if (
            state.duplicates is None
            or state.missing_values is None
            or state.outliers is None
            or state.inconsistencies is None
        ):
            next_node = "analyzer"
        else:
            next_node = "summarize_analysis"
    elif (
        state.user_intent == "chat"
        and state.messages
        and state.messages[-1].type == "human"
    ):
        next_node = "chat_message"
    else:
        next_node = "__end__"

    yield Command(
        update={
            "supervisor_node": "planner",
            "current_node": "planner",
            "next_node": next_node,
        }
    )


async def fixer(state: State, *, config: Optional[RunnableConfig] = None):
    next_node = None

    if state.user_intent == "fix_duplicates":
        if (
            state.duplicate_fixes is None
            or state.duplicate_fixes.validation_error is not None
        ):
            next_node = "fix_duplicates"
        else:
            next_node = "planner"
    elif state.user_intent == "fix_missing_values":
        if (
            state.missing_value_fixes is None
            or state.missing_value_fixes.validation_error is not None
        ):
            next_node = "fix_missing_values"
        else:
            next_node = "planner"
    elif state.user_intent == "fix_outliers":
        if (
            state.outlier_fixes is None
            or state.outlier_fixes.validation_error is not None
        ):
            next_node = "fix_outliers"
        else:
            next_node = "planner"
    elif state.user_intent == "fix_inconsistencies":
        if (
            state.inconsistency_fixes is None
            or state.inconsistency_fixes.validation_error is not None
        ):
            next_node = "fix_inconsistencies"
        else:
            next_node = "planner"
    yield Command(
        update={
            "supervisor_node": "fixer",
            "current_node": "fixer",
            "next_node": next_node,
        }
    )


async def analyzer(state: State, *, config: Optional[RunnableConfig] = None):
    next_node = None

    if state.duplicates is None or state.duplicates.validation_error is not None:
        next_node = "analyze_duplicates"
    elif (
        state.missing_values is None
        or state.missing_values.validation_error is not None
    ):
        next_node = "analyze_missing_values"
    elif state.outliers is None or state.outliers.validation_error is not None:
        next_node = "analyze_outliers"
    elif (
        state.inconsistencies is None
        or state.inconsistencies.validation_error is not None
    ):
        next_node = "analyze_inconsistencies"
    else:
        next_node = "planner"

    yield Command(
        update={
            "supervisor_node": "analyzer",
            "current_node": "analyzer",
            "next_node": next_node,
        }
    )


class AnalyzeNode:
    """Abstract base class for analysis nodes that follow a common pattern."""

    def __init__(self, node_name: str):
        self.node_name = node_name

    def get_prompt_template(self) -> str:
        """Return the prompt template to use for this analysis."""
        raise NotImplementedError

    def get_output_model(self) -> Type[BaseModel]:
        """Return the Pydantic model to parse the output into.

        The returned model must have a `from_llm_output` class method.
        """
        raise NotImplementedError

    def get_state_key(self) -> str:
        """Return the key in the state to update with the result."""
        raise NotImplementedError

    async def __call__(self, state: State, *, config: Optional[RunnableConfig] = None):
        """Execute the analysis node with the common pattern."""
        prompt = ChatPromptTemplate.from_template(self.get_prompt_template())
        output_parser = PydanticOutputParser(pydantic_object=self.get_output_model())

        raw_model = init_model(config)

        def extract_prompt(s: State):
            if s.original is None:
                raise ValueError("Original dataframe is not set")
            return {
                "data": pandas_to_markdown(s.original),
                "format": output_parser.get_format_instructions(),
            }

        chain = extract_prompt | prompt | raw_model

        chunks = []
        async for chunk in chain.astream(state):
            chunks.append(chunk)
            yield Command(update={"messages": [chunk], "current_node": self.node_name})

        final_message = functools.reduce(
            lambda acc, chunk: acc.__add__(chunk),
            chunks,
            BaseMessageChunk(content="", type=""),
        )

        # type: ignore - we know this class has from_llm_output because of our implementation
        model_class = self.get_output_model()
        result = model_class.from_llm_output(str(final_message.content))  # type: ignore
        yield Command(
            update={
                "current_node": self.node_name,
                self.get_state_key(): result,
                "loop_step": 1,
            }
        )


class FixNode(AnalyzeNode):
    """Base class for fix nodes that need to reference analysis results."""

    def get_analysis_key(self) -> str:
        """Return the key in the state to get analysis results from."""
        raise NotImplementedError

    async def __call__(self, state: State, *, config: Optional[RunnableConfig] = None):
        """Execute the fix node with the common pattern."""
        prompt = ChatPromptTemplate.from_template(self.get_prompt_template())
        output_parser = PydanticOutputParser(pydantic_object=self.get_output_model())

        raw_model = init_model(config)

        def extract_prompt(s: State):
            if s.original is None:
                raise ValueError("Original dataframe is not set")

            analysis_key = self.get_analysis_key()
            analysis_result = cast(BaseValidatedModel, getattr(s, analysis_key))
            if analysis_result is None:
                raise ValueError(f"Analysis result {analysis_key} is not set")

            return {
                "data": pandas_to_markdown(s.original),
                "format": output_parser.get_format_instructions(),
                analysis_key: analysis_result.to_markdown(),
            }

        chain = extract_prompt | prompt | raw_model

        chunks = []
        async for chunk in chain.astream(state):
            chunks.append(chunk)
            yield Command(update={"messages": [chunk], "current_node": self.node_name})

        final_message = functools.reduce(
            lambda acc, chunk: acc.__add__(chunk),
            chunks,
            BaseMessageChunk(content="", type=""),
        )

        model_class = self.get_output_model()
        result = model_class.from_llm_output(str(final_message.content))  # type: ignore
        yield Command(
            update={
                "current_node": self.node_name,
                self.get_state_key(): result,
                "loop_step": 1,
            }
        )


class MissingValuesNode(AnalyzeNode):
    def __init__(self):
        super().__init__("analyze_missing_values")

    def get_prompt_template(self) -> str:
        return prompts.MISSING_VALUES_PROMPT

    def get_output_model(self) -> Type[BaseModel]:
        return MissingValues

    def get_state_key(self) -> str:
        return "missing_values"


class OutliersNode(AnalyzeNode):
    def __init__(self):
        super().__init__("analyze_outliers")

    def get_prompt_template(self) -> str:
        return prompts.OUTLIERS_PROMPT

    def get_output_model(self) -> Type[BaseModel]:
        return Outliers

    def get_state_key(self) -> str:
        return "outliers"


class InconsistenciesNode(AnalyzeNode):
    def __init__(self):
        super().__init__("analyze_inconsistencies")

    def get_prompt_template(self) -> str:
        return prompts.INCONSISTENCIES_PROMPT

    def get_output_model(self) -> Type[BaseModel]:
        return Inconsistencies

    def get_state_key(self) -> str:
        return "inconsistencies"


class DuplicatesNode(AnalyzeNode):
    def __init__(self):
        super().__init__("analyze_duplicates")

    def get_prompt_template(self) -> str:
        return prompts.DUPLICATES_PROMPT

    def get_output_model(self) -> Type[BaseModel]:
        return Duplicates

    def get_state_key(self) -> str:
        return "duplicates"


class MissingValueFixesNode(FixNode):
    def __init__(self):
        super().__init__("fix_missing_values")

    def get_prompt_template(self) -> str:
        return prompts.FIX_MISSING_VALUES_PROMPT

    def get_output_model(self) -> Type[BaseModel]:
        return MissingValueFixes

    def get_state_key(self) -> str:
        return "missing_value_fixes"

    def get_analysis_key(self) -> str:
        return "missing_values"


class OutlierFixesNode(FixNode):
    def __init__(self):
        super().__init__("fix_outliers")

    def get_prompt_template(self) -> str:
        return prompts.FIX_OUTLIERS_PROMPT

    def get_output_model(self) -> Type[BaseModel]:
        return OutlierFixes

    def get_state_key(self) -> str:
        return "outlier_fixes"

    def get_analysis_key(self) -> str:
        return "outliers"


class DuplicateFixesNode(FixNode):
    def __init__(self):
        super().__init__("fix_duplicates")

    def get_prompt_template(self) -> str:
        return prompts.FIX_DUPLICATES_PROMPT

    def get_output_model(self) -> Type[BaseModel]:
        return DuplicateFixes

    def get_state_key(self) -> str:
        return "duplicate_fixes"

    def get_analysis_key(self) -> str:
        return "duplicates"


class InconsistencyFixesNode(FixNode):
    def __init__(self):
        super().__init__("fix_inconsistencies")

    def get_prompt_template(self) -> str:
        return prompts.FIX_INCONSISTENCIES_PROMPT

    def get_output_model(self) -> Type[BaseModel]:
        return InconsistencyFixes

    def get_state_key(self) -> str:
        return "inconsistency_fixes"

    def get_analysis_key(self) -> str:
        return "inconsistencies"


class SummaryNode:
    """Abstract base class for summary nodes that follow a common pattern."""

    def __init__(self, node_name: str):
        self.node_name = node_name

    def get_prompt_template(self) -> str:
        """Return the prompt template to use for this summary."""
        raise NotImplementedError

    def get_state_key(self) -> str:
        """Return the key in the state to update with the result."""
        raise NotImplementedError

    def collect_context_data(
        self, state: State
    ) -> Dict[str, Optional[BaseValidatedModel]]:
        """Collect fields/models needed for this summary type.

        Returns a dictionary where keys are section titles and values are
        the BaseValidatedModel instances containing the data.
        """
        raise NotImplementedError

    def _format_as_markdown(
        self, data: Dict[str, Optional[BaseValidatedModel]]
    ) -> List[str]:
        """Format the collected data into markdown sections."""
        markdown_sections = []

        for section_title, model in data.items():
            if model is not None:
                markdown_sections.append(f"## {section_title}\n{model.to_markdown()}")

        return markdown_sections

    async def __call__(self, state: State, *, config: Optional[RunnableConfig] = None):
        """Execute the summary node with the common pattern."""
        prompt = ChatPromptTemplate.from_template(self.get_prompt_template())
        raw_model = init_model(config)

        def extract_prompt(s: State):
            if s.original is None:
                raise ValueError("Original dataframe is not set")

            context_data = self.collect_context_data(s)
            markdown_sections = self._format_as_markdown(context_data)

            return {
                "data": pandas_to_markdown(s.original),
                "context": "\n\n".join(markdown_sections),
            }

        chain = extract_prompt | prompt | raw_model

        chunks = []
        async for chunk in chain.astream(state):
            chunks.append(chunk)
            yield Command(update={"messages": [chunk], "current_node": self.node_name})

        final_message = functools.reduce(
            lambda acc, chunk: acc.__add__(chunk),
            chunks,
            BaseMessageChunk(content="", type=""),
        )

        content = str(final_message.content)  # type: ignore

        yield Command(
            update={
                "current_node": self.node_name,
                self.get_state_key(): Summary(summary=content),
                "messages": [AIMessage(content=content)],
                "next_node": "__end__",
                "loop_step": 1,
            }
        )


class AnalysisSummaryNode(SummaryNode):
    """Node that generates a summary of all analysis results."""

    def __init__(self):
        super().__init__("summarize_analysis")

    def get_prompt_template(self) -> str:
        return prompts.ANALYSIS_SUMMARY_PROMPT

    def get_state_key(self) -> str:
        return "analysis_summary"

    def collect_context_data(
        self, state: State
    ) -> Dict[str, Optional[BaseValidatedModel]]:
        """Collect all analysis models."""
        return {
            "Missing Values Analysis": state.missing_values,
            "Outliers Analysis": state.outliers,
            "Duplicates Analysis": state.duplicates,
            "Inconsistencies Analysis": state.inconsistencies,
        }


class FixesSummaryNode(SummaryNode):
    """Node that generates a summary of all fixes applied."""

    def __init__(self):
        super().__init__("summarize_fixes")

    def get_prompt_template(self) -> str:
        return prompts.FIXES_SUMMARY_PROMPT

    def get_state_key(self) -> str:
        return "fixes_summary"

    def collect_context_data(
        self, state: State
    ) -> Dict[str, Optional[BaseValidatedModel]]:
        """Collect all fix models."""
        if state.user_intent == "fix_duplicates":
            return {
                "Duplicates Fixes": state.duplicate_fixes,
            }
        elif state.user_intent == "fix_missing_values":
            return {
                "Missing Values Fixes": state.missing_value_fixes,
            }
        elif state.user_intent == "fix_outliers":
            return {
                "Outliers Fixes": state.outlier_fixes,
            }
        elif state.user_intent == "fix_inconsistencies":
            return {
                "Inconsistencies Fixes": state.inconsistency_fixes,
            }
        else:
            raise ValueError(f"Unknown user intent: {state.user_intent}")


async def chat_message(state: State, *, config: Optional[RunnableConfig] = None):
    configuration = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_template(prompts.CHAT_MESSAGE_PROMPT)

    raw_model = init_model(config)

    def extract_prompt(s: State):
        return {
            "context": pandas_to_markdown(s.original) if s.original is not None else "",
            "history": "\n".join(
                [str(f"{msg.type}: {msg.content}") for msg in s.messages[-10:-1]]
            ),
            "question": s.messages[-1].content,
        }

    chain = extract_prompt | prompt | raw_model

    yield Command(update={"current_node": "chat_message", "loop_step": 1})

    async for chunk in chain.astream(state):
        yield Command(update={"messages": [chunk], "current_node": "chat_message"})


def fixer_switch(
    state: State,
) -> Literal[
    "fix_missing_values",
    "fix_outliers",
    "fix_duplicates",
    "fix_inconsistencies",
    "planner",
]:
    # Type assertion to ensure next_node is of the correct Literal type
    next_node: Literal[
        "fix_missing_values",
        "fix_outliers",
        "fix_duplicates",
        "fix_inconsistencies",
        "planner",
    ] = state.next_node  # type: ignore
    return next_node


def planner_switch(
    state: State,
) -> Literal[
    "analyzer",
    "fixer",
    "summarize_analysis",
    "summarize_fixes",
    "chat_message",
]:
    # Type assertion to ensure next_node is of the correct Literal type
    next_node: Literal[
        "analyzer",
        "fixer",
        "summarize_analysis",
        "summarize_fixes",
        "chat_message",
    ] = state.next_node  # type: ignore
    return next_node


def analyzer_switch(
    state: State,
) -> Literal[
    "analyze_duplicates",
    "analyze_missing_values",
    "analyze_outliers",
    "analyze_inconsistencies",
    "planner",
]:
    # Type assertion to ensure next_node is of the correct Literal type
    next_node: Literal[
        "analyze_duplicates",
        "analyze_missing_values",
        "analyze_outliers",
        "analyze_inconsistencies",
        "planner",
    ] = state.next_node  # type: ignore
    return next_node


# Initialize node instances
analyze_missing_values = MissingValuesNode()
analyze_outliers = OutliersNode()
analyze_inconsistencies = InconsistenciesNode()
analyze_duplicates = DuplicatesNode()
fix_missing_values = MissingValueFixesNode()
fix_outliers = OutlierFixesNode()
fix_duplicates = DuplicateFixesNode()
fix_inconsistencies = InconsistencyFixesNode()
summarize_analysis = AnalysisSummaryNode()
summarize_fixes = FixesSummaryNode()


workflow = StateGraph(
    state_schema=State,
    input=InputState,
    config_schema=Configuration,
)
workflow.add_node("load_document", load_document)

workflow.add_node("planner", planner)
workflow.add_node("analyzer", analyzer)
workflow.add_node("fixer", fixer)
workflow.add_node("chat_message", chat_message)

# Analyzer
workflow.add_node("analyze_duplicates", analyze_duplicates)
workflow.add_node("analyze_missing_values", analyze_missing_values)
workflow.add_node("analyze_outliers", analyze_outliers)
workflow.add_node("analyze_inconsistencies", analyze_inconsistencies)

# Fixer
workflow.add_node("fix_missing_values", fix_missing_values)
workflow.add_node("fix_outliers", fix_outliers)
workflow.add_node("fix_duplicates", fix_duplicates)
workflow.add_node("fix_inconsistencies", fix_inconsistencies)

# Summarizer
workflow.add_node("summarize_analysis", summarize_analysis)
workflow.add_node("summarize_fixes", summarize_fixes)

workflow.add_edge(START, "load_document")
workflow.add_edge("load_document", "planner")
workflow.add_conditional_edges("planner", planner_switch)
workflow.add_conditional_edges("fixer", fixer_switch)
workflow.add_conditional_edges("analyzer", analyzer_switch)

# Analyzer
workflow.add_edge("analyze_duplicates", "analyzer")
workflow.add_edge("analyze_missing_values", "analyzer")
workflow.add_edge("analyze_outliers", "analyzer")
workflow.add_edge("analyze_inconsistencies", "analyzer")

# Fixer
workflow.add_edge("fix_missing_values", "fixer")
workflow.add_edge("fix_outliers", "fixer")
workflow.add_edge("fix_duplicates", "fixer")
workflow.add_edge("fix_inconsistencies", "fixer")

workflow.add_edge("summarize_analysis", END)
workflow.add_edge("summarize_fixes", END)
workflow.add_edge("chat_message", end_key=END)

graph = workflow.compile()
graph.name = "DocAgent"
