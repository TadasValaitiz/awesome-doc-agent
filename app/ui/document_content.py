import random
from typing import Any, Iterator, Dict, cast
import streamlit as st
from langgraph_sdk.schema import StreamPart
import pandas as pd

from .data_frame import render_data_frame_from_state, render_data_frame_from_update
from service.doc_agent import DocAgent
from auth.types import FirebaseUserDict
from common.types import (
    DuplicateFixes,
    Duplicates,
    Inconsistencies,
    InconsistencyFixes,
    MissingValueFixes,
    MissingValues,
    OutlierFixes,
    Outliers,
    State,
    StateUpdates,
    Summary,
)
import time


def render_document_content(user_info: FirebaseUserDict):
    """Render the document content."""
    render_history(user_info)
    render_actions(user_info)


def render_state_actions(doc_agent: DocAgent, state: State):
    """Render the actions for the document content."""
    random_key = hash(str(state))
    if state.current_node == "init":
        if st.button(
            "Start cleaning",
            use_container_width=True,
            key=f"start_cleaning:{random_key}",
        ):
            stream = doc_agent.run_action(user_intent="analyze_all")
            render_stream(stream, doc_agent)
    else:
        if (
            state.duplicates
            and state.duplicates.duplicates
            and not state.duplicate_fixes
        ):
            if st.button(
                "Fix duplicates",
                use_container_width=True,
                key=f"fix_duplicates:{random_key}",
            ):
                stream = doc_agent.run_action(user_intent="fix_duplicates")
                render_stream(stream, doc_agent)
        if (
            state.missing_values
            and state.missing_values.missing_values
            and not state.missing_value_fixes
        ):
            if st.button(
                "Fix missing values",
                use_container_width=True,
                key=f"fix_missing_values:{random_key}",
            ):
                stream = doc_agent.run_action(user_intent="fix_missing_values")
                render_stream(stream, doc_agent)
        if state.outliers and state.outliers.outliers and not state.outlier_fixes:
            if st.button(
                "Fix outliers",
                use_container_width=True,
                key=f"fix_outliers:{random_key}",
            ):
                stream = doc_agent.run_action(user_intent="fix_outliers")
                render_stream(stream, doc_agent)
        if (
            state.inconsistencies
            and state.inconsistencies.warnings
            and not state.inconsistency_fixes
        ):
            if st.button(
                "Fix inconsistencies",
                use_container_width=True,
                key=f"fix_inconsistencies:{random_key}",
            ):
                stream = doc_agent.run_action(user_intent="fix_inconsistencies")
                render_stream(stream, doc_agent)


def render_actions(user_info: FirebaseUserDict):
    """Render the actions for the document content."""

    if st.session_state.thread_id is not None:
        doc_agent = DocAgent.from_thread(
            thread_id=st.session_state.thread_id, user_id=user_info.get("localId")
        )
        state, threadState = doc_agent.get_state()
        render_state_actions(doc_agent, state)
        if user_message := st.chat_input("Type your message here..."):
            stream = doc_agent.run_action(user_intent="chat", message=user_message)
            render_stream(stream, doc_agent)


blacklisted_nodes = [
    "planner",
    "init",
    "load_document",
    "start",
    "analyzer",
    "fixer",
]


def render_history(user_info: FirebaseUserDict):
    if st.session_state.thread_id is not None:
        doc_agent = DocAgent.from_thread(
            thread_id=st.session_state.thread_id, user_id=user_info.get("localId")
        )
        history = doc_agent.get_history()
        item = doc_agent.get_thread_document()

        if len(history) == 0:
            render_data_frame_from_update(item.df)

        for index, step in enumerate(history):
            if index == 0:
                render_data_frame_from_update(item.df)
            metadata = step.get("metadata")
            writes = metadata.get("writes") if metadata else {}
            state = State(**cast(Dict[str, Any], step.get("values")))
            if writes:
                update = StateUpdates.from_update(writes)
                if update.current_node not in blacklisted_nodes:
                    with st.chat_message("assistant"):
                        render_history_chat_message(
                            data_frame=item.df,
                            update=update,
                            state=state,
                            finished=True,
                        )


def render_history_chat_message(
    data_frame: pd.DataFrame, update: StateUpdates, state: State, finished: bool = True
):
    step_update = update.current_node_state_updates()
    if isinstance(step_update, Summary):
        st.markdown(step_update.summary)
        render_data_frame_from_state(data_frame, state)
    else:
        st.expander(node_name_map(finished=finished, node=update.current_node)).write(
            update.current_node_state_updates()
        )
        render_data_frame_from_update(data_frame, update, show_only_affected=True)


def node_name_map(*, finished: bool, node: str) -> str:
    if not finished:
        return {
            "analyze_missing_values": "🔄 Analyzing for missing values in dataset...",
            "analyze_outliers": "🔄 Detecting outliers in the data...",
            "analyze_duplicates": "🔄 Identifying duplicate records...",
            "analyze_inconsistencies": "🔄 Finding data inconsistencies...",
            "fix_missing_values": "🔄 Fixing missing values...",
            "fix_outliers": "🔄 Fixing outliers...",
            "fix_duplicates": "🔄 Fixing duplicates...",
            "fix_inconsistencies": "🔄 Fixing inconsistencies...",
            "summarize_analysis": "🔄 Summarizing analysis...",
            "summarize_fixes": "🔄 Summarizing fixes...",
            "chat_message": "💬 Agent reasoning...",
        }.get(node, node)
    else:
        return {
            "analyze_missing_values": "✅ Missing Values Analysis",
            "analyze_outliers": "✅ Outliers Detection",
            "analyze_duplicates": "✅ Duplicates Identification",
            "analyze_inconsistencies": "✅ Inconsistencies Analysis",
            "fix_missing_values": "✅ Missing Values Fixes",
            "fix_outliers": "✅ Outliers Fixes",
            "fix_duplicates": "✅ Duplicates Fixes",
            "fix_inconsistencies": "✅ Inconsistencies Fixes",
            "summarize_analysis": "📝 Analysis Summary",
            "summarize_fixes": "📝 Fixes Summary",
            "chat_message": "💬 Chat",
        }.get(node, node)


def render_stream(stream: Iterator[StreamPart], doc_agent: DocAgent):
    with st.container():
        langgraph_node = "unknown_node"
        langgraph_step = 0
        step_dict: Dict[int, str] = {}
        step_containers = {}
        for part in stream:
            if part.event == "messages/partial":
                data = part.data[0]
                content = data.get("content", "")

                container = step_containers.get(langgraph_step, None)
                if container is None:
                    container = st.chat_message("assistant").empty()
                    # Collapse previous containers
                    for prev_step, prev_container in step_containers.items():
                        if prev_step < langgraph_step:
                            prev_container.expander(
                                node_name_map(finished=True, node=step_dict[prev_step]),
                                expanded=True,
                            )

                    step_containers[langgraph_step] = container

                with container.expander(
                    node_name_map(finished=False, node=langgraph_node), expanded=True
                ):
                    st.markdown(content)
            elif part.event == "values":
                data = part.data
                state = State(**data)
                next_node = state.next_node

                print(
                    f"\n===========\n{state.current_node}:{state.next_node}\n===========\n"
                )
                print(state)
                print("\n===========")

                if next_node == "__end__" and (
                    state.current_node == "summarize_fixes"
                    or state.current_node == "summarize_analysis"
                ):
                    # Collapse and rename
                    for prev_step, prev_container in step_containers.items():
                        prev_container.expander(
                            node_name_map(finished=True, node=step_dict[prev_step]),
                            expanded=True,
                        )
                    render_data_frame_from_state(
                        doc_agent.get_thread_document().df,
                        state,
                    )
                    render_state_actions(doc_agent, state)

            elif part.event == "updates":
                data = part.data
                update = StateUpdates.from_update(data)

            elif part.event == "messages/metadata":
                data = part.data

                if isinstance(data, dict):
                    for run_id, run_data in data.items():
                        if isinstance(run_data, dict):
                            metadata = run_data.get("metadata", {})
                            langgraph_node = metadata.get("langgraph_node")
                            langgraph_step = metadata.get("langgraph_step")
                            step_dict[langgraph_step] = langgraph_node

                            break
