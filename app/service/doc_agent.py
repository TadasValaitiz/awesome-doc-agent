import os
from typing import Any, AsyncIterator, Dict, Literal, Optional, Iterator, cast
from langgraph_sdk import get_sync_client, get_client
from langgraph_sdk.schema import StreamPart, Thread
from langchain_core.messages import HumanMessage, SystemMessage
from common.types import DocumentMetadata, State, serialize_dataframe
from service.constants import URL
import streamlit as st


class DocAgent:
    graph_id = "doc_agent"

    def __init__(
        self,
        user_id: str,
        model: Optional[str] = "openai/gpt-4o-mini",
        thread_id: Optional[str] = None,
    ):
        self.user_id = user_id
        self.model = model
        self.thread_id = thread_id
        if "langchain" in st.secrets:
            api_key = st.secrets["langchain"]["api_key"]
        else:
            raise ValueError("langchain api_key is not set in Streamlit secrets")
        self.client = get_sync_client(
            url=URL,
            api_key=api_key,
        )

    @classmethod
    def from_thread(
        cls, thread_id: str, user_id: str, model: Optional[str] = "openai/gpt-4o-mini"
    ):
        """Factory method to create DocAgent instance from an existing thread.

        Args:
            thread_id: The ID of the existing thread
            user_id: Optional user ID to associate with the agent
            model: Optional model override

        Returns:
            DocAgent instance configured for the thread
        """
        return cls(user_id=user_id, model=model, thread_id=thread_id)

    def run_action(
        self,
        user_intent: Literal[
            "analyze_all",
            "fix_duplicates",
            "fix_missing_values",
            "fix_outliers",
            "fix_inconsistencies",
            "chat",
        ],
        message: str = "",
    ) -> Iterator[StreamPart]:
        if self.thread_id is None:
            raise ValueError(
                "No thread_id available. Use from_thread() or new_clean_data_run() first."
            )

        if user_intent == "chat":
            input = {
                "messages": [HumanMessage(content=message)],
                "user_intent": user_intent,
                "current_node": "__start__",
            }
        else:
            input = {
                "messages": [SystemMessage(content=user_intent)],
                "user_intent": user_intent,
                "current_node": "__start__",
            }

        iterator = self.client.runs.stream(
            thread_id=self.thread_id,
            assistant_id=self.graph_id,
            input=input,
            stream_mode=["values", "messages", "updates"],
            config={"configurable": {"model": self.model}},
        )
        return iterator

    def get_history(self):
        if self.thread_id is None:
            raise ValueError(
                "No thread_id available. Use from_thread() or new_clean_data_run() first."
            )
        history = self.client.threads.get_history(thread_id=self.thread_id, limit=100)
        return list(reversed(history))

    def get_state(self):
        if self.thread_id is None:
            raise ValueError("No thread_id available")
        threadState = self.client.threads.get_state(thread_id=self.thread_id)
        values = cast(dict[str, Any], threadState.get("values", {}))
        state = State(**values)
        return state, threadState

    def get_thread_document(self) -> DocumentMetadata:
        if self.thread_id is None:
            raise ValueError("No thread_id available")
        item = self.client.store.get_item(
            ["documents"],
            key=self.thread_id,
        )
        return DocumentMetadata.from_dict(item.get("value"))

    def new_thread(self, metadata: DocumentMetadata) -> Thread:

        thread_metadata = {
            "user_id": self.user_id,
            "file_name": metadata.file_name,
        }

        thread = self.client.threads.create(
            metadata=thread_metadata,
            graph_id=self.graph_id,
        )
        self.thread_id = thread["thread_id"]

        self.client.store.put_item(
            ["documents"],
            key=self.thread_id,
            value=metadata.to_dict(),
        )

        return thread

    def _get_doc_assistant(self):
        assistants = self.client.assistants.search(graph_id=self.graph_id)
        return assistants[0]
