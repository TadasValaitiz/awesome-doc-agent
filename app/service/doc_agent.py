import os
from typing import Optional, Iterator
from langgraph_sdk import get_sync_client
from langgraph_sdk.schema import StreamPart, Thread
from common.types import DocumentMetadata


class DocAgent:
    graph_id = "generic_chat_agent"

    def __init__(
        self,
        user_id: str,
        model: Optional[str] = "openai/gpt-4o-mini",
        thread_id: Optional[str] = None,
    ):
        self.user_id = user_id
        self.model = model
        self.thread_id = thread_id
        api_key = os.getenv("LANGSMITH_API_KEY")
        if not api_key:
            raise ValueError("LANGSMITH_API_KEY is not set")
        self.client = get_sync_client(
            url="https://awesome-doc-agent-4aea3ef58f0f58c9b577c1f02420ef02.us.langgraph.app",
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

    def run_action(self, action: str) -> Iterator[StreamPart]:
        """Run an action on the existing thread with streaming.

        Args:
            action: The action to run (typically a user message)

        Returns:
            Iterator of StreamPart containing chunks of the response
        """
        if self.thread_id is None:
            raise ValueError(
                "No thread_id available. Use from_thread() or new_clean_data_run() first."
            )

        return self.client.runs.stream(
            thread_id=self.thread_id,
            assistant_id=self.graph_id,
            input={
                "messages": [{"role": "user", "content": action}],
            },
            stream_mode=["values", "debug"],  # Get both values and debug info in stream
            config={"configurable": {"model": self.model}},
        )

    def new_clean_data_run(
        self, metadata: DocumentMetadata
    ) -> tuple[Thread, Iterator[StreamPart]]:
        """Create a new thread and start initial streaming run.

        Args:
            metadata: Document metadata to associate with the thread

        Returns:
            Tuple of (thread info, stream iterator)
        """
        # Create metadata dictionary using DocumentMetadata's serialization
        thread_metadata = {
            "user_id": self.user_id,
            "file_name": metadata.file_name,
            "document_metadata": metadata.to_dict(),
        }

        # Create a new thread with metadata
        thread = self.client.threads.create(
            metadata=thread_metadata,
            graph_id=self.graph_id,
        )
        self.thread_id = thread["thread_id"]

        # Start streaming run
        stream = self.client.runs.stream(
            thread_id=self.thread_id,
            assistant_id=self.graph_id,
            input={
                "messages": [{"role": "user", "content": "Are you doc agent?"}],
            },
            stream_mode=["values", "debug"],
            config={"configurable": {"model": self.model}},
        )

        return thread, stream

    def _get_doc_assistant(self):
        assistants = self.client.assistants.search(graph_id=self.graph_id)
        return assistants[0]
