import os
from langgraph_sdk import get_sync_client


class DocAgentManager:
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        api_key = os.getenv("LANGSMITH_API_KEY")
        if not api_key:
            raise ValueError("LANGSMITH_API_KEY is not set")
        self.client = get_sync_client(
            url="https://awesome-doc-agent-4aea3ef58f0f58c9b577c1f02420ef02.us.langgraph.app",
            api_key=api_key,
        )
    
    def list_threads(self):
        # Search for threads associated with this user
        return self.client.threads.search(metadata={"user_id": self.user_id})