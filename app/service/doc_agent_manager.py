import os
from langgraph_sdk import get_sync_client
from service.constants import URL
import streamlit as st


class DocAgentManager:

    def __init__(self, user_id: str):
        self.user_id = user_id

        if "langchain" in st.secrets:
            api_key = st.secrets["langchain"]["api_key"]
        else:
            raise ValueError("langchain api_key is not set in Streamlit secrets")
        self.client = get_sync_client(
            url=URL,
            api_key=api_key,
        )

    def list_threads(self):
        # Search for threads associated with this user
        return self.client.threads.search(metadata={"user_id": self.user_id})
