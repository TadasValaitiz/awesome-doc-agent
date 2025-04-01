import streamlit as st
from typing import List, Dict, Any, Optional, Callable
from auth import FirebaseUserDict


def render_sidebar(
    user_info: Optional[FirebaseUserDict],
) -> None:
    """
    Render the sidebar with conversation list and controls.

    Args:
        conversations: List of conversation dictionaries
        on_select_conversation: Callback for when a conversation is selected
        on_new_conversation: Callback for creating a new conversation
        on_delete_conversation: Callback for deleting a conversation
        current_conversation_id: Currently selected conversation ID
    """
    with st.sidebar:
        if user_info is not None:
            st.markdown("## Sidebar")

