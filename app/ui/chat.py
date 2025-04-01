import streamlit as st
from typing import List, Dict, Any, Callable, Optional
import datetime


def format_time(timestamp: str) -> str:
    """Format a timestamp string to a readable time format."""
    dt = datetime.datetime.fromisoformat(timestamp)
    return dt.strftime("%H:%M")


def render_message(message: Dict[str, Any], is_last: bool = False) -> None:
    """
    Render a single chat message.

    Args:
        message: Message dictionary containing role, content, etc.
        is_last: Whether this is the last message in the conversation
    """
    role = message["role"]
    content = message["content"]
    time_str = format_time(message["created_at"])

    if role == "user":
        st.markdown(
            f"""
            <div class="message-container" style="display: flex; justify-content: flex-end;">
                <div class="user-message">
                    <div>{content}</div>
                    <div style="font-size: 0.7em; text-align: right; opacity: 0.7;">{time_str}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif role == "assistant":
        st.markdown(
            f"""
            <div class="message-container" style="display: flex; justify-content: flex-start;">
                <div class="bot-message">
                    <div>{content}</div>
                    <div style="font-size: 0.7em; text-align: right; opacity: 0.7;">{time_str}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:  # system message or other roles
        st.markdown(
            f"""
            <div style="text-align: center; margin: 10px 0; color: #888; font-size: 0.8em;">
                {content}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_chat(
    messages: List[Dict[str, Any]],
    conversation_name: str,
    on_send_message: Callable[[str], None],
    on_rename_conversation: Callable[[str], None],
) -> None:
    """
    Render the chat interface.

    Args:
        messages: List of message dictionaries
        conversation_name: Name of the current conversation
        on_send_message: Callback for sending a new message
        on_rename_conversation: Callback for renaming the conversation
    """
    # Chat header
    col1, col2 = st.columns([3, 1])
    with col1:
        if "is_editing_title" not in st.session_state:
            st.session_state.is_editing_title = False

        if st.session_state.is_editing_title:
            # Edit conversation name
            new_name = st.text_input(
                "Conversation name", value=conversation_name, key="edit_conv_name"
            )
            col3, col4 = st.columns(2)
            with col3:
                if st.button("Save", key="save_title_btn"):
                    on_rename_conversation(new_name)
                    st.session_state.is_editing_title = False
            with col4:
                if st.button("Cancel", key="cancel_title_btn"):
                    st.session_state.is_editing_title = False
        else:
            # Display conversation name with edit button
            st.markdown(f"## {conversation_name}")

    with col2:
        if not st.session_state.is_editing_title:
            if st.button("Rename", key="rename_btn"):
                st.session_state.is_editing_title = True

    st.markdown("---")

    # Chat messages
    chat_container = st.container()
    with chat_container:
        if not messages:
            st.info("Start a conversation!")
        else:
            for i, message in enumerate(messages):
                render_message(message, is_last=(i == len(messages) - 1))

    # Input area
    st.markdown("---")
    with st.container():
        message_input = st.text_area("Message", key="message_input", height=100)
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("Send", key="send_btn", use_container_width=True):
                if message_input and message_input.strip():
                    on_send_message(message_input)
                    # Clear the input field
                    st.session_state.message_input = ""


def get_streaming_container() -> Any:
    """
    Create a container for streaming AI responses.

    Returns:
        A streamlit container for streaming content
    """
    return st.container()
