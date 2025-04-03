from typing import Iterator, cast, Dict, Any
import streamlit as st
from langgraph_sdk.schema import Run, StreamPart

from service.doc_agent import DocAgent
from auth.types import FirebaseUserDict


def render_document_content(user_info: FirebaseUserDict):
    """Render the document content."""
    render_actions(user_info)
    if st.session_state.stream is not None:
        render_run_content(st.session_state.stream)

def render_actions(user_info: FirebaseUserDict):
    """Render the actions for the document content."""  
    if st.session_state.thread_id is not None:
        if st.button("Random action"):
            doc_agent = DocAgent.from_thread(
                thread_id=st.session_state.thread_id, user_id=user_info.get("localId")
            )
            stream = doc_agent.run_action("Random Country")
            render_run_content(stream)


def render_run_content(stream: Iterator[StreamPart]):
    # Create containers for different types of content
    with st.container():
        # Create a placeholder for the main content
        content_placeholder = st.empty()
        # Create a status indicator
        status = st.empty()
        
        # Initialize content buffer
        content_buffer = ""
        
        # Process each part of the stream
        for part in stream:
            # Handle status updates
            print(f"Part: {part}")
            if hasattr(part, 'data') and isinstance(part.data, dict):
                # Update status if available
                if 'status' in part.data:
                    status.text(f"Status: {part.data['status']}")
                
                # Handle values updates
                if 'values' in part.data:
                    values = part.data['values']
                    if 'messages' in values:
                        messages = values['messages']
                        if messages and len(messages) > 0:
                            # Get the latest message
                            latest_message = messages[-1]
                            if 'content' in latest_message:
                                content_buffer += latest_message['content']
                                content_placeholder.markdown(content_buffer)
            
            # Handle debug information if needed
            if hasattr(part, 'data') and isinstance(part.data, dict) and 'debug' in part.data:
                # You can optionally display debug information
                pass

