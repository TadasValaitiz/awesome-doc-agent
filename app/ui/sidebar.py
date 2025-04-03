import json
import streamlit as st
from typing import List, Dict, Any, Optional, Callable
from common.types import DocumentMetadata
from service.doc_agent_manager import DocAgentManager
from auth import FirebaseUserDict

def render_sidebar(
    user_info: Optional[FirebaseUserDict],
) -> None:
    with st.sidebar:
        if user_info is not None:
            st.markdown("## Document threads")
            doc_agent_manager = DocAgentManager(user_info.get("localId"))
            threads = doc_agent_manager.list_threads()
            for thread in threads:
                metadata_json = thread.get("metadata", {})
                if metadata_json:
                    file_name = metadata_json.get('file_name', None)
                    if file_name:
                        if st.button(file_name):
                            st.session_state.thread_id = thread.get("thread_id")
                            st.rerun()
                else:
                    st.button("No metadata")
