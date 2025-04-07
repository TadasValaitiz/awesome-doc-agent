import json
import streamlit as st
from typing import List, Dict, Any, Optional, Callable
from common.types import DocumentMetadata
from service.doc_agent_manager import DocAgentManager
from auth import FirebaseUserDict
from datetime import datetime
import humanize


def render_sidebar(
    user_info: Optional[FirebaseUserDict],
) -> None:
    with st.sidebar:
        if user_info is not None:
            if st.button("Start New thread", use_container_width=True):
                st.session_state.thread_id = None
                st.rerun()
            st.markdown("## Document threads")

            doc_agent_manager = DocAgentManager(user_info.get("localId"))
            threads = doc_agent_manager.list_threads()

            for thread in threads:
                metadata_json = thread.get("metadata", {})
                thread_id = thread.get("thread_id")
                status = thread.get("status", "Unknown")
                updated_at = thread.get("updated_at", None)

                if metadata_json:
                    file_name = metadata_json.get("file_name", "Unnamed Document")

                    # Format the date if it exists
                    if updated_at:
                        try:
                            # Convert the updated_at to datetime
                            updated_datetime = datetime.fromisoformat(str(updated_at))
                            current_time = datetime.utcnow()
                            # If updated_datetime is timezone-aware, convert current_time to UTC timezone-aware
                            if updated_datetime.tzinfo is not None:
                                from datetime import timezone

                                current_time = datetime.now(timezone.utc)
                            # Get relative time using humanize
                            last_updated_date = humanize.naturaltime(
                                current_time - updated_datetime
                            )
                        except:
                            last_updated_date = "Unknown date"
                    else:
                        last_updated_date = "Not available"

                    # Create a container for each thread
                    with st.container():
                        col1, col2 = st.columns([8, 2])
                        with col1:
                            st.markdown(f"**{file_name}**")
                            st.markdown(f"{thread_id}")
                            st.markdown(f"{status.upper()} • {last_updated_date}")
                        with col2:
                            if st.button("Open", key=f"thread_{thread_id}"):
                                st.session_state.thread_id = thread_id
                                st.rerun()
                        st.divider()
                else:
                    st.button("No metadata", key=f"thread_{thread_id}")
