import os
import datetime
from typing import Dict, Any, Optional
import streamlit as st
import dotenv


def load_env_vars(env_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load environment variables from .env file.

    Args:
        env_path: Path to .env file or None to search in parent directories

    Returns:
        Dictionary of environment variables
    """
    # Find and load .env file
    found = dotenv.load_dotenv(dotenv_path=env_path, override=True)

    if not found:
        # Look in parent directory
        parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
        parent_env = os.path.join(parent_dir, ".env")
        dotenv.load_dotenv(dotenv_path=parent_env, override=True)



def init_session_state() -> None:
    """Initialize Streamlit session state variables if they don't exist."""
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    if "run" not in st.session_state:
        st.session_state.run = None
    if "stream" not in st.session_state:
        st.session_state.stream = None
    if "document_metadata" not in st.session_state:
        st.session_state.document_metadata = None
    if "run_id" not in st.session_state:
        st.session_state.run_id = None
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None


def set_page_config() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="AI Data cleaning",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
