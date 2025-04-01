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

    # Return relevant environment variables
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
    }



def init_session_state() -> None:
    """Initialize Streamlit session state variables if they don't exist."""
    if "user_id" not in st.session_state:
        st.session_state.user_id = None

    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None

    if "is_sending_message" not in st.session_state:
        st.session_state.is_sending_message = False


def set_page_config() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="AI Data cleaning",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
