import os
import streamlit as st
from typing import Dict, Any, Optional, List
from auth import FirebaseAuth, FirebaseUserDict
from ui import (
    render_sidebar,
    render_navbar,
    render_page_content,
)
from utils import (
    load_env_vars,
    init_session_state,
    set_page_config,
)

load_env_vars()

set_page_config()

init_session_state()

firebase_auth = FirebaseAuth()


def handle_logout():
    """Handle user logout."""
    firebase_auth.logout()
    st.session_state.current_conversation_id = None
    st.rerun()


def main():
    """Main application function."""
    current_user = firebase_auth.get_current_user()

    render_navbar(user_info=current_user, on_logout=handle_logout)

    render_sidebar(user_info=current_user)

    render_page_content(
        user_info=current_user,
        firebase_auth=firebase_auth,
    )


if __name__ == "__main__":
    main()
