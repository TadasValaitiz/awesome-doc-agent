import streamlit as st
from typing import Dict, Any, Optional, Callable
from auth import FirebaseUserDict


def render_navbar(
    user_info: Optional[FirebaseUserDict],
    on_logout: Callable[[], None],
) -> None:
    """
    Render the top navigation bar with user account functionality.

    Args:
        user_info: User information dictionary or None if not logged in
        on_login: Callback for login action
        on_logout: Callback for logout action
    """
    if not user_info:
        return

    # Create a container for the navbar
    navbar_container = st.container()

    with navbar_container:
        col1, spacer, col2 = st.columns([5, 1, 2], gap="large")

        # App logo/name
        with col1:
            st.markdown("#### AI Document assistant ")

        with spacer:
            st.write("")

        # User account section
        with col2:
            # User is logged in
            login_type = user_info.get("login_type", "anonymous")
            name = (
                "Anonymous User"
                if login_type == "anonymous"
                else user_info.get("email", "None")
            )

            # Display user info and logout button
            with st.popover(f"👤 {name}", use_container_width=True):
                if st.button("Logout", key="logout_btn"):
                    on_logout()
