import time
from typing import Optional
import streamlit as st
from auth import FirebaseAuth, FirebaseUserDict


def login_page(firebase_auth: FirebaseAuth):
    """
    Display the login page with authentication options
    """
    st.title("AI Strategy Builder")
    st.subheader("Research and create trading strategies with AI")

    # Create tabs for different authentication options
    login_tab, register_tab = st.tabs(["Login", "Register"])

    # Login tab with email/password and anonymous options
    with login_tab:
        # Email/Password Login Section
        st.subheader("Login with Email")
        with st.form("email_login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")

            if submit_button:
                if not email or not password:
                    st.error("Please enter both email and password")
                else:
                    success, user = firebase_auth.email_password_login(email, password)
                    if success:
                        st.success("Login successful!")
                        handle_login( user)
                    else:
                        st.error("Invalid email or password")

        # Anonymous Login Section
        st.markdown("---")
        st.subheader("Or continue as guest")
        if st.button("Anonymous Login"):
            success, user = firebase_auth.anonymous_login()
            if success:
                st.success("Logged in anonymously!")
                handle_login( user)
            else:
                st.error("Anonymous login failed")

    # Registration tab
    with register_tab:
        st.subheader("Create an Account")
        with st.form("registration_form"):
            reg_email = st.text_input("Email")
            reg_password = st.text_input("Password", type="password")
            reg_confirm_password = st.text_input("Confirm Password", type="password")
            register_button = st.form_submit_button("Register")

            if register_button:
                if not reg_email or not reg_password or not reg_confirm_password:
                    st.error("Please fill in all fields")
                elif reg_password != reg_confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, user = firebase_auth.create_user(reg_email, reg_password)
                    if success:
                        st.success("Account created successfully!")
                        # Send verification email
                        firebase_auth.send_verification_email()
                        st.info(
                            "A verification email has been sent to your email address"
                        )
                        st.rerun()
                    else:
                        st.error("Registration failed. Email may already be in use.")

    # Show user info and logout button if logged in
    if firebase_auth.is_logged_in():
        user = firebase_auth.get_current_user()
        with st.sidebar:
            if user:
                st.write(f"Logged in as: {user.get('email', 'Anonymous User')}")
            else:
                st.write("Logged in")

            if st.button("Logout"):
                firebase_auth.logout()
                st.rerun()


def handle_login(user_info: Optional[FirebaseUserDict]):
    """Handle user login."""

    if user_info:
   
        time.sleep(1)
        st.rerun()
