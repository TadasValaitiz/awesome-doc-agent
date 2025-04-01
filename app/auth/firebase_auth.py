import streamlit as st
from typing import Dict, Any, Optional, Tuple, cast

from .types import FirebaseUserDict
from .auth_functions import (
    sign_in_anonymous,
    sign_in_with_email_and_password,
    create_user_with_email_and_password,
    send_email_verification,
    send_password_reset_email,
    get_account_info,
)
from .local_storage import LStorage


class FirebaseAuth:
    def __init__(self):
        """
        Initialize Firebase authentication with the provided configuration.

        Args:
            config: Firebase configuration dictionary
        """
        self.storage = LStorage()
        # Try to restore user from local storage if not in session state
        if "firebase_user" not in st.session_state:
            stored_user = self.storage.get("firebase_user")
            if stored_user:
                # Cast to the correct type for type checking
                user_data = cast(FirebaseUserDict, stored_user)
                st.session_state["firebase_user"] = user_data
                print("Restored user from local storage")

    def anonymous_login(self) -> Tuple[bool, Optional[FirebaseUserDict]]:
        """
        Perform anonymous login using Firebase.

        Returns:
            Tuple of (success, user_info)
        """
        try:
            # Check if user is already logged in via session state
            if "firebase_user" in st.session_state:
                return True, cast(FirebaseUserDict, st.session_state["firebase_user"])

            # Use the sign_in_anonymous function from auth_functions
            user_info = sign_in_anonymous()
            if user_info is None:
                return False, None
            # Convert to proper type with all required fields
            firebase_user: FirebaseUserDict = {
                "localId": user_info["localId"],
                "idToken": user_info["idToken"],
                "refreshToken": user_info["refreshToken"],
                "expiresIn": user_info["expiresIn"],
                "login_type": user_info["login_type"],
                "auth_provider": user_info["auth_provider"],
                "email": None,
                "name": None,
            }

            # Store in session state
            st.session_state["firebase_user"] = firebase_user
            # Also store in local storage for persistence across sessions
            self.storage.set("firebase_user", cast(Dict[str, Any], firebase_user))

            return True, firebase_user
        except Exception as e:
            # Handle error but don't show in UI to avoid confusing user
            print(f"Authentication error: {str(e)}")
            return False, None

    def email_password_login(
        self, email: str, password: str
    ) -> Tuple[bool, Optional[FirebaseUserDict]]:
        """
        Perform email/password login using Firebase.

        Args:
            email: User's email address
            password: User's password

        Returns:
            Tuple of (success, user_info)
        """
        try:
            # Use the sign_in_with_email_and_password function from auth_functions
            user_info = sign_in_with_email_and_password(email, password)
            if user_info is None:
                return False, None

            # Convert to proper type with all required fields
            firebase_user: FirebaseUserDict = {
                "localId": user_info["localId"],
                "idToken": user_info["idToken"],
                "refreshToken": user_info["refreshToken"],
                "expiresIn": user_info["expiresIn"],
                "login_type": "password",
                "auth_provider": "password",
                "email": user_info.get("email"),
                "name": None,  # Could be populated from account info if needed
            }

            # Store in session state
            st.session_state["firebase_user"] = firebase_user
            # Also store in local storage for persistence across sessions
            self.storage.set("firebase_user", cast(Dict[str, Any], firebase_user))

            return True, firebase_user
        except Exception as e:
            print(f"Email/password login error: {str(e)}")
            return False, None

    def create_user(
        self, email: str, password: str
    ) -> Tuple[bool, Optional[FirebaseUserDict]]:
        """
        Create a new user with email and password.

        Args:
            email: User's email address
            password: User's password

        Returns:
            Tuple of (success, user_info)
        """
        try:
            # Use the create_user_with_email_and_password function from auth_functions
            user_info = create_user_with_email_and_password(email, password)
            if user_info is None:
                return False, None

            # Convert to proper type with all required fields
            firebase_user: FirebaseUserDict = {
                "localId": user_info["localId"],
                "idToken": user_info["idToken"],
                "refreshToken": user_info["refreshToken"],
                "expiresIn": user_info["expiresIn"],
                "login_type": "password",
                "auth_provider": "password",
                "email": user_info.get("email"),
                "name": None,
            }

            # Automatically log in the user after account creation
            st.session_state["firebase_user"] = firebase_user
            self.storage.set("firebase_user", cast(Dict[str, Any], firebase_user))

            return True, firebase_user
        except Exception as e:
            print(f"User creation error: {str(e)}")
            return False, None

    def send_verification_email(self) -> bool:
        """
        Send email verification to the current user.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current user
            user = self.get_current_user()
            if user is None or user.get("idToken") is None:
                return False

            # Send verification email using the id token
            send_email_verification(user["idToken"])
            return True
        except Exception as e:
            print(f"Email verification error: {str(e)}")
            return False

    def reset_password(self, email: str) -> bool:
        """
        Send password reset email to the specified email address.

        Args:
            email: Email address to send password reset to

        Returns:
            True if successful, False otherwise
        """
        try:
            send_password_reset_email(email)
            return True
        except Exception as e:
            print(f"Password reset error: {str(e)}")
            return False

    def logout(self) -> bool:
        """
        Logout the current user.

        Returns:
            True if successful, False otherwise
        """
        try:
            if "firebase_user" in st.session_state:
                del st.session_state["firebase_user"]

            # Also remove from local storage
            self.storage.remove("firebase_user")
            return True
        except Exception:
            return False

    def get_current_user(self) -> Optional[FirebaseUserDict]:
        """
        Get the currently logged in user.

        Returns:
            User info dictionary or None if not logged in
        """
        if "firebase_user" in st.session_state:
            return cast(FirebaseUserDict, st.session_state["firebase_user"])

        # Try to get from local storage as fallback
        storage_user = self.storage.get("firebase_user")
        if storage_user is not None:
            return cast(FirebaseUserDict, storage_user)

        return None

    def is_logged_in(self) -> bool:
        """
        Check if a user is currently logged in.

        Returns:
            True if logged in, False otherwise
        """
        # First check session state as it's faster
        if "firebase_user" in st.session_state:
            return True

        # Then check local storage as fallback
        storage_user = self.storage.get("firebase_user")
        return storage_user is not None
