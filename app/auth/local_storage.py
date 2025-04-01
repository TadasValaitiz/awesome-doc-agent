import time
import streamlit as st
import json
from streamlit_local_storage import LocalStorage
from typing import Dict, Any, Optional, Union


class LStorage:
    def __init__(self):
        self.storage = LocalStorage(
            key="auth_storage",
        )

    def set(self, storage_key: str, value: Dict[str, Any]) -> None:
        """
        Store a dictionary value in browser's local storage.

        Args:
            storage_key: The key under which to store the value
            value: Dictionary to store
        """

        # Convert dictionary to JSON string before storing
        json_value = json.dumps(value)

        # Store in local storage using the global instance
        self.storage.setItem(itemKey=storage_key, itemValue=json_value)

        self.storage.getItem(storage_key)
        time.sleep(1)
        # Also keep a copy in session state for faster access
        st.session_state[storage_key] = value

    def get(self, storage_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a value from local storage.

        Args:
            storage_key: The key to retrieve

        Returns:
            The stored dictionary or None if not found
        """
        # First check session state for faster access
        if storage_key in st.session_state:
            return st.session_state[storage_key]

        # If not in session state, try to get from local storage
        json_value = self.storage.getItem(storage_key)

        # If value exists, convert JSON string back to dictionary
        if json_value is not None:
            try:
                value = json.loads(json_value)
                # Cache the result in session state for future access
                st.session_state[storage_key] = value
                return value
            except json.JSONDecodeError:
                return None

        return None

    def remove(self, storage_key: str) -> None:
        """
        Remove a value from local storage.

        Args:
            storage_key: The key to remove
        """
        # Remove from local storage
        self.storage.deleteItem(storage_key)
        time.sleep(1)

        # Also remove from session state
        if storage_key in st.session_state:
            del st.session_state[storage_key]
