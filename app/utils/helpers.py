import os
import datetime
from typing import Dict, Any, Optional
from pandas import DataFrame
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


def panda_sample() -> DataFrame:
    """Create a sample DataFrame with data inconsistencies for testing cleanup agents.

    Returns:
        DataFrame: A DataFrame with 20 rows and various data inconsistencies.
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate base data
    n_rows = 20

    # Generate IDs
    ids = list(range(1, n_rows + 1))

    # Generate dates with one inconsistent format
    base_date = datetime.now() - timedelta(days=365)
    dates = [
        (base_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_rows)
    ]
    # Make one date inconsistent (European format)
    dates[5] = "15/03/2023"

    # Generate fullnames with one incorrect format
    first_names = [
        "John",
        "Jane",
        "Michael",
        "Sarah",
        "David",
        "Emma",
        "Robert",
        "Lisa",
        "William",
        "Emily",
        "James",
        "Jennifer",
        "Thomas",
        "Mary",
        "Daniel",
        "Patricia",
        "Joseph",
        "Linda",
        "Charles",
        "Barbara",
    ]
    last_names = [
        "Smith",
        "Johnson",
        "Williams",
        "Brown",
        "Jones",
        "Garcia",
        "Miller",
        "Davis",
        "Rodriguez",
        "Martinez",
        "Hernandez",
        "Lopez",
        "Gonzalez",
        "Wilson",
        "Anderson",
        "Thomas",
        "Taylor",
        "Moore",
        "Jackson",
        "Martin",
    ]

    fullnames = [f"{first} {last}" for first, last in zip(first_names, last_names)]
    # Make one fullname incorrect (missing space)
    fullnames[10] = "JamesWilson"

    # Generate emails with one duplicate
    emails = [
        f"{first.lower()}.{last.lower()}@example.com"
        for first, last in zip(first_names, last_names)
    ]
    # Make one email duplicate
    emails[15] = emails[3]  # Duplicate of 'sarah.brown@example.com'

    # Generate average_order_value with mixed currencies
    avg_order_values = np.random.uniform(10, 200, n_rows).round(2).astype(str)
    # Convert some to string with currency symbols
    avg_order_values[7] = f"€{avg_order_values[7]}"
    avg_order_values[12] = f"£{avg_order_values[12]}"
    avg_order_values[18] = f"${avg_order_values[18]}"

    # Generate number_of_purchases with one incorrect type
    num_purchases = np.random.randint(1, 20, n_rows)
    # Make one incorrect type (string instead of integer)
    num_purchases[3] = 0.5

    # Create DataFrame
    df = pd.DataFrame(
        {
            "id": ids,
            "date": dates,
            "fullname": fullnames,
            "email": emails,
            "average_order_value": avg_order_values,
            "number_of_purchases": num_purchases,
        }
    )

    return df
