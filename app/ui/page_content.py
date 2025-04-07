import streamlit as st
import pandas as pd
from service.doc_agent import DocAgent
from common.types import DocumentMetadata
from auth import FirebaseUserDict, FirebaseAuth
from typing import List, Optional, cast

from .login import login_page
from .document_content import render_document_content

welcome = """

Your AI-powered assistant for inspecting and analyzing documents.

- 🔍 Automatically detect data quality issues
- 🧹 Clean missing values, outliers, and duplicates
- 📊 Generate data quality reports
- ⬇️ Download your cleaned dataset

Upload a CSV or Excel file to get started!


"""


def validate_data(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Basic validation of the uploaded data.

    Args:
        df: Pandas DataFrame to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "The uploaded file is empty"

    if len(df.columns) == 0:
        return False, "The file has no columns"

    return True, "Data validation successful"


def render_page_content(
    user_info: Optional[FirebaseUserDict],
    firebase_auth: FirebaseAuth,
):
    if user_info is None:
        login_page(firebase_auth)
    else:
        if st.session_state.thread_id is None:
            render_file_upload()
            render_file_preview(user_info)
        else:
            render_document_content(user_info)


def clear_file_state():
    st.session_state.uploaded_file = None
    st.session_state.document_metadata = None


def render_file_upload():

    # Initialize session state for uploaded file if not exists
    if "uploaded_file" not in st.session_state:
        st.markdown(welcome)
        st.session_state.document_metadata = None
        st.session_state.thread_id = None
        st.session_state.run_id = None

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your data file (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        key="file_uploader",
        accept_multiple_files=False,
        on_change=clear_file_state,
    )

    if uploaded_file is not None:
        try:
            # Read the file based on its type
            if uploaded_file.name.endswith(".csv"):
                # Try different delimiters
                try:
                    # First try comma
                    df = pd.read_csv(uploaded_file, sep=",")
                    # Check if we got only one column (likely wrong delimiter)
                    if len(df.columns) == 1:
                        raise ValueError(
                            "Data appears to be single column, likely wrong delimiter"
                        )
                except Exception:
                    # If comma fails, try semicolon
                    uploaded_file.seek(0)  # Reset file pointer
                    try:
                        df = pd.read_csv(uploaded_file, sep=";")
                    except Exception as e:
                        st.error(
                            f"Error reading CSV file with both comma and semicolon delimiters: {str(e)}"
                        )
                        st.session_state.uploaded_file = None
                        st.session_state.df = None
                        return
            else:
                df = pd.read_excel(uploaded_file)

            # Basic validation
            is_valid, validation_message = validate_data(df)

            if not is_valid:
                st.error(validation_message)
                st.session_state.uploaded_file = None
                st.session_state.df = None
                return

            # Store in session state
            st.session_state.uploaded_file = uploaded_file
            st.session_state.document_metadata = DocumentMetadata(
                uploaded_file.name, df
            )

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.session_state.uploaded_file = None
            st.session_state.df = None


def render_file_preview(user_info: FirebaseUserDict):
    """Render the content for the uploaded file."""
    if st.session_state.document_metadata is not None:
        metadata = cast(DocumentMetadata, st.session_state.document_metadata)
        # Display data info
        with st.expander("Preview", expanded=False):
            render_document_metadata(metadata)

        # Display the data
        st.dataframe(metadata.df.head(n=5), use_container_width=True)
        if st.button("Procced with document analysis"):
            doc_service = DocAgent(user_id=user_info.get("localId"))
            thread = doc_service.new_thread(metadata)
            st.session_state.thread_id = thread.get("thread_id")
            st.rerun()


def render_document_metadata(metadata: DocumentMetadata):
    """Render the metadata for the uploaded file."""
    st.write(f"Number of rows: {metadata.num_rows}")
    st.write(f"Number of columns: {len(metadata.columns)}")
    st.write(f"Columns: {', '.join(str(col) for col in metadata.columns)}")
