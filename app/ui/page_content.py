import streamlit as st
from auth import FirebaseUserDict, FirebaseAuth
from typing import List, Optional
from .login import login_page

welcome = """

Your AI-powered assistant for developing and evaluating trading strategies.

### What You Can Do:

* **Build Trading Strategies** - Create custom strategies with specific entry/exit conditions
* **Research Technical Indicators** - Learn about indicators and how to apply them effectively
* **Evaluate Performance** - Get AI-powered feedback on your strategy's strengths and weaknesses
* **Compare with Existing Approaches** - See how your ideas stack up against established methods

Start by New Conversation and describe your trading idea!
"""


def render_page_content(
    user_info: Optional[FirebaseUserDict],
    firebase_auth: FirebaseAuth,
):

    if user_info is None:
        login_page(firebase_auth)
    else:
        render_conversation()


def render_conversation():
    st.markdown(welcome)

