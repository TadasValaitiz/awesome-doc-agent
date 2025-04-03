"""Utility functions used in our graph."""

from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from pandas import DataFrame

from enrichment_agent.configuration import Configuration


def get_message_text(msg: AnyMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def init_model(config: Optional[RunnableConfig] = None) -> BaseChatModel:
    """Initialize the configured chat model."""
    configuration = Configuration.from_runnable_config(config)
    fully_specified_name = configuration.model
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = None
        model = fully_specified_name
    return init_chat_model(model, model_provider=provider)


def pandas_to_markdown(df: DataFrame) -> str:
    """Convert a pandas DataFrame to a markdown table."""
    return df.to_markdown()


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
