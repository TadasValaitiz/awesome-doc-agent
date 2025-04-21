"""Utility & helper functions."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from typing import List, Optional, Tuple

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
import io
import base64
import pandas as pd
from langchain_core.documents import Document

from strategy_agent.configuration import Configuration


def get_message_text(msg: BaseMessage) -> str:
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
    fully_specified_name = configuration.chat_model
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = None
        model = fully_specified_name
    return init_chat_model(model, model_provider=provider)

def reciprocal_rank_fusion(results: List[List[Document]], k=60):
    """Reciprocal_rank_fusion that takes multiple lists of ranked documents
    and an optional parameter k used in the RRF formula"""

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}
    docs_map = {}
    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_id = doc.id
            docs_map[doc_id] = doc
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_id]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_id] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (docs_map[doc_id], score)
        for doc_id, score in sorted(
            fused_scores.items(), key=lambda x: x[1], reverse=True
        )
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


def take_top_k(ranked_docs: List[Tuple[Document, float]], k: int = 5) -> List[int]:
    return [doc.metadata.get("id", -1) for doc, score in ranked_docs[:k]]
