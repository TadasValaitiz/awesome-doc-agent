import chromadb
from chromadb.api import ClientAPI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from langgraph.store.base import BaseStore, Item, SearchItem, GetOp, SearchOp, PutOp
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union, cast
from datetime import datetime
import json
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class VectorDB:
    db: ClientAPI

    def __init__(self):
        self.db = chromadb.PersistentClient("data/chroma-db")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def vectorstore(self, collection_name: str):
        if self.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        return Chroma(
            client=self.db,
            collection_name=collection_name,
            embedding_function=OpenAIEmbeddings(api_key=SecretStr(self.openai_api_key)),
        )

    def strategy_store(self):
        return self.vectorstore("strategy")

    def strategy_retriever(self):
        return self.strategy_store().as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {"theme": "strategy"},
            },
        )


class AsyncVectorDB:
    def __init__(self):
        self.db = None
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    async def get_db(self):
        """Initialize the ChromaDB client in a non-blocking way"""
        if self.db is None:
            # Run PersistentClient initialization in a separate thread
            self.db = await asyncio.to_thread(
                chromadb.PersistentClient, "data/chroma-db"
            )
        return self.db

    async def vectorstore(self, collection_name: str):
        """Create a vectorstore in a non-blocking way"""
        if self.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # Get the database client
        db = await self.get_db()

        # Create embedding function
        embedding_function = OpenAIEmbeddings(api_key=SecretStr(self.openai_api_key))

        # Create Chroma instance in a thread to avoid blocking
        return await asyncio.to_thread(
            Chroma,
            client=db,
            collection_name=collection_name,
            embedding_function=embedding_function,
        )

    async def strategy_store(self):
        """Get the strategy vectorstore"""
        return await self.vectorstore("strategy")

    async def strategy_retriever(self):
        """Get the strategy retriever in a non-blocking way"""
        strategy_store = await self.strategy_store()

        # Create retriever in a thread to avoid blocking
        retriever = await asyncio.to_thread(
            strategy_store.as_retriever,
            search_kwargs={
                "k": 5,
                "filter": {"theme": "strategy"},
            },
        )

        return retriever
