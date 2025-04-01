import chromadb
from chromadb.api import ClientAPI
from chromadb import Collection
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr


class VectorDB:
    db: ClientAPI

    def __init__(self, openai_api_key: str | None):
        self.db = chromadb.PersistentClient("data/chroma-db")
        self.openai_api_key = openai_api_key


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
