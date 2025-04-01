"""Define a generic chat agent agent."""

from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from pydantic import BaseModel, Field

from generic_chat_agent import prompts
from generic_chat_agent.configuration import Configuration
from generic_chat_agent.state import State
from generic_chat_agent.utils import init_model


def chat_with_agent(state: State, *, config: Optional[RunnableConfig] = None):
    configuration = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_template(configuration.prompt)
    raw_model = init_model(config)

    def extract_prompt(s: State):
        def serialize_message(msg: BaseMessage) -> str:
            return f"{msg.type}: {msg.content}"

        return  {
            "question": s.messages[-1].content,
            "messages": "\n".join([serialize_message(msg) for msg in s.messages[:-1]])
        }

    chain = extract_prompt | prompt | raw_model
    assistant_message = chain.invoke(state)
    return Command(update=State(messages=[assistant_message], loop_step=1))


workflow = StateGraph(State, config_schema=Configuration)
workflow.add_node("chat_with_agent", chat_with_agent)
workflow.add_edge(START, "chat_with_agent")
workflow.add_edge("chat_with_agent", END)

graph = workflow.compile()
graph.name = "GenericChatAgent"
