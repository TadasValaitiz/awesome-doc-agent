# from functools import partial
# from operator import itemgetter
# import os
# from typing import Callable, List, Dict, Any, Literal, Optional, TypeGuard, cast
# import streamlit as st
# from pydantic import BaseModel, Field, SecretStr

# from langchain.output_parsers import PydanticOutputParser
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.callbacks.base import BaseCallbackHandler
# from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
# from shared import (
#     ChatMessage,
#     ContextDict,
#     UserStrategy,
#     RouteContext,
#     QaContext,
#     EvaluationContext,
# )
# from .prompts import (
#     routing,
#     rag_fusion,
#     trading_idea,
#     general_question,
#     question_with_context,
#     evaluation,
# )

# # Absolute imports from the root common module
# from common import TradingStrategyDefinition
# from common.utils import reciprocal_rank_fusion, take_top_k
# from database import VectorDB, Database


# class StreamHandler:
#     """Handler for streaming responses to Streamlit."""

#     def __init__(
#         self,
#         on_step_update: Callable[[str, List[str]], None],
#         on_reasoning_update: Callable[[str, str], None],
#         on_reasoning_finish: Callable[[str, str], None],
#     ):
#         self.text = {}
#         self.reasoning = {}
#         self.steps = []
#         self.current_step = "initial"
#         # Initialize the dictionaries with the initial step
#         self.text[self.current_step] = ""
#         self.reasoning[self.current_step] = ""
#         self.on_step_update = on_step_update
#         self.on_reasoning_update = on_reasoning_update
#         self.on_reasoning_finish = on_reasoning_finish

#     def reasoning_update(self, text_chunk: str) -> None:
#         """Update the displayed text with a new chunk."""
#         # Ensure the key exists before accessing it
#         if self.current_step not in self.text:
#             self.text[self.current_step] = ""
#             self.reasoning[self.current_step] = ""

#         self.text[self.current_step] += text_chunk
#         self.reasoning[self.current_step] = self.text[self.current_step] + "▌"
#         self.on_reasoning_update(self.current_step, self.reasoning[self.current_step])

#     def reasoning_finish(self) -> None:
#         """Finalize the displayed text."""
#         # Ensure the key exists before accessing it
#         if self.current_step not in self.reasoning:
#             return

#         self.reasoning[self.current_step] = self.text[self.current_step]
#         self.on_reasoning_update(self.current_step, self.reasoning[self.current_step])

#         # Call the reasoning_finish callback for the current step
#         self.on_reasoning_finish(self.current_step, self.reasoning[self.current_step])

#     def step_update(self, step: str) -> None:
#         """Update important step info"""
#         self.steps.append(step)
#         self.current_step = step
#         # Initialize the dictionaries for the new step
#         self.text[self.current_step] = ""
#         self.reasoning[self.current_step] = ""
#         self.on_step_update(step, self.steps)

#     def get_steps(self) -> List[str]:
#         """Get the steps."""
#         return self.steps

#     def get_text(self) -> Dict[str, str]:
#         """Get the full text."""
#         return self.text


# class StreamingCallback(BaseCallbackHandler):
#     """Callback handler for streaming tokens to a StreamHandler."""

#     def __init__(self, stream_handler: StreamHandler):
#         self.stream_handler = stream_handler

#     def on_llm_new_token(self, token: str, **kwargs) -> None:
#         """Run on new token."""
#         self.stream_handler.reasoning_update(token)

#     def on_llm_end(self, response, **kwargs) -> None:
#         """Run when LLM ends running"""
#         self.stream_handler.reasoning_finish()


# class AIService:
#     """Service for interacting with AI models using Langchain."""

#     def __init__(self):
#         """Initialize the AI service with a specific model."""

#         openai_api_key = None

#         try:
#             if "langchain" in st.secrets:
#                 os.environ["LANGCHAIN_API_KEY"] = st.secrets["langchain"]["api_key"]
#                 os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["langchain"]["endpoint"]
#                 os.environ["LANGCHAIN_TRACING_V2"] = str(
#                     st.secrets["langchain"].get("tracing_v2", "true")
#                 ).lower()

#             if "api_keys" in st.secrets:
#                 openai_api_key = st.secrets["api_keys"]["openai"]

#         except Exception as e:
#             raise ValueError("Secrets mechanism is not available")



#         if openai_api_key:
#             self.vector_db = VectorDB(openai_api_key)
#             self.db = Database()
#             self.openai_api_key = SecretStr(openai_api_key)
#         else:
#             raise ValueError("OPENAI_API_KEY environment variable is not set")

#     def load_strategies(self, ids: List[int]) -> List[TradingStrategyDefinition]:
#         return list(map(lambda x: x["strategy"], self.db.list_strategies_by_ids(ids)))

#     def generate_response_stream(
#         self,
#         message: str,
#         stream_handler: StreamHandler,
#         model: str = "gpt-4o-mini-2024-07-18",
#         initial_context: Optional[ContextDict] = None,
#     ) -> str | ContextDict:
#         """
#         Generate a response with streaming to a Streamlit container.

#         Args:
#             messages: List of message dictionaries with 'role' and 'content' keys
#             container: Streamlit container to stream to
#             system_prompt: Optional system prompt to prepend

#         Returns:
#             The complete generated response as a string
#         """
#         streaming_callback = StreamingCallback(stream_handler)

#         if initial_context is None:
#             initial_context = ContextDict.empty()
#         else:
#             initial_context = initial_context.new_run()

#         try:
#             llm = ChatOpenAI(
#                 model=model,
#                 api_key=self.openai_api_key,  # Convert string to SecretStr
#                 temperature=0.0,
#                 streaming=True,
#                 timeout=60,
#                 callbacks=[streaming_callback],
#             )

#             def answer_update(answer: str, context: ContextDict):
#                 context.conversations.append(QaContext(question=message, answer=answer))
#                 return context

#             # -------------------------------
#             # 1. Routing Chain
#             # -------------------------------
#             routing_prompt = ChatPromptTemplate.from_template(routing)
#             routing_parser = PydanticOutputParser(pydantic_object=RouteContext)

#             def route_update(route: RouteContext, context: ContextDict):
#                 context.route = route
#                 return context

#             router_chain = RunnableLambda(
#                 lambda dict: (
#                     (
#                         lambda _: {
#                             "question": dict.get("question"),
#                             "context": dict.get("context"),
#                             "format_instructions": routing_parser.get_format_instructions(),
#                         }
#                     )
#                     | RunnablePassthrough(
#                         lambda x: stream_handler.step_update("Routing")
#                     )
#                     | routing_prompt
#                     | llm
#                     | routing_parser
#                     | RunnableLambda(
#                         partial(route_update, context=dict.get("context_dict"))
#                     )
#                 )
#             )

#             # -------------------------------
#             # 2. General Message Chain
#             # -------------------------------
#             general_message_prompt = ChatPromptTemplate.from_template(general_question)
#             general_message_chain = RunnableLambda(
#                 lambda dict: (
#                     (
#                         lambda _: {
#                             "question": dict.get("question"),
#                         }
#                     )
#                     | RunnablePassthrough(
#                         lambda x: stream_handler.step_update("Answering Message")
#                     )
#                     | general_message_prompt
#                     | llm
#                     | StrOutputParser()
#                     | RunnableLambda(
#                         partial(answer_update, context=dict.get("context_dict"))
#                     )
#                 )
#             )

#             # -------------------------------
#             # 2. Answer with Context Chain
#             # -------------------------------
#             question_with_context_prompt = ChatPromptTemplate.from_template(
#                 question_with_context
#             )
#             question_with_context_chain = RunnableLambda(
#                 lambda dict: (
#                     (
#                         lambda _: {
#                             "question": dict.get("question"),
#                             "context": dict.get("context"),
#                         }
#                     )
#                     | RunnablePassthrough(
#                         lambda x: stream_handler.step_update(
#                             "Answering Message with Context"
#                         )
#                     )
#                     | question_with_context_prompt
#                     | llm
#                     | StrOutputParser()
#                     | RunnableLambda(
#                         partial(answer_update, context=dict.get("context_dict"))
#                     )
#                 )
#             )

#             # -------------------------------
#             # 3. Trading Idea Chain
#             # -------------------------------

#             def tradin_idea_update(trading_idea: UserStrategy, context: ContextDict):
#                 context.user_strategy = trading_idea
#                 answer = (
#                     trading_idea.direct_answer if trading_idea.direct_answer else ""
#                 )
#                 if (
#                     trading_idea.followup_questions
#                     and len(trading_idea.followup_questions) > 0
#                 ):
#                     answer += f"\nFollow-up questions:\n"
#                     for i, q in enumerate(trading_idea.followup_questions):
#                         answer += f"{i+1}. {q}\n"

#                 context.conversations.append(QaContext(question=message, answer=answer))
#                 return context

#             trading_idea_template = ChatPromptTemplate.from_template(trading_idea)
#             trading_idea_parser = PydanticOutputParser(pydantic_object=UserStrategy)

#             trading_idea_chain = RunnableLambda(
#                 lambda dict: (
#                     (
#                         lambda _: {
#                             "context": dict.get("context"),
#                             "question": dict.get("question"),
#                             "format_instructions": trading_idea_parser.get_format_instructions(),
#                         }
#                     )
#                     | RunnablePassthrough(
#                         lambda x: stream_handler.step_update(
#                             step="Generating Trading Idea"
#                         )
#                     )
#                     | trading_idea_template
#                     | llm
#                     | trading_idea_parser
#                     | RunnableLambda(
#                         partial(tradin_idea_update, context=dict.get("context_dict"))
#                     )
#                 )
#             )

#             # -------------------------------
#             # 4. Rag Fusion Strategy Chain
#             # -------------------------------

#             def strategies_update(
#                 strategies: List[TradingStrategyDefinition], context: ContextDict
#             ):
#                 context.rag_strategies = strategies
#                 return context

#             rag_fusion_prompt = ChatPromptTemplate.from_template(rag_fusion)

#             rag_fusion_chain = RunnableLambda(
#                 lambda dict: (
#                     (
#                         lambda x: {
#                             "question": dict.get("question"),
#                             "context": dict.get("context"),
#                         }
#                     )
#                     | rag_fusion_prompt
#                     | RunnablePassthrough(
#                         lambda x: stream_handler.step_update(step="Querying Rag")
#                     )
#                     | llm
#                     | StrOutputParser()
#                     | (lambda x: x.split("\n"))
#                     | self.vector_db.strategy_retriever().map()
#                     | reciprocal_rank_fusion
#                     | partial(take_top_k, k=5)
#                     | self.load_strategies
#                     | RunnableLambda(
#                         partial(strategies_update, context=dict.get("context_dict"))
#                     )
#                 )
#             )

#             # -------------------------------
#             # 5. Evaluation Chain
#             # -------------------------------

#             def evaluation_update(evaluation: EvaluationContext, context: ContextDict):
#                 context.evaluation = evaluation
#                 return context

#             evaluation_prompt = ChatPromptTemplate.from_template(evaluation)
#             evaluation_parser = PydanticOutputParser(pydantic_object=EvaluationContext)
#             evaluation_chain = RunnableLambda(
#                 lambda dict: (
#                     (
#                         lambda _: {
#                             "context": dict.get("context"),
#                             "format_instructions": evaluation_parser.get_format_instructions(),
#                         }
#                     )
#                     | RunnablePassthrough(
#                         lambda x: stream_handler.step_update(step="Evaluating")
#                     )
#                     | evaluation_prompt
#                     | llm
#                     | evaluation_parser
#                     | RunnableLambda(
#                         partial(evaluation_update, context=dict.get("context_dict"))
#                     )
#                 )
#             )

#             # -------------------------------
#             # 5. Route Chain
#             # -------------------------------

#             def route_type(
#                 obj: Any,
#             ) -> Literal[
#                 "non-related", "instruction", "question", "evaluation", "follow-up"
#             ]:
#                 if not isinstance(obj, ContextDict):
#                     return "non-related"
#                 if obj.route is None:
#                     return "non-related"
#                 if obj.route.message_type is None:
#                     return "non-related"
#                 if obj.route.message_type == "follow-up":
#                     return "follow-up"
#                 return obj.route.message_type

#             def input(context: ContextDict, fn: Callable[[ContextDict], str]):
#                 return {
#                     "question": message,
#                     "context": fn(context),
#                     "context_dict": context,
#                 }

#             # ------------------------------
#             # 5.1. Instruction Chain
#             # ------------------------------

#             instruction_branch = (
#                 RunnableLambda(
#                     partial(input, fn=lambda c: c.strategy_with_conversation())
#                 )
#                 | rag_fusion_chain
#                 | RunnableLambda(partial(input, fn=lambda c: c.strategy_with_rag()))
#                 | trading_idea_chain
#             )

#             # ------------------------------
#             # 5.2. Question Chain
#             # ------------------------------

#             question_branch = (
#                 RunnableLambda(
#                     partial(input, fn=lambda c: c.strategy_with_conversation())
#                 )
#                 | rag_fusion_chain
#                 | RunnableLambda(
#                     partial(
#                         input,
#                         fn=lambda c: c.rag_context(),
#                     )
#                 )
#                 | question_with_context_chain
#             )

#             # ------------------------------
#             # 5.3. Follow-up Chain
#             # ------------------------------

#             followup_branch = (
#                 RunnableLambda(
#                     partial(input, fn=lambda c: c.strategy_with_conversation())
#                 )
#             ) | trading_idea_chain

#             # ------------------------------
#             # 5.4. Non Related Chain and Default Chain
#             # ------------------------------

#             general_branch = (
#                 RunnableLambda(
#                     partial(input, fn=lambda c: c.to_full_strategy_context())
#                 )
#                 | general_message_chain
#             )

#             # ------------------------------
#             # 5.5. Evaluation Chain
#             # ------------------------------

#             evaluation_branch = (
#                 RunnableLambda(
#                     partial(input, fn=lambda c: c.to_full_strategy_context())
#                 )
#                 | evaluation_chain
#             )

#             chain = router_chain | RunnableBranch(
#                 (lambda c: route_type(c) == "non-related", general_branch),
#                 (lambda c: route_type(c) == "instruction", instruction_branch),
#                 (lambda c: route_type(c) == "follow-up", followup_branch),
#                 (lambda c: route_type(c) == "question", question_branch),
#                 (lambda x: route_type(x) == "evaluation", evaluation_branch),
#                 general_branch,
#             )

#             # Execute the branch chain and get the result
#             result: ContextDict = chain.invoke(
#                 input(initial_context, lambda c: c.router_context())
#             )
#             return result

#         except Exception as e:
#             import traceback
#             import sys

#             # Get the full stack trace
#             exc_type, exc_value, exc_traceback = sys.exc_info()
#             stack_trace = traceback.format_exception(exc_type, exc_value, exc_traceback)
#             stack_trace_str = "".join(stack_trace)

#             # Check if OpenAI API key exists and is valid
#             api_key_info = ""
#             try:
#                 # Get the first few and last few characters of the API key for debugging
#                 # (avoid exposing the full key for security)
#                 if hasattr(self, "openai_api_key") and self.openai_api_key:
#                     api_key_value = self.openai_api_key.get_secret_value()
#                     if api_key_value:
#                         prefix = api_key_value[:4]
#                         suffix = api_key_value[-4:] if len(api_key_value) > 8 else ""
#                         masked_key = f"{prefix}...{suffix}"
#                         api_key_info += f"✅ API Key exists (starts with {prefix}, ends with {suffix})\n"
#                         api_key_info += f"Length: {len(api_key_value)} characters\n"
#                     else:
#                         api_key_info += "❌ API Key exists but is empty\n"
#                 else:
#                     api_key_info += "❌ API Key is not set\n"
#             except Exception as key_error:
#                 api_key_info += f"❌ Error checking API key: {str(key_error)}\n"

#             # Format the error message with stack trace and API key info
#             error_message = f"""
# Error Details

# ```
# {type(e).__name__}
# Error Message: {str(e)}
# ```
# API Key Status:
# ```
# {api_key_info}
# ```
# Stack Trace:
# ```
# {stack_trace_str}
# ```
# """
#             # Return the detailed error message
#             return error_message
