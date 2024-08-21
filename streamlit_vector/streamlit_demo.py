import os
import streamlit as st
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.base import BaseEvent
from llama_index.core.instrumentation.events.llm import LLMChatEndEvent, LLMChatInProgressEvent, LLMCompletionEndEvent
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


class OpenAITokenCounter(BaseEventHandler):
    """Custom event handler to count tokens used in OpenAI completions."""

    prev_prompt_token_count: int = 0
    prev_response_token_count: int = 0

    @property
    def prev_total_token_count(self):
        return self.prev_prompt_token_count + self.prev_response_token_count

    def handle(self, event: BaseEvent, **kwargs) -> None:
        """Logic for getting token counts from raw openai responses."""
        raw_response = None

        if isinstance(event, LLMChatEndEvent):
            raw_response = event.response.raw
        elif isinstance(event, LLMCompletionEndEvent):
            raw_response = event.response.raw
        elif isinstance(event, LLMChatInProgressEvent):
            raw_response = event.response.raw

        if raw_response and hasattr(raw_response, "usage"):
            self.prev_response_token_count = raw_response.usage.completion_tokens
            self.prev_prompt_token_count = raw_response.usage.prompt_tokens


index_name = "./saved_index"
documents_folder = "./documents"

dispatcher = get_dispatcher()

if "token_counter" not in st.session_state:
    st.session_state.token_counter = OpenAITokenCounter()
    dispatcher.add_event_handler(st.session_state.token_counter)


def initialize_index(index_name, documents_folder):
    # load or create index
    embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
    if os.path.exists(index_name):
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name),
            embed_model=embed_model,
        )
    else:
        documents = SimpleDirectoryReader(documents_folder).load_data()
        index = VectorStoreIndex.from_documents(
            documents, embed_model=embed_model,
        )
        index.storage_context.persist(persist_dir=index_name)

    return index


def chat_with_data(query_text):
    chat_engine = st.session_state.chat_engine
    response = chat_engine.stream_chat(query_text)
    return response


st.title("🦙 Llama Index Demo 🦙")
st.header("Welcome to the Llama Index Streamlit Demo")
st.write(
    "Enter a query about Paul Graham's essays. You can check out the original essay [here](https://raw.githubusercontent.com/jerryjliu/llama_index/main/examples/paul_graham_essay/data/paul_graham_essay.txt). Your query will be answered using the essay as context, using embeddings from text-ada-002 and LLM completions from gpt-3.5-turbo. You can read more about Llama Index and how this works in [our docs!](https://gpt-index.readthedocs.io/en/latest/index.html)"
)

index = None
api_key = st.text_input("Enter your OpenAI API key here:", value=os.environ.get("OPENAI_API_KEY", None), type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

if "OPENAI_API_KEY" in os.environ and "chat_engine" not in st.session_state:
    os.environ["OPENAI_API_KEY"] = api_key
    index = initialize_index(index_name, documents_folder)

    st.session_state.chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=index.as_retriever(similarity_top_k=2),
        llm=OpenAI(
            model="gpt-4o-mini", 
            additional_kwargs={"stream_options": {"include_usage": True}}
        ),
    )

if "chat_engine" not in st.session_state:
    st.warning("Please enter your api key first.")

st.markdown("## Chat")

text = st.text_input("User Input:")

if 'chat_engine' in st.session_state and st.button("Send Chat") and text is not None:
    for message in st.session_state.chat_engine.chat_history:
        with st.chat_message(name=message.role.value):
            st.write(message.content)

    with st.chat_message(name="user"):
        st.write(text)

    response = chat_with_data(text)
    
    with st.chat_message(name="assistant"):
        st.write_stream(response.response_gen)
    
    st.divider()
    
    st.markdown("## Sources")
    for node in response.source_nodes:
        st.write(str(node))

    st.divider()

    st.markdown(
        f"LLM Tokens Used: {st.session_state.token_counter.prev_total_token_count}"
    )
