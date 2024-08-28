from json import tool
from lib2to3.pgen2.token import NL
import streamlit as st
from sqlalchemy import create_engine

from llama_index.agent.openai import OpenAIAgent
from llama_index.core.agent import AgentRunner
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import NLSQLRetriever
from llama_index.core.tools import FunctionTool
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from constants import (
    DEFAULT_SQL_PATH,
    DEFAULT_BUSINESS_TABLE_DESCRP,
    DEFAULT_VIOLATIONS_TABLE_DESCRP,
    DEFAULT_INSPECTIONS_TABLE_DESCRP,
    DEFAULT_TOOL_DESCRP,
)
from utils import get_sql_retriever_fn


def initialize_retriever(
    llm_name: str, 
    model_temperature: float, 
    table_context_dict: dict, 
    api_key: str, 
    sql_path: str=DEFAULT_SQL_PATH
) -> NLSQLRetriever:
    """Create the NLSQLRetriever object."""
    llm = OpenAI(model=llm_name, temperature=model_temperature, api_key=api_key)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)

    engine = create_engine(sql_path)
    sql_database = SQLDatabase(engine)

    return NLSQLRetriever(
        sql_database,
        llm=llm,
        embed_model=embed_model,
        context_query_kwargs=table_context_dict,
    )


def initialize_agent(
    llm_name: str, 
    model_temperature: float, 
    tool_description: str, 
    api_key: str, 
    _sql_retriever: NLSQLRetriever
) -> AgentRunner:
    """Create a custom agent with a sql tool."""
    sql_tool = FunctionTool.from_defaults(
        fn=get_sql_retriever_fn(_sql_retriever),
        name="sql_query_tool",
        description=tool_description,
    )

    llm = OpenAI(model=llm_name, temperature=model_temperature, api_key=api_key)

    agent = OpenAIAgent.from_tools(
        [sql_tool],
        llm=llm,
        verbose=False,
    )

    return agent


st.title("🦙 Llama Index SQL Sandbox 🦙")
st.markdown(
    (
        "This sandbox uses a sqlite database by default, powered by [Llama Index](https://gpt-index.readthedocs.io/en/latest/index.html) and OpenAI.\n\n"
        "The database contains information on health violations and inspections at restaurants in San Francisco."
        "This data is spread across three tables - businesses, inspections, and violations.\n\n"
        "Using the setup page, you can adjust LLM settings, change the context for the SQL tables, and change the tool description for the tool."
        "The other tabs will perform chatbot and text2sql operations.\n\n"
        "Read more about LlamaIndex's structured data support [here!](https://gpt-index.readthedocs.io/en/latest/guides/tutorials/sql_guide.html)"
    )
)


setup_tab, llama_tab, agent_tab = st.tabs(
    ["Setup", "Single-Shot Query", "Agent + Chat History"]
)

with setup_tab:
    st.subheader("LLM Setup")
    api_key = st.text_input("Enter your OpenAI API key here", type="password")
    llm_name = st.selectbox(
        "Which LLM?", ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo", "gpt-4o-mini", "gpt-4"]
    )
    model_temperature = st.slider(
        "LLM Temperature", min_value=0.0, max_value=1.0, step=0.1
    )

    st.subheader("Table Setup")
    business_table_descrp = st.text_area(
        "Business table description", value=DEFAULT_BUSINESS_TABLE_DESCRP
    )
    violations_table_descrp = st.text_area(
        "Violation table description", value=DEFAULT_VIOLATIONS_TABLE_DESCRP
    )
    inspections_table_descrp = st.text_area(
        "Inspection table description", value=DEFAULT_INSPECTIONS_TABLE_DESCRP
    )

    table_context_dict = {
        "businesses": business_table_descrp,
        "inspections": inspections_table_descrp,
        "violations": violations_table_descrp,
    }

    use_table_descrp = st.checkbox("Use table descriptions?", value=True)
    tool_descrp = st.text_area("Tool Description", value=DEFAULT_TOOL_DESCRP)

with llama_tab:
    st.subheader("Text2SQL with Llama Index")
    if st.button("Initialize Index", key="init_index_1"):
        st.session_state["llama_index"] = initialize_retriever(
            llm_name,
            model_temperature,
            table_context_dict if use_table_descrp else None,
            api_key,
        )

    if "llama_index" in st.session_state:
        query_text = st.text_input(
            "Query:", value="Which restaurant has the most violations?"
        )
        use_nl = st.checkbox("Return natural language response?")
        if st.button("Run Query") and query_text:
            with st.spinner("Getting response..."):
                try:
                    retriever = st.session_state["llama_index"]
                    response_synthesizer = get_response_synthesizer(
                        llm=OpenAI(model=llm_name, temperature=model_temperature, api_key=api_key),
                    )
                    nodes, metadata = retriever.retrieve_with_metadata(query_text)
                    if use_nl:
                        response_text = str(response_synthesizer.synthesize(query_text, nodes))
                    else:
                        response_text = "\n\n".join([str(node.get_content()) for node in nodes])
                    response_sql = metadata["sql_query"]
                except Exception as e:
                    response_text = "Error running SQL Query."
                    response_sql = str(e)

            col1, col2 = st.columns(2)
            with col1:
                st.text("SQL Result:")
                st.markdown(response_text)

            with col2:
                st.text("SQL Query:")
                st.markdown(response_sql)

with agent_tab:
    st.subheader("Llama Index SQL Demo")

    if st.button("Initialize Agent"):
        st.session_state["llama_index"] = initialize_retriever(
            llm_name,
            model_temperature,
            table_context_dict if use_table_descrp else None,
            api_key,
        )
        st.session_state["agent"] = initialize_agent(
            llm_name,
            model_temperature,
            tool_descrp,
            api_key,
            st.session_state["llama_index"],
        )

    model_input = st.text_input(
        "Message:", value="Which restaurant has the most violations?"
    )
    if "agent" in st.session_state and st.button("Send"):
        for msg in st.session_state["agent"].chat_history:
            if msg.role.value in ("assistant", "user") and msg.content:
                with st.chat_message(msg.role.value):
                    st.write(msg.content)
        
        with st.chat_message("user"):
            st.write(model_input)

        with st.spinner("Getting response..."):
            response = st.session_state["agent"].stream_chat(model_input)
    
        with st.chat_message("assistant"):
            st.write_stream(response.response_gen)
