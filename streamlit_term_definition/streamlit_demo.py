from token import OP
import streamlit as st

from llama_index.core import (
    Document,
    SummaryIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from constants import DEFAULT_TERM_STR, DEFAULT_TERMS, REFINE_TEMPLATE, TEXT_QA_TEMPLATE


if "all_terms" not in st.session_state:
    st.session_state["all_terms"] = DEFAULT_TERMS


def extract_terms(documents, term_extract_str, llm_name, model_temperature, api_key):
    llm = OpenAI(model=llm_name, temperature=model_temperature, api_key=api_key)

    temp_index = SummaryIndex.from_documents(documents)
    terms_definitions = str(
        temp_index.as_query_engine(response_mode="tree_summarize", llm=llm).query(
            term_extract_str
        )
    )
    terms_definitions = [
        x
        for x in terms_definitions.split("\n")
        if x and "Term:" in x and "Definition:" in x
    ]
    # parse the text into a dict
    terms_to_definition = {
        x.split("Definition:")[0]
        .split("Term:")[-1]
        .strip(): x.split("Definition:")[-1]
        .strip()
        for x in terms_definitions
    }
    return terms_to_definition


def insert_terms(terms_to_definition):
    for term, definition in terms_to_definition.items():
        doc = Document(text=f"Term: {term}\nDefinition: {definition}")
        st.session_state["llama_index"].insert(doc)


@st.cache_resource
def initialize_index(llm_name, model_temperature, api_key):
    """Create the VectorStoreIndex object."""
    embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)

    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./initial_index"),
        embed_model=embed_model,
    )

    return index


st.title("🦙 Llama Index Term Extractor 🦙")
st.markdown(
    (
        "This demo allows you to upload your own documents (either a screenshot/image or the actual text) and extract terms and definitions, building a knowledge base!\n\n"
        "Powered by [Llama Index](https://gpt-index.readthedocs.io/en/latest/index.html) and OpenAI, you can augment the existing knowledge of an "
        "LLM using your own notes, documents, and images. Then, when you ask about a term or definition, it will use your data first! "
        "The app is currently pre-loaded with terms from the NYC Wikipedia page."
    )
)

setup_tab, terms_tab, upload_tab, query_tab = st.tabs(
    ["Setup", "All Terms", "Upload/Extract Terms", "Query Terms"]
)

with setup_tab:
    st.subheader("LLM Setup")
    api_key = st.text_input("Enter your OpenAI API key here", type="password")
    llm_name = st.selectbox(
        "Which LLM?", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]
    )
    model_temperature = st.slider(
        "LLM Temperature", min_value=0.0, max_value=1.0, step=0.1
    )
    term_extract_str = st.text_area(
        "The query to extract terms and definitions with.", value=DEFAULT_TERM_STR
    )


with terms_tab:
    st.subheader("Current Extracted Terms and Definitions")
    st.json(st.session_state["all_terms"])


with upload_tab:
    st.subheader("Extract and Query Definitions")
    if st.button("Initialize Index and Reset Terms", key="init_index_1"):
        st.session_state["llama_index"] = initialize_index(
            llm_name, model_temperature, api_key
        )
        st.session_state["all_terms"] = DEFAULT_TERMS

    if "llama_index" in st.session_state:
        st.markdown(
            "Upload some text to extract terms and definitions."
        )
        
        document_text = st.text_area("Enter raw text")
        if st.button("Extract Terms and Definitions") and document_text:
            st.session_state["terms"] = {}
            terms_docs = {}
            with st.spinner("Extracting..."):
                terms_docs.update(
                    extract_terms(
                        [Document(text=document_text)],
                        term_extract_str,
                        llm_name,
                        model_temperature,
                        api_key,
                    )
                )
                
            st.session_state["terms"].update(terms_docs)

    if "terms" in st.session_state and st.session_state["terms"]:
        st.markdown("Extracted terms")
        st.json(st.session_state["terms"])

        if st.button("Insert terms?"):
            with st.spinner("Inserting terms"):
                insert_terms(st.session_state["terms"])
            st.session_state["all_terms"].update(st.session_state["terms"])
            st.session_state["terms"] = {}
            st.markdown("Terms inserted!")
            st.rerun()

with query_tab:
    st.subheader("Query for Terms/Definitions!")
    st.markdown(
        (
            "The LLM will attempt to answer your query, and augment it's answers using the terms/definitions you've inserted. "
            "If a term is not in the index, it will answer using it's internal knowledge."
        )
    )
    if st.button("Initialize Index and Reset Terms", key="init_index_2"):
        st.session_state["llama_index"] = initialize_index(
            llm_name, model_temperature, api_key
        )
        st.session_state["all_terms"] = DEFAULT_TERMS

    if "llama_index" in st.session_state:
        query_text = st.text_input("Ask about a term or definition:")
        if query_text:
            with st.spinner("Generating answer..."):
                llm = OpenAI(model=llm_name, temperature=model_temperature, api_key=api_key)

                response = (
                    st.session_state["llama_index"]
                    .as_query_engine(
                        llm=llm,
                        similarity_top_k=5,
                        response_mode="compact",
                        text_qa_template=TEXT_QA_TEMPLATE,
                        refine_template=REFINE_TEMPLATE,
                    )
                    .query(query_text)
                )
            st.markdown(str(response))
