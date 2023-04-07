import os
import streamlit as st

from PIL import Image
from llama_index import (
    Document,
    GPTSimpleVectorIndex,
    GPTListIndex,
    LLMPredictor,
    ServiceContext,
    SimpleDirectoryReader,
    PromptHelper,
)
from llama_index.readers.file.base import DEFAULT_FILE_EXTRACTOR, ImageParser

from constants import DEFAULT_TERM_STR, DEFAULT_TERMS, REFINE_TEMPLATE, TEXT_QA_TEMPLATE
from utils import get_llm


if "all_terms" not in st.session_state:
    st.session_state["all_terms"] = DEFAULT_TERMS


@st.cache_resource
def get_file_extractor():
    image_parser = ImageParser(keep_image=True, parse_text=True)
    file_extractor = DEFAULT_FILE_EXTRACTOR
    file_extractor.update(
        {
            ".jpg": image_parser,
            ".png": image_parser,
            ".jpeg": image_parser,
        }
    )

    return file_extractor


file_extractor = get_file_extractor()


def extract_terms(documents, term_extract_str, llm_name, model_temperature, api_key):
    llm = get_llm(llm_name, model_temperature, api_key, max_tokens=1024)

    service_context = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(llm=llm),
        prompt_helper=PromptHelper(
            max_input_size=4096, max_chunk_overlap=20, num_output=1024
        ),
        chunk_size_limit=1024,
    )

    temp_index = GPTListIndex.from_documents(documents, service_context=service_context)
    terms_definitions = str(
        temp_index.query(term_extract_str, response_mode="tree_summarize")
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
        doc = Document(f"Term: {term}\nDefinition: {definition}")
        st.session_state["llama_index"].insert(doc)


@st.cache_resource
def initialize_index(llm_name, model_temperature, api_key):
    """Create the GPTSQLStructStoreIndex object."""
    llm = get_llm(llm_name, model_temperature, api_key)

    service_context = ServiceContext.from_defaults(llm_predictor=LLMPredictor(llm=llm))

    index = GPTSimpleVectorIndex.load_from_disk(
        "./index.json", service_context=service_context
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
        "Which LLM?", ["text-davinci-003", "gpt-3.5-turbo", "gpt-4"]
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
            "Either upload an image/screenshot of a document, or enter the text manually."
        )
        uploaded_file = st.file_uploader(
            "Upload an image/screenshot of a document:", type=["png", "jpg", "jpeg"]
        )
        document_text = st.text_area("Or enter raw text")
        if st.button("Extract Terms and Definitions") and (
            uploaded_file or document_text
        ):
            st.session_state["terms"] = {}
            terms_docs = {}
            with st.spinner("Extracting (images may be slow)..."):
                if document_text:
                    terms_docs.update(
                        extract_terms(
                            [Document(document_text)],
                            term_extract_str,
                            llm_name,
                            model_temperature,
                            api_key,
                        )
                    )
                if uploaded_file:
                    Image.open(uploaded_file).convert("RGB").save("temp.png")
                    img_reader = SimpleDirectoryReader(
                        input_files=["temp.png"], file_extractor=file_extractor
                    )
                    img_docs = img_reader.load_data()
                    os.remove("temp.png")
                    terms_docs.update(
                        extract_terms(
                            img_docs,
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
            st.experimental_rerun()

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
                response = st.session_state["llama_index"].query(
                    query_text, similarity_top_k=5, response_mode="compact",
                    text_qa_template=TEXT_QA_TEMPLATE, refine_template=REFINE_TEMPLATE
                )
            st.markdown(str(response))
