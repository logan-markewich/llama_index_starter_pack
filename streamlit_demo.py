import os

import streamlit as st
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader

# NOTE: for local testing only, do NOT deploy with your key hardcoded
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]


index_name = "./index.json"
documents_folder = "./documents"


@st.cache_resource
def initialize_index(index_name, documents_folder):
    if os.path.exists(index_name):
        index = GPTSimpleVectorIndex.load_from_disk(index_name)
    else:
        documents = SimpleDirectoryReader(documents_folder).load_data()
        index = GPTSimpleVectorIndex(documents)
        index.save_to_disk(index_name)

    return index


@st.cache_data(max_entries=200, persist=True)
def query_index(_index, query_text):
    response = _index.query(query_text)
    return str(response)


# This should be cached and only fully runs once
index = initialize_index(index_name, documents_folder)


st.title("Llama Index")

st.header("Welcome to the Llama Index streamlit")

st.text("Please enter a query about Paul Graham's essay?")

text = st.text_input("Query text:")

if st.button("Run Query") and text is not None:
    response = query_index(index, text)
    st.markdown(response)
