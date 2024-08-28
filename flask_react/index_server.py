import os
import pickle

# NOTE: for local testing only, do NOT deploy with your key hardcoded
os.environ['OPENAI_API_KEY'] = "your key here"

from multiprocessing import Lock
from multiprocessing.managers import BaseManager
from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex, 
    StorageContext, 
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

index = None
stored_docs = {}
lock = Lock()

index_name = "./saved_index"
pkl_name = "stored_documents.pkl"


def initialize_index():
    """Create a new global index, or load one from the pre-set path."""
    global index, stored_docs
    
    transformations = SentenceSplitter(chunk_size=512)
    embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
    with lock:
        if os.path.exists(index_name):
            index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=index_name), embed_model=embed_model
            )
        else:
            index = VectorStoreIndex(nodes=[], embed_model=embed_model)
            index.storage_context.persist(persist_dir=index_name)
        if os.path.exists(pkl_name):
            with open(pkl_name, "rb") as f:
                stored_docs = pickle.load(f)


def query_index(query_text):
    """Query the global index."""
    global index
    llm = OpenAI(model="gpt-4o-mini")
    response = index.as_query_engine(
        similarity_top_k=2,
        llm=llm,
    ).query(query_text)
    return response


def insert_into_index(doc_file_path, doc_id=None):
    """Insert new document into global index."""
    global index, stored_docs
    documents = SimpleDirectoryReader(input_files=[doc_file_path]).load_data()

    with lock:
        for document in documents:
            if doc_id is not None:
                document.id_ = doc_id
            index.insert(document)

            stored_docs[document.id_] = document.text[0:200]  # only take the first 200 chars

        index.storage_context.persist(persist_dir=index_name)

        first_document = documents[0]
        # Keep track of stored docs -- llama_index doesn't make this easy
        stored_docs[first_document.doc_id] = first_document.text[0:200] # only take the first 200 chars

        with open(pkl_name, "wb") as f:
            pickle.dump(stored_docs, f)

    return

def get_documents_list():
    """Get the list of currently stored documents."""
    global stored_doc
    documents_list = []
    for doc_id, doc_text in stored_docs.items():
        documents_list.append({"id": doc_id, "text": doc_text})

    return documents_list


if __name__ == "__main__":
    # init the global index
    print("initializing index...")
    initialize_index()

    # setup server
    # NOTE: you might want to handle the password in a less hardcoded way
    manager = BaseManager(('', 5602), b'password')
    manager.register('query_index', query_index)
    manager.register('insert_into_index', insert_into_index)
    manager.register('get_documents_list', get_documents_list)
    server = manager.get_server()

    print("server started...")
    server.serve_forever()
