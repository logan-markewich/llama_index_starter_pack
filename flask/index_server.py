import os

# NOTE: for local testing only, do NOT deploy with your key hardcoded
os.environ['OPENAI_API_KEY'] = "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"

from multiprocessing import Lock
from multiprocessing.managers import BaseManager
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, Document

index = None
lock = Lock()

index_name = "./index.json"
documents_folder = "./documents"


def initialize_index():
    """Create a new global index, or load one from the pre-set path."""
    global index

    with lock:
        if os.path.exists(index_name):
            index = GPTSimpleVectorIndex.load_from_disk(index_name)
        else:
            documents = SimpleDirectoryReader(documents_folder).load_data()
            index = GPTSimpleVectorIndex(documents)
            index.save_to_disk(index_name)


def query_index(query_text):
    """Query the global index."""
    global index
    response = index.query(query_text)
    return str(response).strip()


def insert_into_index(doc_text, doc_id=None):
    """Insert new document into global index."""
    global index
    print(doc_text)
    document = Document(doc_text)
    if doc_id is not None:
        document.doc_id = doc_id

    with lock:
        index.insert(document)


if __name__ == "__main__":
    # init the global index
    print("initializing index...")
    initialize_index()

    # setup server
    # NOTE: you might want to handle the password in a less hardcoded way
    manager = BaseManager(('', 5602), b'password')
    manager.register('query_index', query_index)
    manager.register('insert_into_index', insert_into_index)
    server = manager.get_server()

    print("starting server...")
    server.serve_forever()
