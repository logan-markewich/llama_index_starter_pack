import os
import sys
import logging
import pickle
from multiprocessing import Lock
from multiprocessing.managers import BaseManager
from typing import Any, Dict, List, Optional
import datetime

from langchain.chat_models import ChatOpenAI
#from langchain.embeddings import OpenAIEmbeddings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.chat_engine.types import BaseChatEngine
from llama_index import (Document, GPTVectorStoreIndex, LLMPredictor,
                         ServiceContext, SimpleDirectoryReader, StorageContext,
                         load_index_from_storage)
from llama_index.indices.base import BaseIndex
from llama_index.llms.base import ChatMessage
from llama_index.callbacks import CallbackManager, LlamaDebugHandler, CBEventType

# NOTE: for local testing only, do NOT deploy with your key hardcoded
# os.environ['OPENAI_API_KEY'] = "your key here"

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

indexes: Dict[str, BaseIndex] = {}
stored_docs: Dict[str, Any] = {}
lock = Lock()

def index_path(id: str, makedir: bool = True):
    index_name = "/root/stored_indexes/"
    path = os.path.join(index_name, id)
    if not os.path.exists(path) and makedir:
        os.makedirs(path)
    return path

def pkl_file(id: str):
    doc_name = "/root/stored_documents/"
    if not os.path.exists(doc_name):
        os.makedirs(doc_name)
    return os.path.join(doc_name, id + ".pkl")

def initialize_index(key: str):
    """Create a new global index, or load one from the pre-set path."""
    assert key is not None
    print(f"key = {key}")
    global indexes, stored_docs
    with lock:
        if key in indexes and key in stored_docs:
            return
        else:
            print("initializing the global index...")
            llama_debug = LlamaDebugHandler(print_trace_on_end=True)
            callback_manager = CallbackManager([llama_debug])
            service_context = ServiceContext.from_defaults(
                callback_manager=callback_manager,
                llm=ChatOpenAI(openai_api_key=key, model_name="gpt-3.5-turbo", temperature=0.0),
                embed_model = OpenAIEmbedding(api_key=key),
                chunk_size_limit=512,
            )
            
            index_name = index_path(key, makedir=False)
            if os.path.exists(index_name):
                storage_context = StorageContext.from_defaults(persist_dir=index_name)
                index = load_index_from_storage(
                    storage_context, service_context=service_context
                )
            else:
                index = GPTVectorStoreIndex([], service_context=service_context)
                index.storage_context.persist(persist_dir=index_name)
            
            indexes[key] = index

            pkl_name = pkl_file(key)
            if os.path.exists(pkl_name):
                with open(pkl_name, "rb") as f:
                    stored_docs[key] = pickle.load(f)
            
            assert key in indexes


async def query_index(key: str, query_text: str, history: Optional[List[ChatMessage]] = None):
    """Query the global index."""
    global indexes
    if key not in indexes:
        initialize_index(key)
    query_engine = indexes[key].as_query_engine(chat_history=history)
    response = await query_engine.aquery(query_text)
    return response


async def chat_index(key: str, chat_text, history: Optional[List[ChatMessage]] = None):
    """Chat the global index."""
    global indexes
    if key not in indexes:
        initialize_index(key)
    if history is None:
        chat_engine = indexes[key].as_chat_engine(chat_mode="context", verbose=True)
    else:
        chat_engine = indexes[key].as_chat_engine(
            chat_mode="context",
            chat_history=history,
            verbose=True,
        )

    response = await chat_engine.achat(chat_text)
    now = datetime.datetime.now()
    print(f"finish = {now}:{response}")

    return response


def insert_doc_index(key: str, doc_file_path, doc_id=None):
    """Insert new document into global index."""
    global indexes, stored_docs
    
    document = SimpleDirectoryReader(input_files=[doc_file_path]).load_data()[0]
    if doc_id is not None:
        document.doc_id = doc_id
        
    if key not in indexes:
        initialize_index(key)
            
    with lock:
        index = indexes[key]
        stored_doc = {}
        if key not in stored_docs:
            stored_docs[key] = stored_doc
        # Keep track of stored docs -- llama_index doesn't make this easy
        stored_doc[document.doc_id] = document.text[
            0:200
        ]  # only take the first 200 chars
        index_name = index_path(key)
        index.insert(document)
        index.storage_context.persist(persist_dir=index_name)
        print(f"index type = {type(index)} save index:{index_name}")
        pkl_name = pkl_file(key)
        with open(pkl_name, "wb") as f:
            pickle.dump(stored_doc, f)

        return dict(doc_hash=document.hash, doc_id=document.doc_id)

def insert_chunk_index(key: str, text_chunk: str, doc_id: str):
    """Insert new document into global index."""
    global indexes, stored_docs
    
    document = Document(text = text_chunk, doc_id=doc_id)  if doc_id else Document(text = text_chunk)
    
    if key not in indexes:
        initialize_index(key)
            
    with lock:
        index = indexes[key]
        stored_doc = {}
        if key not in stored_docs:
            stored_docs[key] = stored_doc
        
        index.insert(document)
        index_name = index_path(key)
        index.storage_context.persist(persist_dir=index_name)
        # Keep track of stored docs -- llama_index doesn't make this easy
        stored_doc[document.doc_id] = document.text[
            0:200
        ]  # only take the first 200 chars
        pkl_name = pkl_file(key)
        with open(pkl_name, "wb") as f:
            pickle.dump(stored_doc, f)

        return dict(doc_hash=document.hash, doc_id=document.doc_id)


def delete_from_index(key: str, doc_id):
    """Delete document from global index."""
    print(doc_id)
    global indexes, stored_docs
    if key not in indexes:
        initialize_index(key)
    with lock:
        indexes[key].delete_ref_doc(doc_id, delete_from_docstore=True)
        if key in stored_docs:
            sorted_doc = stored_docs[key]
            if doc_id in sorted_doc:
                del sorted_doc[doc_id]
                
            pkl_name = pkl_file(key)
            with open(pkl_name, "wb") as f:
                pickle.dump(stored_docs[key], f)

    return


def get_documents_list(key: str):
    """Get the list of currently stored documents."""
    global indexes, stored_doc
    if key not in indexes:
        initialize_index(key)

    with lock:
        documents_list = []
        if key not in stored_docs:
            return documents_list
        
        for doc_id, doc_text in stored_docs[key].items():
            documents_list.append({"id": doc_id, "text": doc_text})

        return documents_list

test_key = "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"
#test_key = "qwqw121e212222"
def test_insert_chunk_index():
    import hashlib
    md5 = hashlib.md5()
    chunk = "光环国际是做什么的？是一家培训项目管理与领导力的机构"
    md5.update(chunk.encode('utf-8'))
    doc_id = md5.hexdigest()
    print(doc_id)
    insert_chunk_index(test_key,chunk, doc_id)
    
    docs = get_documents_list(test_key)
    print(docs)

def test_query_index():
    text = "商之讯软件有限公司什么时候成立的"
    res = query_index(test_key, text)
    print(f"{text} = {res}")

def test_chat_index():
    text = "白菜怎么卖的？"
    history = []
    history.append(ChatMessage(role="user", content="明朝的第一位皇帝是谁"))
    history.append(ChatMessage(role="assistant", content="朱元璋"))
    history.append(ChatMessage(role="user", content="明朝的最后一个皇帝是谁"))
    history.append(ChatMessage(role="assistant", content="永历皇帝"))
    res = chat_index(test_key, text, history)
    print(f"{text} = {res}")
    
 
def test_get_documents_list():
    res = get_documents_list(test_key)
    print(f"docs = {res}")

def test_delete_from_index():
    doc_id = 'afe1bb62-4714-4e22-9061-7344c4519465'
    delete_from_index(test_key, doc_id)
    test_get_documents_list()

def test_insert_doc_index():
    insert_doc_index(test_key, "/root/documents/test.pdf", "songchao_doc_id")
    test_get_documents_list()
         
if __name__ == "__main__":
    env = os.environ
    if "OPENAI_API_KEY" in env:
        env.pop("OPENAI_API_KEY")
        
    #test_insert_chunk_index()
    #test_query_index()
    test_chat_index()
    #test_get_documents_list()
    #test_insert_doc_index()
    #test_delete_from_index()
    exit()
    pass
    # setup server
    # NOTE: you might want to handle the password in a less hardcoded way
    manager = BaseManager(("", 5602), b"password")
    manager.register("query_index", query_index)
    manager.register("chat_index", chat_index)
    manager.register("insert_doc_index", insert_doc_index)
    manager.register("insert_chunk_index", insert_chunk_index)
    manager.register("delete_from_index", delete_from_index)
    manager.register("get_documents_list", get_documents_list)
    #manager.register("test", test)
    server = manager.get_server()

    print("server started...")
    server.serve_forever()





# from langchain.schema import AgentAction, AgentFinish, OutputParserException


# class ConvoOutputParser(AgentOutputParser):
#     def get_format_instructions(self) -> str:
#         return FORMAT_INSTRUCTIONS

#     def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
#         try:
#             response = parse_json_markdown(text)
#             action, action_input = response["action"], response["action_input"]
#             if action == "Final Answer":
#                 return AgentFinish({"output": action_input}, text)
#             else:
#                 return AgentAction(action, action_input, text)
#         except Exception as e:
#             raise OutputParserException(f"Could not parse LLM output: {text}") from e

#     @property
#     def _type(self) -> str:
#         return "conversational_chat"
