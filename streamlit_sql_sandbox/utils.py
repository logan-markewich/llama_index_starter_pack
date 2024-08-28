import os
from typing import Callable
from llama_index.core.retrievers import NLSQLRetriever



def get_sql_retriever_fn(sql_retriever: NLSQLRetriever) -> Callable[[str], str]:

    def run_sql_retriever_query(query_text: str) -> str:
        try:
            nodes, metadata = sql_retriever.retrieve_with_metadata(query_text)
        except Exception as e:
            return f"Error running SQL {e}.\nNot able to retrieve answer."
        text = "\n\n".join([str(node.get_content()) for node in nodes])
        metadata = str(metadata)
        return f"SQL Result: {text}\n\nMetadata: {metadata}"

    return run_sql_retriever_query
