import os
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI


def get_sql_index_tool(sql_index, table_context_dict):
    table_context_str = "\n".join(table_context_dict.values())

    def run_sql_index_query(query_text):
        try:
            response = sql_index.as_query_engine(synthesize_response=False).query(query_text)
        except Exception as e:
            return f"Error running SQL {e}.\nNot able to retrieve answer."
        text = str(response)
        sql = response.extra_info["sql_query"]
        return f"Here are the details on the SQL table: {table_context_str}\nSQL Query Used: {sql}\nSQL Result: {text}\n"
        # return f"SQL Query Used: {sql}\nSQL Result: {text}\n"

    return run_sql_index_query


def get_llm(llm_name, model_temperature, api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    if llm_name == "text-davinci-003":
        return OpenAI(temperature=model_temperature, model_name=llm_name)
    else:
        return ChatOpenAI(temperature=model_temperature, model_name=llm_name)
