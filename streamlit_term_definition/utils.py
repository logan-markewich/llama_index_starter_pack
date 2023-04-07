import os
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI


def get_llm(llm_name, model_temperature, api_key, max_tokens=256):
    os.environ["OPENAI_API_KEY"] = api_key
    if llm_name == "text-davinci-003":
        return OpenAI(
            temperature=model_temperature, model_name=llm_name, max_tokens=max_tokens
        )
    else:
        return ChatOpenAI(
            temperature=model_temperature, model_name=llm_name, max_tokens=max_tokens
        )
