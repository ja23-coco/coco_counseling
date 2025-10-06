# core/llm_utils.py
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import config 
from apisecret import get_secret

def init_llm_emb():
    llm = ChatOpenAI(
        model_name=config.FT_MODEL,
        api_key= get_secret("OPENAI_API_KEY"),
        temperature=0.4,
    )
    emb = OpenAIEmbeddings(
        model=config.EMBED_MODEL,
        api_key= get_secret("OPENAI_API_KEY"),
    )
    return llm, emb
