# test_init.py
from utilis.llm_utils import init_llm_emb

llm, emb = init_llm_emb()
print("LLM initialized:", type(llm))
print("Embedding initialized:", type(emb))


# main.py の先頭付近に追加することでLLMとEmbeddingの初期化が正常か確認できます。

