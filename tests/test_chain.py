# tests/test_chain.py
import sys, os
from utils import init_chain

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
retriever, chain = init_chain()


# テスト用データを追加
texts = ["これはテスト文書です。", "もう一つのテスト情報。"]
ids = retriever.vectorstore.add_texts(texts)

# 1) 検索＋チェーン実行テスト
query = "テスト情報"
context_docs = retriever.get_relevant_documents(query)
context = "\n\n".join(d.page_content for d in context_docs)
response = chain.run(input="テスト", context=context)

print("Router Response:", response)
