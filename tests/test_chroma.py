# tests/test_chroma.py
from utilis.retriever_utils import init_retriever

retriever = init_retriever()

# ダミーデータを追加して検索テスト
texts = ["こんにちは、世界！", "今日はいい天気ですね"]
ids = retriever.vectorstore.add_texts(texts)

results = retriever.get_relevant_documents("天気")
print("検索結果件数:", len(results))
for doc in results:
    print("-", doc.page_content)
