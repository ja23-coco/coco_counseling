import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# src/utils.py にまとめたヘルパーをインポート
from utilis.memory_utils import init_memory

memory = init_memory()
print("Memory object:", type(memory))

memory.save_context({"input": "こんにちは"}, {"output": "こんにちは！どうされましたか？"})
summarized = memory.load_memory_variables({"input": "これまでの会話を要約して"})
print("Summarized history:", summarized.get("history", summarized))
