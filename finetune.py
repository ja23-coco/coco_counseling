""""""
# src/finetune.py
# JSONL → OpenAI FT API
"""
LangSmith で収集済みの会話ログ (JSONL) を
  1. `category` 列で教師データに整形
  2. OpenAI Fine-Tuning API へ登録する
"""

import json, pathlib, subprocess, secrets, config

DATA_DIR = config.GDRIVE_RAG_DIR / "fine_tune_data"
FILE_OUT  = DATA_DIR / "train.jsonl"

def generate_train_jsonl():
    # TODO: LangSmith API からログを pull 済みと仮定
    merged = []
    for fp in DATA_DIR.glob("*.jsonl"):
        with open(fp, encoding="utf-8") as f:
            merged.extend(json.loads(l) for l in f)
    with open(FILE_OUT, "w", encoding="utf-8") as f:
        for item in merged:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

def run_finetune():
    subprocess.run([
        "openai",
        "api",
        "fine_tunes.create",
        "-t", str(FILE_OUT),
        "-m", "gpt-4o-mini"
    ])

if __name__ == "__main__":
    generate_train_jsonl()
    run_finetune()
