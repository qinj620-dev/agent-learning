"""构建时预下载 embedding 模型（已废弃，保留为空操作）。

2026-11-05：检索已改成纯 Python TF-IDF，不再需要 embedding 模型，
Render 免费版 512MB 内存也装不下 torch。本文件只留一个空操作，
让 Render 上现有的 Build Command
  pip install -r requirements.txt && python download_model.py
不用改也能继续跑。以后可以删掉本文件，并把 Build Command 简化回
  pip install -r requirements.txt
"""
if __name__ == "__main__":
    print("no model download needed (TF-IDF retriever, no torch)")
