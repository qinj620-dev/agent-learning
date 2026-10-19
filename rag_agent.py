"""任务7：RAG —— 给 Agent 接本地知识库

流程：读 knowledge_base.md → 按 ## 切块 → 向量化 → 提问时检索最相关的 2 块 → 带着原文去问 DeepSeek
"""
import os
import re
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ---------- 1. 加载知识库并切块 ----------
text = open("knowledge_base.md", encoding="utf-8").read()
chunks = [c.strip() for c in re.split(r"\n(?=## )", text) if c.strip().startswith("##")]
print(f"知识库切成 {len(chunks)} 块")

# ---------- 2. 向量化（模型任务 6 已下载，这次直接加载，很快） ----------
model = SentenceTransformer("shibing624/text2vec-base-chinese")
chunk_vecs = model.encode(chunks, normalize_embeddings=True)
print("向量建好了，可以提问了（直接回车退出）\n")

# ---------- 3. DeepSeek 客户端 ----------
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")


def retrieve(query, top_k=2):
    q = model.encode([query], normalize_embeddings=True)[0]
    scores = chunk_vecs @ q  # 归一化后点积 = 余弦相似度
    top = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top]


while True:
    query = input("你问：").strip()
    if not query:
        break
    hits = retrieve(query)
    print("\n检索命中的资料块：")
    for chunk, s in hits:
        title = chunk.split("\n")[0]
        print(f"  {s:.3f}  {title}")

    context = "\n\n---\n\n".join(c for c, _ in hits)
    resp = client.chat.completions.create(
        model="deepseek-chat",
        temperature=0,
        messages=[
            {"role": "system", "content": "你是一个答疑助手。只根据【资料】回答用户问题，资料里没有相关内容就直说不知道，不要编造。"},
            {"role": "user", "content": f"【资料】\n{context}\n\n用户问题：{query}"},
        ],
    )
    print(f"\n回答：{resp.choices[0].message.content}\n")
