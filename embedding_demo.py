"""任务6：Embedding 初体验 —— 意思越像，向量离得越近。"""
from sentence_transformers import SentenceTransformer
import numpy as np

print("正在加载向量模型（第一次会下载几百 MB，泡杯茶等一下）...")
model = SentenceTransformer("shibing624/text2vec-base-chinese")

# 语料：故意埋了"陷阱"，用来证明这不是关键词匹配
corpus = [
    # ---- 天气 ----
    "今天北京天气怎么样",
    "京城今日气温如何",
    "明天会下雨吗",
    "后天适合出去玩吗",
    # ---- "苹果"陷阱：同一个词，意思完全不同 ----
    "我啃了一个苹果",
    "这个苹果真甜",
    "苹果公司发布了新手机",
    "苹果手机掉地上，屏幕摔碎了",
    # ---- 运动 ----
    "今晚去跑步",
    "周末去爬山",
    "打篮球能减肥吗",
    "每天游泳半小时",
    # ---- 美食 ----
    "附近有什么好吃的川菜",
    "这家火锅排队两小时",
    "水煮牛肉的做法",
    # ---- 技术 ----
    "Python 怎么读取 csv 文件",
    "git push 被拒绝怎么办",
    "什么是向量数据库",
]

print(f"正在给 {len(corpus)} 句话计算向量...")
vecs = model.encode(corpus, normalize_embeddings=True)
print(f"每句话变成了一个 {vecs.shape[1]} 维的向量")


def search(query, top_k=3):
    q = model.encode([query], normalize_embeddings=True)[0]
    # 向量都归一化过，点积 = 余弦相似度：越接近 1，意思越像
    scores = vecs @ q
    top = np.argsort(scores)[::-1][:top_k]
    print(f"\n查询：{query}")
    for i in top:
        print(f"  {scores[i]:.3f}  {corpus[i]}")


search("明天北京冷不冷")
search("iPhone 屏幕碎了怎么办")
search("想吃点辣的")
