"""构建时预下载 embedding 模型，打进镜像里。

背景：Render 免费档休眠后会丢掉运行时下载的文件，首次提问又要重新下载
400MB 的模型，0.1 核上要几分钟。构建时下载的文件留在镜像里，以后唤醒
不用重新下载，首次提问只剩加载。

用法：Render 的 Build Command 改成
    pip install -r requirements.txt && python download_model.py
运行时不改代码：sentence-transformers 默认读同一个缓存目录，模型在里面
就直接加载，不再联网下载。
"""
from sentence_transformers import SentenceTransformer

MODEL_NAME = "shibing624/text2vec-base-chinese"

if __name__ == "__main__":
    print(f"预下载模型：{MODEL_NAME}")
    SentenceTransformer(MODEL_NAME)
    print("模型已缓存到镜像。")
