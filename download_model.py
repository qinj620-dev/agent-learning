"""构建时预下载 embedding 模型，打进镜像里。

教训：模型不能只放 ~/.cache——构建和运行时用的缓存位置不一样，
构建时下好的模型运行时找不到（2026-11-05 首次提问报 OSError）。
所以改成下载到代码旁边的 models/ 文件夹里，这个文件夹会跟着进镜像，
运行时直接从本地加载，不用再联网。

用法：Render 的 Build Command：
    pip install -r requirements.txt && python download_model.py
"""
import os

# 构建时必须联网下载：HF_HUB_OFFLINE 是运行时用的离线变量，
# 但 Render 的环境变量在构建时也生效，先去掉，免得构建时下不到模型。
os.environ.pop("HF_HUB_OFFLINE", None)

from huggingface_hub import snapshot_download

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "bge-small-zh-v1.5"
)

if __name__ == "__main__":
    print(f"预下载模型：{MODEL_NAME} -> {LOCAL_DIR}")
    snapshot_download(MODEL_NAME, local_dir=LOCAL_DIR)
    print("模型已缓存到镜像。")
