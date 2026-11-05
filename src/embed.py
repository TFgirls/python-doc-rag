import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 模型已缓存在本地，禁用 HF 联网检查：否则每次初始化都可能向 HF 发一次请求，
# 代理不稳定时抛 httpx.ProxyError(502) 直接打断评测。要下新模型时用
# `set HF_HUB_OFFLINE=0` 覆盖本行即可。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from .config import EMBEDDING_MODEL

_model = None


def get_model():
    """返回全局唯一的 SentenceTransformer
模型单例，避免重复加载模型到内存。"""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


class BgeEmbeddingFunction(EmbeddingFunction[Documents]):
    """Chroma 1.x 要求的 Embedding 对象：必须带 name() 和 __call__()。"""

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def name() -> str:
        return "bge-small-zh-v1.5"

    def __call__(self, input: Documents) -> Embeddings:
        return get_model().encode(input).tolist()

    @staticmethod
    def build_from_config(config):
        return BgeEmbeddingFunction()

    def get_config(self):
        return {}


_embed_fn = None


def get_embedding_function():
    """返回 Chroma 集合可用的 Embedding
函数单例（是对象，不是普通函数）。"""
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = BgeEmbeddingFunction()
    return _embed_fn


_QUERY_PREFIX = "为这个句子生成表示以用于检索相关篇章："


def encode_query(text: str):
    """查询端加 bge 检索指令前缀后编码，返回向量列表。"""
    return get_model().encode(_QUERY_PREFIX + text).tolist()