import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 与 embed.py 一致：模型已缓存，禁用 HF 联网检查，避免代理抖动时抛 ProxyError。
# 要下新模型时用 `set HF_HUB_OFFLINE=0` 覆盖。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from sentence_transformers import CrossEncoder

RERANK_MODEL = "BAAI/bge-reranker-base"

_model = None


def get_reranker():
    """CrossEncoder 单例，避免重复加载模型（首次会下载约 1.1GB）。"""
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def rerank(query: str, docs, top_k: int):
    """对候选片段重排，返回保留下来的下标（按相关性从高到低）。

    rerank 的作用是"把对的挪到第一"，评测里看的是 hit@1 / MRR，而不是 hit@5。
    """
    if not docs or len(docs) <= top_k:
        return list(range(len(docs)))

    scores = get_reranker().predict([(query, d) for d in docs])
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    return order[:top_k]
