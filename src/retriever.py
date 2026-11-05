from .config import COLLECTION, TOP_K, get_client
from .embed import encode_query, get_embedding_function
from .fusion import rrf_fuse
from .rerank import rerank

# 先粗召回更多候选，再由 rerank 精选，才能体现重排的价值
CANDIDATE_K = 20
_RERANK_KEYS = ("ids", "documents", "metadatas", "distances")


def retrieve(query, top_k=TOP_K, use_rerank=False):
    """向量检索（可选重排），返回 Chroma 风格的 query 结果。

    查询端使用带 bge 检索指令前缀的向量（见 encode_query）；文档入库不加前缀，
    因此已有向量库无需重建。

    重排默认关闭，但原因分两种语料，别记混：
    - 英文语料（DOCSET=en）：跨语言打分不可靠，重排反而拉低效果
      （hit@1 70%→50%，hit@5 100%→90%）。
    - 中文语料（DOCSET=zh，默认）：重排有效（hit@1 90%→100%，MRR 0.920→1.000），
      但单次检索从 22ms 涨到 5.8s，交互式问答不划算，因此仍不默认开启。
    需要实验时可传 use_rerank=True，用 `python eval.py --rerank` 复现对比。
    """
    ef = get_embedding_function()
    client = get_client()
    collection = client.get_or_create_collection(COLLECTION, embedding_function=ef)

    n = max(CANDIDATE_K, top_k) if use_rerank else top_k
    res = collection.query(query_embeddings=[encode_query(query)], n_results=n)
    if not use_rerank:
        return res

    docs = res["documents"][0]
    if not docs:
        return res
    try:
        order = rerank(query, docs, top_k)
    except Exception as e:  # 重排不可用时降级，保证检索仍能返回结果
        print(f"[retriever] 重排失败，退回原始顺序：{type(e).__name__}")
        order = list(range(min(top_k, len(docs))))

    out = {}
    for key in _RERANK_KEYS:
        if key in res and res[key]:
            out[key] = [[res[key][0][i] for i in order]]
    for key in ("embeddings", "uris", "data"):
        if key in res:
            out[key] = res[key]
    return out


def _raw_query(query, n_results):
    """单路向量检索，返回 Chroma 风格结果（供多查询融合复用）。"""
    client = get_client()
    collection = client.get_or_create_collection(
        COLLECTION, embedding_function=get_embedding_function()
    )
    return collection.query(query_embeddings=[encode_query(query)], n_results=n_results)


# 原查询用中文、变体是英文翻译；等权融合会稀释原查询的首位结果，
# 所以给原查询更高权重（实测 2 倍时 hit@1 才不被拉低）
_ORIG_WEIGHT = 2.0


def retrieve_multi(query, top_k=TOP_K, n_per_query=6):
    """查询扩展 + 多路检索 + RRF 融合。

    扩展失败或没配 Key 时自动退回单路检索，行为与 retrieve() 完全一致，
    因此可以安全地作为可选路径接入。
    """
    from .query_expand import expand_query

    queries = expand_query(query)
    if len(queries) <= 1:
        return retrieve(query, top_k)

    results = [_raw_query(q, n_per_query) for q in queries]
    weights = [_ORIG_WEIGHT] + [1.0] * (len(queries) - 1)
    fused = rrf_fuse(results, top_n=top_k, weights=weights)
    fused["queries_used"] = queries
    return fused
