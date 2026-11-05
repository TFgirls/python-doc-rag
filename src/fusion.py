"""多路检索结果的融合：Reciprocal Rank Fusion（RRF）。

多查询检索会产生多份排序结果，不能简单按距离合并——不同查询的向量空间
不可比，距离值没有统一尺度。RRF 只看**排名**不看分值，所以天然可融合：

    score(doc) = Σ 1 / (k + rank_i(doc))

k 取 60 是论文里的常用值（k 越大，排名靠后的结果影响越平滑）。
"""

from typing import Dict, List

RRF_K = 60


def rrf_fuse(results: List[dict], top_n: int = 5, k: int = RRF_K, weights=None) -> dict:
    """融合多份 Chroma 风格查询结果，返回同结构的前 top_n 条。

    weights 为各路权重（默认等权）。原查询要加权的原因：英文语料下实测，
    原查询（中文）与变体（英文翻译）等权融合会把原查询的首位结果稀释掉
    （hit@1 70%→60%），所以给原查询更高权重。注：默认中文语料下查询扩展
    已自动停用（见 query_expand.py），本函数的加权逻辑暂无调用方。

    distances 取该片段在各路结果里的最小值，保留「最相似过一次」的含义，
    方便调用方继续做距离统计或阈值判断。
    """
    if not results:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    score: Dict[str, float] = {}
    payload: Dict[str, tuple] = {}
    best_dist: Dict[str, float] = {}

    for idx, res in enumerate(results):
        w = weights[idx] if weights and idx < len(weights) else 1.0
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0] if res.get("distances") else [None] * len(ids)

        for rank, (i, d, m, dist) in enumerate(zip(ids, docs, metas, dists)):
            score[i] = score.get(i, 0.0) + w / (k + rank + 1)
            if i not in payload:
                payload[i] = (d, m)
                best_dist[i] = dist if dist is not None else float("inf")
            elif dist is not None and dist < best_dist[i]:
                best_dist[i] = dist

    order = sorted(score, key=lambda x: score[x], reverse=True)[:top_n]

    return {
        "ids": [order],
        "documents": [[payload[i][0] for i in order]],
        "metadatas": [[payload[i][1] for i in order]],
        "distances": [
            [None if best_dist[i] == float("inf") else best_dist[i] for i in order]
        ],
    }
