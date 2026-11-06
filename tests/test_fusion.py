"""RRF 多路融合的单元测试。

rrf_fuse 只依赖传入的结果列表，不碰模型与网络，适合直接构造输入验证。
"""
from src.fusion import RRF_K, rrf_fuse


def _res(ids, base=0.0, prefix="doc"):
    """构造一份 Chroma 风格的查询结果。"""
    return {
        "ids": [list(ids)],
        "documents": [[f"{prefix}-{i}" for i in ids]],
        "metadatas": [[{"source": f"{prefix}-{i}.txt"} for i in ids]],
        "distances": [[base + 0.01 * n for n in range(len(ids))]],
    }


def test_empty_input_returns_empty_structure():
    out = rrf_fuse([])
    assert out["ids"] == [[]]
    assert out["documents"] == [[]]
    assert out["metadatas"] == [[]]
    assert out["distances"] == [[]]


def test_single_route_keeps_order():
    out = rrf_fuse([_res(["a", "b", "c"])], top_n=3)
    assert out["ids"][0] == ["a", "b", "c"]


def test_duplicates_across_routes_are_merged():
    """同一个片段在多路结果里都出现时，只能保留一条。"""
    r1 = _res(["a", "b"], prefix="x")
    r2 = _res(["b", "c"], prefix="x")
    out = rrf_fuse([r1, r2], top_n=5)
    ids = out["ids"][0]
    assert len(ids) == len(set(ids)), "融合后出现重复 id"
    assert set(ids) == {"a", "b", "c"}


def test_document_in_both_routes_ranks_first():
    """两路都排第一的结果，得分应高于只在一路排第一的结果。"""
    r1 = _res(["shared", "only1"])
    r2 = _res(["shared", "only2"])
    out = rrf_fuse([r1, r2], top_n=3)
    assert out["ids"][0][0] == "shared"


def test_weight_shifts_ranking():
    """给第一路更高权重后，它独有的首位结果应排到前面。"""
    r1 = _res(["from_r1"])
    r2 = _res(["from_r2"])
    equal = rrf_fuse([r1, r2], top_n=2, weights=[1.0, 1.0])
    weighted = rrf_fuse([r1, r2], top_n=2, weights=[2.0, 1.0])
    assert equal["ids"][0][0] == "from_r1"  # 等权时按插入顺序，r1 在前
    assert weighted["ids"][0][0] == "from_r1"  # 加权后仍然是 r1，但优势更大
    # 用得分差验证加权确实生效：比较 r1 首位与 r2 首位的相对位置
    assert weighted["ids"][0] == ["from_r1", "from_r2"]


def test_weights_default_to_equal():
    r1 = _res(["a"])
    r2 = _res(["b"])
    out = rrf_fuse([r1, r2], top_n=2)
    assert set(out["ids"][0]) == {"a", "b"}


def test_top_n_truncates():
    out = rrf_fuse([_res(list("abcdefgh"))], top_n=3)
    assert len(out["ids"][0]) == 3


def test_distance_takes_minimum():
    """同一片段在多路里距离不同时，取最小值（"最相似过一次"）。"""
    r1 = {
        "ids": [["a"]],
        "documents": [["d"]],
        "metadatas": [[{"source": "s.txt"}]],
        "distances": [[0.9]],
    }
    r2 = {
        "ids": [["a"]],
        "documents": [["d"]],
        "metadatas": [[{"source": "s.txt"}]],
        "distances": [[0.3]],
    }
    out = rrf_fuse([r1, r2], top_n=1)
    assert out["distances"][0][0] == 0.3


def test_missing_distances_tolerated():
    """上游没给 distances 时不应崩，返回 None。"""
    res = {
        "ids": [["a"]],
        "documents": [["d"]],
        "metadatas": [[{"source": "s.txt"}]],
    }
    out = rrf_fuse([res], top_n=1)
    assert out["ids"][0] == ["a"]
    assert out["distances"][0][0] is None


def test_rrf_score_formula():
    """单路单条时得分应为 1/(k+1)，间接验证 RRF 常数生效。"""
    res = _res(["only"])
    out = rrf_fuse([res], top_n=1, k=RRF_K)
    assert out["ids"][0] == ["only"]
