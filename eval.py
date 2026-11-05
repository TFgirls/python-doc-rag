"""检索评测：正样本算 hit@K，负样本算拒答率。

不调用大模型，纯本地检索 + 词面兜底，跑一次几秒钟。

用法（在 python-doc-rag 根目录）：
    python eval.py              # 正式评测
    python eval.py --calibrate  # 打印每题 top1 距离（用于说明"阈值不可分"）
"""
import argparse
import statistics
from pathlib import Path

from src.config import TOP_K
from src.rag import out_of_scope
from src.retriever import retrieve, retrieve_multi

# 正样本：问题 → 可接受的章节名（tuple，任一命中即算对）
#
# 为什么用多标签：同一问题的章节归属在不同语言版本里可能不同。例如「命令行参数」，
# 英文版在 interpreter.txt 的 "Argument Passing" 里正文反复出现 command line；
# 中文版该节译作「传入参数」，而叫「命令行参数」的整节在 stdlib.txt（argparse）。
# 用单标签会把「标注歧义」误判成「检索失败」，污染对照实验。
GOLD = [
    ("类和对象", ("classes",)),
    ("读写文件", ("inputoutput",)),
    ("异常处理", ("errors",)),
    ("字典", ("datastructures",)),
    ("模块导入", ("modules",)),
    ("for 循环", ("controlflow",)),
    ("虚拟环境", ("venv",)),
    ("命令行参数", ("interpreter", "stdlib")),
    ("标准库", ("stdlib", "stdlib2")),
    ("浮点数", ("floatingpoint",)),
]

# 负样本：答案不在教程文档里，正确行为是拒答而不是硬编
NEGATIVE = [
    "怎么用 Python 连接 Redis 缓存？",
    "Django 里怎么做数据库迁移？",
    "怎么用 Python 操作 MySQL 数据库？",
    "PyTorch 怎么训练一个图像分类模型？",
    "怎么用 Nginx 反向代理 Python 服务？",
    "pandas 怎么读取 Excel 文件？",
    "怎么用 Docker 部署 Python 应用？",
    "Flask 怎么配置跨域？",
    "怎么用 PyInstaller 把脚本打包成程序？",
    "matplotlib 怎么画折线图？",
]


def top1(query: str, use_rerank: bool = False, use_expand: bool = False):
    """返回 (top1 距离, top_k 命中来源, top_k 片段文本)。

    use_expand=True 走「查询扩展 + 多路检索 + RRF 融合」路径，用于 A/B 对比。
    """
    res = retrieve_multi(query) if use_expand else retrieve(query, use_rerank=use_rerank)
    dists = res.get("distances")
    dist = dists[0][0] if dists and dists[0] else None
    sources = [m["source"] for m in res["metadatas"][0]]
    return dist, sources, res["documents"][0]


def _stats(values):
    if not values:
        return "-"
    return f"min={min(values):.3f} 中位={statistics.median(values):.3f} max={max(values):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true", help="打印每题 top1 距离")
    ap.add_argument("--rerank", action="store_true", help="开启重排实验（默认关闭）")
    ap.add_argument("--expand", action="store_true", help="开启查询扩展 + RRF 融合（默认关闭）")
    args = ap.parse_args()
    use_rerank = args.rerank
    use_expand = args.expand

    pos = [(q, e, *top1(q, use_rerank=use_rerank, use_expand=use_expand)) for q, e in GOLD]
    # 拒答判定不依赖重排；开启扩展时正负样本走同一链路，保证对比公平
    neg = [(q, *top1(q, use_rerank=False, use_expand=use_expand)) for q in NEGATIVE]

    pos_d = [x[2] for x in pos if x[2] is not None]
    neg_d = [x[1] for x in neg if x[1] is not None]

    if args.calibrate:
        print("== 正样本 top1 距离 ==")
        for q, e, d, _, _ in pos:
            print(f"  {d:.3f}  {q}")
        print("== 负样本 top1 距离 ==")
        for q, d, _, _ in neg:
            print(f"  {d:.3f}  {q}")
        print()
        print(f"正样本 {_stats(pos_d)}")
        print(f"负样本 {_stats(neg_d)}")
        print("结论：两组区间重叠，距离阈值无法区分正负样本。")
        return

    hits = 0
    hit1 = 0
    mrr = 0.0
    mode = "查询扩展+RRF" if use_expand else ("单路+重排" if use_rerank else "单路")
    print(f"检索评测  top_k={TOP_K}  正样本={len(pos)}  负样本={len(neg)}  模式={mode}")
    print("-" * 62)
    for q, expects, d, sources, _ in pos:
        # 用文件名精确比对（tutorial/classes.txt -> classes）。先前用子串包含，
        # 会把 stdlib2 判成 stdlib 命中，指标虚高。
        stems = [Path(s).stem for s in sources]
        rank = next((i + 1 for i, st in enumerate(stems) if st in expects), None)
        hits += int(rank is not None)
        hit1 += int(rank == 1)
        mrr += (1.0 / rank) if rank else 0.0
        where = f"top{rank}" if rank else "未命中"
        print(f"[{where}] {q}  期望含「{'; '.join(expects)}」 距离={d:.3f}")

    refused = sum(1 for q, d, _, docs in neg if out_of_scope(q, docs))
    false_refuse = sum(1 for q, e, d, s, docs in pos if out_of_scope(q, docs))

    print("-" * 62)
    print(f"命中率 hit@{TOP_K}: {hits}/{len(pos)} = {hits / len(pos):.0%}")
    print(f"命中率 hit@1: {hit1}/{len(pos)} = {hit1 / len(pos):.0%}   MRR = {mrr / len(pos):.3f}")
    print(f"拒答率（负样本）: {refused}/{len(neg)} = {refused / len(neg):.0%}")
    print(f"误拒率（正样本）: {false_refuse}/{len(pos)} = {false_refuse / len(pos):.0%}")
    print(f"距离分布  正样本 {_stats(pos_d)}")
    print(f"          负样本 {_stats(neg_d)}  ← 与正样本重叠，故不用阈值")


if __name__ == "__main__":
    main()
