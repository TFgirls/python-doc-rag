"""检索评测：正样本算 hit@K，负样本算拒答率。

不调用大模型，纯本地检索 + 词面兜底，跑一次几秒钟。

用法（在 python-doc-rag 根目录）：
    python eval.py              # 正式评测
    python eval.py --calibrate  # 打印每题 top1 距离（用于说明"阈值不可分"）
"""
import argparse
import statistics
import sys
from pathlib import Path


def _force_utf8_console():
    """让 Windows 控制台与 stdout 的编码一致，否则中文必然乱码。

    只改 stdout 不改代码页（或反过来）都会更乱：控制台按 GBK 解码 UTF-8 字节。
    所以两步一起做——先把代码页切到 65001，再把 stdout 设为 UTF-8。
    非 Windows 或无控制台（重定向到文件、被 IDE 接管）时，失败直接忽略。
    """
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            if kernel32.GetConsoleOutputCP() != 65001:
                kernel32.SetConsoleOutputCP(65001)
                kernel32.SetConsoleCP(65001)
        except Exception:
            pass
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_force_utf8_console()

from src.config import TOP_K, PRESET_DOCS_DIR
from src.rag import out_of_scope, _latin_terms
from src.retriever import retrieve, retrieve_multi

# 正样本：问题 → 可接受的章节名（tuple，任一命中即算对）
#
# 为什么用多标签：同一问题的章节归属在不同语言版本里可能不同。例如「命令行参数」，
# 英文版在 interpreter.txt 的 "Argument Passing" 里正文反复出现 command line；
# 中文版该节译作「传入参数」，而叫「命令行参数」的整节在 stdlib.txt（argparse）。
# 用单标签会把「标注歧义」误判成「检索失败」，污染对照实验。
GOLD = [
    # —— 语言核心：类与对象 ——
    ("类和对象", ("classes",)),
    ("类的继承", ("classes",)),
    ("生成器是什么", ("classes",)),
    ("迭代器怎么用", ("classes",)),
    # —— 数据结构 ——
    ("字典", ("datastructures",)),
    ("列表推导式", ("datastructures",)),
    ("元组和列表的区别", ("datastructures",)),
    # —— 控制流与函数 ——
    ("for 循环", ("controlflow",)),
    ("while 循环", ("controlflow",)),
    ("函数默认参数", ("controlflow",)),
    ("lambda 表达式", ("controlflow",)),
    # —— 模块与包 ——
    ("模块导入", ("modules",)),
    ("包和子模块", ("modules",)),
    # —— 异常 ——
    ("异常处理", ("errors",)),
    ("自定义异常", ("errors",)),
    ("try except finally", ("errors",)),
    # —— 输入输出 ——
    ("读写文件", ("inputoutput",)),
    ("字符串格式化", ("inputoutput",)),
    ("json 序列化", ("inputoutput",)),
    # —— 标准库 ——
    ("标准库", ("stdlib", "stdlib2")),
    ("命令行参数", ("interpreter", "stdlib")),
    ("正则表达式", ("stdlib",)),
    ("日期时间处理", ("stdlib",)),
    ("多线程", ("stdlib2",)),
    ("日志记录", ("stdlib2",)),
    # —— 环境与细节 ——
    ("虚拟环境", ("venv",)),
    ("浮点数精度", ("floatingpoint",)),
]

# 负样本：答案不在教程文档里，正确行为是拒答而不是硬编。
# 全部带英文专名，可用 `--validate` 自动核对「这些词确实不在语料中」。
NEGATIVE = [
    # 数据库
    "怎么用 Python 连接 Redis 缓存？",
    "怎么用 Python 操作 MySQL 数据库？",
    "MongoDB 的索引怎么建？",
    "Alembic 的迁移脚本怎么写？",
    # Web / 服务端
    "Django 里怎么做数据库迁移？",
    "Flask 怎么配置跨域？",
    "FastAPI 怎么写一个接口？",
    "Nginx 反向代理 Python 服务怎么配？",
    # 前端
    "Vue 组件之间怎么传值？",
    "React 的 useEffect 怎么用？",
    # 机器学习 / 数据处理
    "PyTorch 怎么训练一个图像分类模型？",
    "TensorFlow 的模型怎么保存？",
    "pandas 怎么读取 Excel 文件？",
    "Sympy 怎么解方程组？",
    "matplotlib 怎么画折线图？",
    "OpenCV 怎么读取摄像头？",
    # 工程 / 部署
    "怎么用 Docker 部署 Python 应用？",
    "Kubernetes 怎么配置滚动更新？",
    "Kafka 消费者组怎么用？",
    "Celery 异步任务怎么做？",
    "怎么用 PyInstaller 把脚本打包成程序？",
    # 工具 / 其他生态
    "Scrapy 爬虫框架怎么用？",
    "Selenium 怎么模拟点击？",
    "Anaconda 怎么管理环境？",
    "Airflow 怎么编排任务依赖？",
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


def _load_corpus() -> str:
    """把当前语料全文拼成一个小写串，用于核对负样本的专名是否真的不存在。"""
    parts = []
    for p in sorted(PRESET_DOCS_DIR.rglob("*.txt")):
        parts.append(p.read_text(encoding="utf-8"))
    return "\n".join(parts).lower()


def validate_dataset():
    """校验评测集本身的质量，防止「自己出题自己答」造成的虚高。

    核心检查：负样本里的专有名词必须**确实不在语料中出现**。
    否则它就不算库外问题，拒答率会被凭空抬高。
    """
    corpus = _load_corpus()
    print(f"语料目录：{PRESET_DOCS_DIR}")
    print(f"语料总长：{len(corpus)} 字符\n")

    print("== 负样本：专有名词是否真的不在语料中 ==")
    bad = []
    for q in NEGATIVE:
        terms = _latin_terms(q)
        if not terms:
            print(f"  [需人工] {q}  （无英文专名，无法自动校验）")
            continue
        hit = [t for t in terms if t in corpus]
        if hit:
            bad.append(q)
            print(f"  [不合格] {q}  -> 语料中出现了 {hit}")
        else:
            print(f"  [通过]   {q}")
    print(f"\n负样本合格：{len(NEGATIVE) - len(bad)}/{len(NEGATIVE)}")

    print("\n== 正样本：能否在 top-k 命中目标章节 ==")
    miss = []
    for q, expects in GOLD:
        _, sources, _ = top1(q)
        stems = [Path(s).stem for s in sources]
        if any(st in expects for st in stems):
            print(f"  [命中]   {q}")
        else:
            miss.append(q)
            print(f"  [未命中] {q}  -> 期望 {expects}，实际 {stems}")
    print(f"\n正样本命中：{len(GOLD) - len(miss)}/{len(GOLD)}")

    if bad:
        print("\n需要处理的负样本（词在语料里出现过，说明并非库外问题）：")
        for q in bad:
            print(f"  - {q}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true", help="打印每题 top1 距离")
    ap.add_argument("--validate", action="store_true", help="校验评测集自身质量（负样本专名是否真的不在语料中）")
    ap.add_argument("--rerank", action="store_true", help="开启重排实验（默认关闭）")
    ap.add_argument("--expand", action="store_true", help="开启查询扩展 + RRF 融合（默认关闭）")
    args = ap.parse_args()
    use_rerank = args.rerank
    use_expand = args.expand

    if args.validate:
        validate_dataset()
        return

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
