"""生成质量评测：只测**可自动验证**的硬指标。

## 为什么不做 faithfulness

用 DeepSeek 给 DeepSeek 的答案打分会系统性偏高（模型偏向自己的表述），
得出的"忠实度 0.9"没有说服力。可靠做法是人工抽检，那不在自动化范围内。
所以这里只算不需要判断力的客观指标，faithfulness **明确标注为未验证**。

## 测什么

1. **非拒答率**：正样本必须给出答案。被拒答说明生成环节掉了链子。
2. **引用存在率**：答案里应出现至少一个 `[来源:xxx]`。
3. **引用合法率**：答案里标注的文件，必须真的在本次检索结果中。
   这条能抓住"模型编造出处"——如果它引了一个没检索到的文件，就是幻觉。

用法：
    python eval_generation.py            # 全量
    python eval_generation.py --limit 5  # 只跑前 5 条（省钱、调试用）
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval import GOLD, _force_utf8_console  # noqa: E402

from src.rag import REFUSAL, answer  # noqa: E402

# 模型被要求在正文里标注 [来源:文件名]
_CITE = re.compile(r"\[来源[:：]\s*([^\]]+?)\s*\]")


def _basename(token: str) -> str:
    """把 'tutorial/classes.txt' 归一成 'classes'，便于与检索来源比对。"""
    name = token.strip().split("/")[-1]
    return name[:-4] if name.endswith(".txt") else name


def check_one(query: str):
    """返回单条评测结果。"""
    text, sources, _ = answer(query)
    legal = {_basename(s) for s in sources}
    cites = {_basename(c) for c in _CITE.findall(text)}
    refused = text.strip() == REFUSAL

    return {
        "query": query,
        "refused": refused,
        "has_cite": bool(cites),
        "cites": sorted(cites),
        "legal": sorted(legal),
        "bad_cites": sorted(cites - legal),  # 引用了检索结果里没有的文件
        "len": len(text),
        "text": text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条")
    args = ap.parse_args()

    _force_utf8_console()
    items = GOLD[: args.limit] if args.limit else GOLD

    print(f"生成质量评测：{len(items)} 条正样本（每条一次 LLM 调用）")
    print("-" * 62)

    rows = []
    for i, (q, _expects) in enumerate(items, 1):
        r = check_one(q)
        rows.append(r)
        flags = []
        if r["refused"]:
            flags.append("被拒答")
        if not r["has_cite"]:
            flags.append("无引用")
        if r["bad_cites"]:
            flags.append(f"编造出处:{r['bad_cites']}")
        mark = "OK " if not flags else "!! "
        print(f"{mark}[{i:>2}] {q}  ({r['len']}字) {' '.join(flags)}")

    total = len(rows)
    refused = [r for r in rows if r["refused"]]
    no_cite = [r for r in rows if not r["has_cite"] and not r["refused"]]
    bad_cite = [r for r in rows if r["bad_cites"]]
    answered = [r for r in rows if not r["refused"]]
    avg_len = sum(r["len"] for r in answered) / len(answered) if answered else 0

    print("-" * 62)
    print(f"非拒答率        : {total - len(refused)}/{total} = {(total - len(refused)) / total:.0%}")
    if answered:
        ok_cite = len(answered) - len(no_cite)
        ok_legal = len(answered) - len(bad_cite)
        print(f"引用存在率      : {ok_cite}/{len(answered)} = {ok_cite / len(answered):.0%}")
        print(f"引用合法率      : {ok_legal}/{len(answered)} = {ok_legal / len(answered):.0%}")
    else:
        print("引用存在率      : -")
        print("引用合法率      : -")
    print(f"平均答案长度    : {avg_len:.0f} 字")
    print()
    print("未评测：faithfulness（答案是否忠于原文）、答案相关性。")
    print("原因：LLM 自评有偏，需人工抽检，不在本脚本自动化范围内。")

    if refused:
        print("\n被拒答的正样本（应修复）：")
        for r in refused:
            print(f"  - {r['query']}")
    if bad_cite:
        print("\n出现编造出处的答案（引用文件不在检索结果中）：")
        for r in bad_cite:
            print(f"  - {r['query']} -> {r['bad_cites']}（实际来源 {r['legal']}）")


if __name__ == "__main__":
    main()
