"""查询扩展：把中文问题改写成多个检索查询，缓解「中文问题 × 英文文档」的跨语言错位。

## 适用范围（重要）

本模块**只为英文语料设计**，即 `DOCSET=en` 时才有意义。默认语料已切换为中文版
（`DOCSET=zh`），此时查询与文档同语言，翻译分支失去意义，实测反而负收益
（hit@1 90%→80%，同义扩展稀释了原始意图），因此 `expand_query()` 在中文语料下
直接返回原查询、不调用模型。

## 为什么需要这一步（英文语料场景）

英文语料时 hit@5 已经 100%，但 hit@1 只有 70%——召回没问题，**排序有问题**。
根因是语料为英文 Python 官方文档，而用户用中文提问：bge 是中文向量模型，
中文 query 与英文 chunk 在向量空间里对不齐，导致正确片段经常排到第 2、第 3。

之前试过 reranker 补救但失败了（hit@1 70% → 50%），因为 bge-reranker-base
同样是单语模型，跨语言打分不可靠。所以换思路：**在检索前就把查询翻译过去**。

## 实现

用 LangChain 的 LCEL 组合 prompt 与模型：

    prompt | llm | JsonOutputParser

一次调用生成 3 个变体（英文翻译 / 关键词精简 / 同义扩展），再分别检索后融合。
失败一律降级为「只用原查询」，不影响主流程。
"""

import logging

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, DOCSET

logger = logging.getLogger(__name__)

# 注意：此 prompt 假定"问题为中文、文档为英文"，仅在 DOCSET=en 时被调用
_SYSTEM = """你是检索查询改写器。用户的问题是中文，而知识库文档是英文的 Python 官方文档。
直接拿中文去检索英文文档会失真，所以需要你生成更利于检索的查询变体。

请生成 3 个变体：
1. 英文翻译版：把问题译成自然的英文检索语句（最重要）
2. 关键词版：去掉「怎么」「如何」「是什么」这类疑问词，只留核心术语
3. 同义扩展版：用该主题在 Python 文档里更常见的术语重新表述

要求：
- 不要回答问题，只给检索用的查询
- 每个变体不超过 20 个词
- 只输出 JSON，格式：{{"queries": ["变体1", "变体2", "变体3"]}}"""

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM), ("human", "{question}")]
)

# 最多生成几个变体（不含原查询）
MAX_VARIANTS = 3

_chain = None


def _get_chain():
    """LCEL 链：prompt | llm | JSON 解析。复用同一个实例，避免重复构造。"""
    global _chain
    if _chain is None:
        llm = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            base_url=DEEPSEEK_BASE_URL,
            api_key=DEEPSEEK_API_KEY,
            temperature=0,
            timeout=20,
        )
        _chain = _PROMPT | llm | JsonOutputParser()
    return _chain


def expand_query(query: str):
    """返回 [原查询, 变体1, 变体2, ...]。任何异常都降级为只用原查询。

    中文语料（DOCSET != "en"）下直接返回原查询、不调模型：查询与文档同语言时
    翻译分支失去意义，实测查询扩展为负收益（hit@1 90%→80%），详见模块 docstring。
    """
    if DOCSET != "en":
        return [query]
    if not DEEPSEEK_API_KEY:
        return [query]

    try:
        data = _get_chain().invoke({"question": query})
        variants = [str(q).strip() for q in (data.get("queries") or []) if str(q).strip()]
    except Exception as e:  # 网络/配额/JSON 解析失败都不该影响检索
        logger.warning("查询扩展失败，退回原查询：%s: %s", type(e).__name__, e)
        return [query]

    # 去重（含原查询），并限制数量
    seen = {query}
    out = [query]
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out[: MAX_VARIANTS + 1]
