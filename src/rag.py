import re

from .generator import generate
from .retriever import retrieve

REFUSAL = "文档中未找到相关信息。"
# 模型层拒答的识别标记（generator 的提示词要求它原样回这句）
_REFUSAL_MARK = "未找到相关信息"
# 拒答文案本身很短。加长度上限，避免正常长答案里恰好出现这几个字
# （比如用户问"你是怎么判断文档里没有的"，模型会复述这句话）被误判成拒答。
_REFUSAL_MAX_LEN = 40

# 英文/专有名词（Redis、Django、MySQL…）；这类词是"库里到底有没有"的最强信号
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_+#.\-]{1,}")
# 这些词在问题里出现不代表库外，不参与判定
_COMMON = {
    "python", "pip", "for", "while", "if", "else", "elif", "in", "and", "or", "not",
    "def", "class", "return", "import", "from", "try", "except", "with", "as",
    "idle", "vs", "tk", "print",
}


def _latin_terms(query: str):
    return {t.lower() for t in _LATIN.findall(query)} - _COMMON


def out_of_scope(query: str, docs):
    """词面兜底：问题里的专有名词在检索片段中一个都没出现，判定为库外问题。

    说明：曾尝试用 top1 距离阈值拒答，实测正负样本距离重叠
    （正样本 0.88–1.10，负样本 0.67–1.09），阈值法不可分，故改用词面兜底。
    该规则只覆盖"带专有名词"的库外问题；纯中文的语义级拒答仍依赖模型判断。
    """
    terms = _latin_terms(query)
    if not terms:
        return False
    corpus = "\n".join(docs).lower()
    return not any(t in corpus for t in terms)


def answer(query):
    """RAG 主流程：检索 -> 库外判定 -> 拼接上下文 -> 生成 -> 返回答案与来源。

    拒答有两条路径，对外表现统一为 (REFUSAL, [], docs)：
    1. 词面兜底 out_of_scope：问题含库内完全没有的专有名词，不调模型直接拒；
    2. 模型层语义拒答：模型判定片段无相关信息，此处归一化话术并清空来源，
       避免把无关片段当成"依据的文件"回给前端。
    """
    res = retrieve(query)
    docs = res["documents"][0]
    if not docs:
        return ("⚠️ 知识库还是空的，请先在左侧点击「建立 / 刷新预置知识库」再提问。", [], [])
    if out_of_scope(query, docs):
        return (REFUSAL, [], docs)
    metas = res["metadatas"][0]
    # 把来源文件名拼进每个片段，模型才"看得到"出处，避免编造引用（核心卖点：可追溯/防幻觉）
    context = "\n---\n".join(
        f"[来源:{m['source']}] {d}" for d, m in zip(docs, metas)
    )
    text = generate(query, context)
    # 只有"短且含拒答标记"才算拒答；长答案即使提到这句话也不当成拒答
    if len(text.strip()) <= _REFUSAL_MAX_LEN and _REFUSAL_MARK in text:
        return (REFUSAL, [], docs)
    sources = sorted({m["source"] for m in metas})
    return text, sources, docs
