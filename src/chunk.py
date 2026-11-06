import re

# 句末切分符：中文句号/叹号/问号/分号 + 英文 ! ? ; + 换行。
# 故意不按英文句点 "." 切，避免把小数/缩写（如 3.14、e.g.）误断成两句。
_SENT_RE = re.compile(r"(?<=[。！？!?；;\n])")


def split_text(text, size=600, overlap=100):
    """句子/段落感知切片：先按句末与换行断句，再把句子贪心装箱进 ~size 的窗口。

    相比纯字符滑窗，片段始终从句子边界开始，不会把一句话或一行代码拦腰切断；
    仍保留行首缩进与换行，代码块不被压扁。
    """
    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 逐行：保留行首缩进，仅压缩行内连续空格/制表符
    lines = []
    for ln in text.split("\n"):
        stripped = ln.strip(" \t")
        indent = ln[: len(ln) - len(ln.lstrip(" \t"))]
        lines.append(indent + re.sub(r"[ \t]+", " ", stripped))
    text = "\n".join(lines)
    # 过多连续空行压成单个空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not text.strip():
        return []

    # 断句（句末标点/换行后断开，标点留在前半句）
    # 这里用 rstrip 而不是 strip：只去掉行尾空白，保留行首缩进。
    # 若用 strip，Python 代码块会被彻底压平（def / if / return 全部顶格），
    # 语义和可读性都被破坏——和本函数"代码块不被压扁"的承诺直接矛盾。
    sentences = [s.rstrip() for s in _SENT_RE.split(text) if s.strip()]
    if not sentences:
        return []

    # 贪心装箱：凑满 size 才换行到下一个片段，重叠区承接上一片段尾部
    chunks, cur = [], ""
    for s in sentences:
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= size:
            cur += "\n" + s
        else:
            chunks.append(cur)
            cur = (cur[-overlap:] + "\n" + s) if overlap > 0 and len(cur) > overlap else s
    if cur.strip():
        chunks.append(cur)
    return [c for c in chunks if c.strip()]
