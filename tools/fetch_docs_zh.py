"""抓取 Python 官方中文教程，转成纯文本存入 docs_zh/。

用途：为 doc-rag 建立中文版知识库，与英文版 docs/ 做对照实验。
章节与英文版一一对应，文件名保持一致，便于同一套评测脚本复用。

用法：
    python tools/fetch_docs_zh.py            # 全部抓取
    python tools/fetch_docs_zh.py classes    # 只抓某一章
"""
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://docs.python.org/zh-cn/3/tutorial/"
# 与英文库保持同样的子目录层级（docs/tutorial/），使 source 标签一致（tutorial/xxx.txt）
OUT_DIR = Path(__file__).resolve().parent.parent / "docs_zh" / "tutorial"

# 与 docs/tutorial/*.txt 同名同序，保证两套库可对照
CHAPTERS = [
    ("index", "index"),
    ("appetite", "appetite"),
    ("interpreter", "interpreter"),
    ("introduction", "introduction"),
    ("controlflow", "controlflow"),
    ("datastructures", "datastructures"),
    ("modules", "modules"),
    ("inputoutput", "inputoutput"),
    ("errors", "errors"),
    ("classes", "classes"),
    ("stdlib", "stdlib"),
    ("stdlib2", "stdlib2"),
    ("venv", "venv"),
    ("whatnow", "whatnow"),
    ("interactive", "interactive"),
    ("floatingpoint", "floatingpoint"),
    ("appendix", "appendix"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) doc-rag-fetch/1.0"
}

# 正文里需要整块丢弃的噪声节点
DROP_SELECTORS = [
    "div.related", "div.sphinxsidebar", "div.clearer", "div.footer",
    "div#searchbox", "script", "style", "a.headerlink",
    "div.admonition.seealso > p:last-child",  # 保留正文，仅去掉尾部跳转句
]


def _clean(node):
    for sel in DROP_SELECTORS:
        for tag in node.select(sel):
            tag.decompose()


def _block_text(el) -> str:
    """把单个块级元素转成文本；代码块保留原始缩进与换行。"""
    if el.name == "pre":
        return el.get_text().strip("\n")
    if el.name == "li":
        # 只取本层文字，嵌套列表由各自的 li 负责，避免外层把内层内容重复吞进来
        chunks = []
        for child in el.children:
            if getattr(child, "name", None) in ("ul", "ol"):
                continue
            text = child.get_text(" ", strip=True) if hasattr(child, "get_text") else str(child)
            if text.strip():
                chunks.append(text.strip())
        return " ".join(chunks)
    return el.get_text(" ", strip=True)


def html_to_text(main) -> str:
    """按块遍历，保留"标题 / 段落 / 列表 / 代码"的基本结构。"""
    parts = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "pre", "li"]):
        # pre 内部的元素由 pre 自己整体处理，跳过避免重复
        if el.name != "pre" and el.find_parent("pre"):
            continue
        # li 内部的 p 已由外层 li 统一取走，跳过避免同一句话出现两遍
        if el.name == "p" and el.find_parent("li"):
            continue
        text = _block_text(el)
        if not text:
            continue
        if el.name == "pre":
            parts.append("```\n" + text + "\n```")
        elif el.name in ("h1", "h2", "h3", "h4"):
            parts.append("\n" + text + "\n")
        elif el.name == "li":
            parts.append("  - " + text)
        else:
            parts.append(text)
    return "\n\n".join(parts)


def fetch_one(slug: str) -> str:
    url = BASE_URL + slug + ".html"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")
    main = soup.select_one("div.body") or soup.select_one("article") or soup.body
    if main is None:
        raise RuntimeError(f"未找到正文容器：{url}")
    _clean(main)
    text = html_to_text(main)
    # 折叠连续空行
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, []
    for name, slug in CHAPTERS:
        if only and only != name:
            continue
        try:
            text = fetch_one(slug)
            if len(text) < 200:
                raise RuntimeError(f"内容过短（{len(text)} 字符），可能抓错了容器")
            path = OUT_DIR / f"{name}.txt"
            path.write_text(text, encoding="utf-8")
            print(f"[OK]   {name:16s} {len(text):>7d} 字符")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {name:16s} {type(e).__name__}: {e}")
            fail.append(name)
        time.sleep(0.4)  # 对官方站点客气一点
    print(f"\n完成：成功 {ok} 个，失败 {len(fail)} 个")
    if fail:
        print("失败章节：" + ", ".join(fail))
        sys.exit(1)


if __name__ == "__main__":
    main()
