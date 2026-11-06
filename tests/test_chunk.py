"""切片逻辑的单元测试。

split_text 是纯函数（不碰模型、不碰网络、不碰磁盘），是整个链路里
最该被测试覆盖的一环：它一旦出错，后面所有环节都建立在错误的分片上。
"""
import pytest

from src.chunk import split_text


def test_empty_input_returns_empty_list():
    assert split_text("") == []
    assert split_text("   ") == []
    assert split_text("\n\n\n") == []


def test_short_text_stays_in_one_chunk():
    text = "Python 是一门解释型语言。它支持面向对象编程。"
    chunks = split_text(text)
    assert len(chunks) == 1
    assert "解释型语言" in chunks[0]


def test_long_text_splits_into_multiple_chunks():
    text = "".join(f"这是第 {i} 句话，用来把文本撑到需要切分的长度。" for i in range(60))
    chunks = split_text(text, size=600, overlap=100)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_chunk_size_not_exceeded():
    """装箱时每加一句都检查过窗口上限，所以片段不会超过 size（除非单句本身就超）。"""
    sentence = "这是一句长度适中的中文句子用于测试窗口边界。"  # 23 字
    text = sentence * 100
    chunks = split_text(text, size=600, overlap=100)
    assert max(len(c) for c in chunks) <= 600


def test_consecutive_chunks_overlap():
    """窗口之间要有重叠，避免答案正好落在接缝处时被切断。"""
    text = "".join(f"第{i}句话讲的是某个具体的知识点内容。" for i in range(80))
    chunks = split_text(text, size=600, overlap=100)
    assert len(chunks) >= 3
    # 前一片的尾部应当出现在后一片的开头（重叠区承接）
    assert chunks[0][-50:] in chunks[1]


def test_code_indent_is_preserved():
    """回归测试：曾用 strip() 断句，导致 Python 代码块被压平。

    def / if / return 全部顶格后，代码语义就变了，必须确保缩进保留。
    """
    code = "def f(x):\n    if x > 0:\n        return x\n    return -x"
    joined = "\n".join(split_text(code))
    assert "    if x > 0" in joined, "四空格缩进丢失"
    assert "        return x" in joined, "八空格缩进丢失"


def test_no_content_lost():
    """所有句子都应出现在切片结果里，不能因为装箱逻辑丢内容。"""
    sentences = [f"第{i}个句子。" for i in range(50)]
    text = "".join(sentences)
    joined = "".join(split_text(text, size=200, overlap=50))
    for s in sentences:
        assert s in joined


@pytest.mark.parametrize("size,overlap", [(200, 0), (300, 50), (600, 100)])
def test_various_window_params(size, overlap):
    """不同窗口参数下都应产出非空且不超限的片段。"""
    text = "。".join(f"句子{i}的内容描述" for i in range(100)) + "。"
    chunks = split_text(text, size=size, overlap=overlap)
    assert chunks
    assert all(len(c) <= max(size, len(text)) for c in chunks)
