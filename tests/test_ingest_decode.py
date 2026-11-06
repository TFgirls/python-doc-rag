"""上传文件解码的单元测试。

回归背景：早期直接 `data.decode("utf-8")`，用户传一个记事本存的 GBK txt
就抛 UnicodeDecodeError，且上传路径没有兜底，异常直接冒到 Streamlit 变成整页报错。
"""
import pytest

from src.ingest import _decode_bytes

TEXT = "类和对象是 Python 的核心概念"


@pytest.mark.parametrize(
    "encoding",
    ["utf-8", "utf-8-sig", "gbk", "gb18030"],
)
def test_decode_supported_encodings(encoding):
    raw = TEXT.encode(encoding)
    assert _decode_bytes(raw) == TEXT


def test_decode_strips_bom():
    """记事本另存的 UTF-8 带 BOM，解码后不能残留 \\ufeff。"""
    raw = b"\xef\xbb\xbf" + TEXT.encode("utf-8")
    out = _decode_bytes(raw)
    assert out == TEXT
    assert not out.startswith("\ufeff")


def test_decode_gbk_file_that_is_not_valid_utf8():
    """原始 bug 的最小复现：GBK 字节不是合法 UTF-8，旧实现会直接抛错。"""
    raw = TEXT.encode("gbk")
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")  # 旧实现 `data.decode("utf-8")` 的失败方式
    assert _decode_bytes(raw) == TEXT


def test_undecodable_binary_raises_value_error():
    with pytest.raises(ValueError) as exc:
        _decode_bytes(b"\xff\xfe\x00\x01\x02\x03")
    assert "编码" in str(exc.value)


def test_empty_bytes_returns_empty_string():
    assert _decode_bytes(b"") == ""
