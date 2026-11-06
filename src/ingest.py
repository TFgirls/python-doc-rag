import io
import uuid
from pathlib import Path

from .config import PRESET_DOCS_DIR, COLLECTION, get_client
from .chunk import split_text
from .embed import get_embedding_function


# 上传文件的常见编码，按优先级尝试：
#   utf-8-sig 先切掉 Windows 记事本加的 BOM；
#   gb18030 是 GBK/GB2312 的超集，能覆盖国内用户直接用记事本存的 txt。
_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")


def _decode_bytes(data: bytes) -> str:
    """按候选编码依次尝试解码，全失败时抛带可读信息的 ValueError。

    不直接 .decode("utf-8")：用户传一个 GBK 存的 txt 会抛 UnicodeDecodeError，
    而上传路径没有兜底，异常会直接冒到 Streamlit 变成整页报错。
    """
    for enc in _ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(
        f"无法识别文件编码（已尝试 {'/'.join(_ENCODINGS)}）。"
        "请将文件另存为 UTF-8 后重试。"
    )


def _read_text(source):
    """读取一个 .txt 源，返回全文（str）。

    source 可为路径/Path（预置），或 BytesIO（上传）。
    上传的字节流按 ``_decode_bytes`` 的多编码策略解码。
    """
    if hasattr(source, "read"):  # BytesIO（上传）
        data = source.read()
        if isinstance(data, bytes):
            data = _decode_bytes(data)
        return data
    with open(source, encoding="utf-8") as f:
        return f.read()


def _unique_id(path: Path) -> str:
    """用相对路径生成唯一 base_id，避免不同子目录下同名文件互相覆盖。"""
    return str(path.relative_to(PRESET_DOCS_DIR)).replace("\\", "_").replace("/", "_")


def _source_label(path: Path) -> str:
    """用相对路径作为来源标签，避免不同子目录同名文件的引用歧义。"""
    return str(path.relative_to(PRESET_DOCS_DIR)).replace("\\", "/")


def _add_doc(collection, text, source, base_id):
    """把一份文档的文本切片后 upsert 进集合。

    先按 base_id 删除该文档的旧片段，避免内容变短后残留孤儿 chunk，
    再写入新片段（幂等：相同 id 覆盖）。
    """
    docs, ids, metas = [], [], []
    for i, chunk in enumerate(split_text(text)):
        docs.append(chunk)
        ids.append(f"{base_id}-{i}")
        metas.append({"source": source, "base_id": base_id})
    if docs:
        collection.delete(where={"base_id": base_id})
        collection.upsert(documents=docs, ids=ids, metadatas=metas)
    return len(docs)


def ingest_preset():
    """扫描预置目录（含子目录）下所有 .txt，建立/刷新知识库。返回片段总数。

    单文件失败（损坏/编码问题）会被跳过并继续；若文件之前入库过、本次却读取为空，
    会先删掉它的旧片段再跳过，避免孤儿数据残留。
    """
    ef = get_embedding_function()
    client = get_client()
    collection = client.get_or_create_collection(COLLECTION, embedding_function=ef)
    total = 0
    failures = []
    for src in sorted(PRESET_DOCS_DIR.rglob("*.txt")):
        try:
            text = _read_text(src)
            if not text.strip():
                failures.append((src.name, "读取到的文字为空"))
                collection.delete(where={"base_id": _unique_id(src)})
                continue
            total += _add_doc(collection, text, _source_label(src), _unique_id(src))
        except Exception as e:
            failures.append((src.name, str(e)))
            continue
    if failures:
        print(f"[ingest] 跳过 {len(failures)} 个文件：{failures}")
    return total


def ingest_upload(data: bytes, filename: str):
    """接收上传的 .txt 字节流，增量入库。返回新增片段数。

    与 ingest_preset 对齐：检测空文本并跳过；用随机后缀生成唯一 id，
    避免同名文件重复上传互相覆盖。
    """
    ef = get_embedding_function()
    client = get_client()
    collection = client.get_or_create_collection(COLLECTION, embedding_function=ef)
    text = _read_text(io.BytesIO(data))
    if not text.strip():
        print(f"[ingest] 上传的 {filename} 未读到文字（可能是空文件），已跳过")
        return 0
    base_id = f"{Path(filename).stem}_{uuid.uuid4().hex[:8]}"
    return _add_doc(collection, text, filename, base_id)
