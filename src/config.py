import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "db"

from chromadb import PersistentClient

_client = None


def get_client():
    """返回全局唯一的 Chroma 持久化客户端单例，避免每次检索/入库重复打开 SQLite。"""
    global _client
    if _client is None:
        _client = PersistentClient(path=str(DB_DIR))
    return _client

# ---------------------------------------------------------------------------
# 数据集切换（对照实验用）
#
#   en = Python 官方英文教程（docs/tutorial/，17 篇）
#   zh = 同内容官方中文版（docs_zh/tutorial/，17 篇，章节一一对应）
#
# 两套写入不同 collection，互不干扰，可用同一套评测样本直接做 A/B。
#   切换方式：set DOCSET=zh  /  $env:DOCSET="zh"
# ---------------------------------------------------------------------------
DOCSET = os.getenv("DOCSET", "zh").strip().lower()
_DOCSETS = {
    "en": ("docs", "documents"),
    "zh": ("docs_zh", "documents_zh"),
}
if DOCSET not in _DOCSETS:
    raise ValueError(f"未知 DOCSET={DOCSET!r}，可选：{', '.join(_DOCSETS)}")
_DEFAULT_DOCS_SUBDIR, _DEFAULT_COLLECTION = _DOCSETS[DOCSET]

# 预置知识库数据
# 默认跟随 DOCSET，也可用环境变量 PRESET_DOCS_DIR 直接覆盖（便于指向外部目录）
PRESET_DOCS_DIR = Path(os.getenv("PRESET_DOCS_DIR", BASE_DIR / _DEFAULT_DOCS_SUBDIR))

# DeepSeek（OpenAI 兼容接口）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# 本地中文 Embedding 模型（离线可用，不依赖外部服务）
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# 切片与检索参数
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 5

COLLECTION = os.getenv("COLLECTION", _DEFAULT_COLLECTION)
