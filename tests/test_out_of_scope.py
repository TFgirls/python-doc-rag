"""库外判定（词面兜底）的单元测试。

out_of_scope 是纯函数：只看 query 里的专有名词是否出现在检索片段里。
它是拒答的第一道防线，判断错了会导致误拒（该答不答）或漏拒（硬编答案）。
"""
from src.rag import _latin_terms, out_of_scope

DOCS_EN = ["Python 提供标准库 os、sys 等模块。", "使用 import 语句可以导入模块。"]
DOCS_WITH_REDIS = ["Redis 是一个内存数据库，可通过 redis-py 连接。"]
DOCS_ZH = ["类提供了把数据和功能绑定在一起的方法。", "实例对象可以访问类属性。"]


def test_latin_terms_extracts_and_lowercases():
    assert _latin_terms("怎么用 Redis 和 Django") == {"redis", "django"}


def test_latin_terms_drops_common_keywords():
    assert _latin_terms("for 循环和 import 语句") == set()
    assert _latin_terms("Python 里 class 怎么定义") == set()


def test_latin_terms_ignores_single_letter():
    """单字母（如 C）不算专有名词，避免误判。"""
    assert _latin_terms("C 语言怎么写") == set()


def test_out_of_scope_true_when_term_absent():
    assert out_of_scope("怎么用 Redis 做缓存", DOCS_EN) is True


def test_out_of_scope_false_when_term_present():
    assert out_of_scope("怎么用 Redis 做缓存", DOCS_WITH_REDIS) is False


def test_out_of_scope_case_insensitive():
    assert out_of_scope("REDIS 怎么用", DOCS_WITH_REDIS) is False


def test_pure_chinese_query_is_not_flagged():
    """无英文专名的查询不由词面兜底处理，交给模型语义判断。"""
    assert out_of_scope("红烧肉怎么做", DOCS_ZH) is False
    assert out_of_scope("类和对象什么关系", DOCS_ZH) is False


def test_common_keyword_only_is_not_flagged():
    assert out_of_scope("for 循环怎么写", DOCS_ZH) is False
    assert out_of_scope("Python 怎么定义类", DOCS_ZH) is False


def test_any_term_present_is_enough():
    """多个专名只要有一个出现在片段里，就不算库外。"""
    assert out_of_scope("Redis 和 MongoDB 的区别", ["我们讨论过 Redis 的用法。"]) is False


def test_all_terms_absent_is_out_of_scope():
    assert out_of_scope("Redis 和 MongoDB 的区别", DOCS_EN) is True


def test_empty_docs_is_out_of_scope():
    assert out_of_scope("怎么用 Redis", []) is True
