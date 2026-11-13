# python-doc-rag
 python-doc-rag

> 本地 RAG 问答服务：把 Python 官方教程切片入库，中文提问返回带出处的答案；问题超出语料范围时明确拒答，不硬编、不拿无关片段充数。

这个项目的重点不是「能问答」，而是每个结论都有可复现的数字：自建 52 道评测题，评测集本身还带一道校验，防止「自己出题自己答」把指标做虚。

## 关键数字

| 指标 | 结果 |
| --- | --- |
| 检索 hit@5 / hit@1 / MRR | 100% / 89% / 0.929（27 题） |
| 库外拒答率（负样本 25 条） | 25/25 = 100% |
| 正样本误拒 | 0/27 |
| 生成引用合法率 | 27/27 |
| 语料规模 | 中文 275 片段、英文 544 片段 |

## 1. 能做什么

- 扫目录批量建库；也支持网页上传单个 `.txt` 增量入库（UTF-8 / GBK / GB18030 / 带 BOM）
- 提问后返回答案 + 参考来源文件
- 问题超出文档范围时明确拒答，不把无关片段当依据
- 网页界面（Streamlit），浏览器打开即用

## 2. 快速开始

前置：Python 3.12（本机实测版本）、一个 DeepSeek API Key。

```bash
pip install -r requirements.txt
cp .env.example .env            # Windows 用 copy
# 填入 DEEPSEEK_API_KEY
python build.py                 # 建库，中文库 275 片段
streamlit run app.py            # http://localhost:8501
```

> `db/` 必须可写。Chroma 用 SQLite 持久化，`build.py` 会往 `db/chroma.sqlite3` 写文件——目录只读会直接报 `readonly database`。

Windows 上可以直接双击 `run.bat`（已切 UTF-8 代码页，中文日志不乱码）和 `build.bat`。

首次提问会先加载 bge 模型，之后单次检索毫秒级返回。

## 3. 链路

```text
.txt → 读取(多编码) → 句子感知切片(600字/100重叠) → bge 向量化 → Chroma
                                                                    ↑
提问 → 加检索指令前缀 → 向量检索 Top-5 → 拼 [来源:文件] 上下文 → DeepSeek → 答案 + 来源
```
关键参数在 `src/config.py`：`CHUNK_SIZE=600`、`CHUNK_OVERLAP=100`、`TOP_K=5`；向量模型 `BAAI/bge-small-zh-v1.5`（92 MB，本地跑），生成模型 `deepseek-chat`。

## 4. 两套语料

同一份官方教程有中英两个版本，17 篇章节一一对应，文件名完全相同。

| 语料 | 目录 | 片段数 | collection |
| --- | --- | --- | --- |
| 中文（默认） | `docs_zh/tutorial/` | 275 | `documents_zh` |
| 英文 | `docs/tutorial/` | 544 | `documents` |

因为文件名两边一致，同一套评测题不用改一个字，就能在两个库上各跑一遍。

切换方式：

```bash
set DOCSET=en            # CMD
$env:DOCSET="en"         # PowerShell
export DOCSET=en         # bash
```

切换后对应 collection 是空的，需要重新点一次「建立 / 刷新预置知识库」。

英文片段数多是正常的——英文原文更长，同样的 600 字符窗口自然切出更多段；两边都是 17 个 base_id，无孤儿片段，不是重复入库。

## 5. 实测结果

### 5.1 检索（27 正样本 + 25 负样本，不调大模型）

```bash
python eval.py                       # 中文库
set DOCSET=en && python eval.py      # 英文库（先按上节切 DOCSET）
```

| 配置 | hit@5 | hit@1 | MRR |
| --- | --- | --- | --- |
| 中文库 | 100% | 89% | 0.929 |
| 英文库 | 89% | 59% | 0.707 |

原始命中数：中文库 27/27 和 24/27，英文库 24/27 和 16/27（共 27 道题）。

同一套样本、同一个向量模型、同一套切片参数，只换语料语言：hit@1 从 59% 到 89%，hit@5 从 89% 到 100%。bge-small-zh-v1.5 是中文模型，拿它编码英文 chunk 属于跨语言借用，排序掉得厉害，而且有 3 条正样本的正确章节根本没进前 5——跨语言不只是排序变差，是真会漏召。

拒答率和误拒率（两种语料都是 100% / 0%）由 `out_of_scope` 的词面规则判定，不调模型。它只覆盖带英文专名的库外问题，纯中文的库外问题靠 5.4 里的模型判断。

### 5.2 评测集本身也要校验

```bash
python eval.py --validate
```

防的是「自己出题自己答」：负样本里的英文专名必须真的不在语料中出现，否则它根本不是库外问题，拒答率会被凭空抬高。

用这条检查剔除过 SQLAlchemy、numpy 两条——教程里确实提过。当前中文语料 132,862 字符，负样本 25/25 合格、正样本 27/27 命中。

### 5.3 重排（可选，默认关闭）

粗排取 20 条，`BAAI/bge-reranker-base` 重排后取 5。

| 配置 | hit@5 | hit@1 | MRR |
| --- | --- | --- | --- |
| 中文库 | 100% | 89% | 0.929 |
| 中文库 + 重排 | 100% | 93% | 0.963 |
| 英文库 | 89% | 59% | 0.707 |
| 英文库 + 重排 | 96% | 74% | 0.819 |

两种语料下重排都是正收益，英文库增益更大（+15pt）。代价是延迟：重排在 CPU 上跑，单次检索从毫秒级涨到秒级。中文库已经是 89%，+4pt 换几百倍延迟不划算，所以默认关闭。

```bash
python eval.py --rerank
```

复盘一句：这个结论我一开始搞错了。用 10 条样本测英文库，重排看着是负收益（hit@1 70%→50%），我还给它编了个「跨语言导致单语重排器失效」的解释。样本扩到 52 条后结论反了——英文库上重排增益最大。10 条样本足够得出一个自信又错误的结论，现在这 27+25 条也算不上充分。

### 5.4 拒答为什么不用距离阈值

```bash
python eval.py --calibrate
```

中文库正样本 top1 距离 0.614–0.999（中位 0.767），负样本 0.699–1.175（中位 0.930）。中位数差得明显，但区间重叠——距离最近的负样本（0.699）比距离最远的正样本（0.999）还近，没有任何单一阈值能同时不误拒、不漏拒。

所以改成两层：

1. 词面兜底：问题里的英文专名（Redis、Django、MySQL…）在 top-5 片段里一个都没出现，直接拒，不调模型。
2. 模型判断：prompt 要求「片段无对应信息就只回一句『文档中未找到相关信息』」，纯中文的库外问题走这条。

第一条只管带专名的库外问题，这是有意的分层，不是漏判。

### 5.5 生成质量（27 条，每条一次 LLM 调用）

```bash
python eval_generation.py
```

| 指标 | 结果 |
| --- | --- |
| 非拒答率 | 27/27 = 100% |
| 引用存在率 | 27/27 = 100% |
| 引用合法率 | 27/27 = 100% |
| 平均答案长度 | 431 字 |

引用合法率是最有用的一条：它检查答案里标的文件是否真的在这次检索结果中，能抓住模型编造出处。

没做 faithfulness：让 DeepSeek 给 DeepSeek 的答案打分会系统性偏高，得出的分数没有说服力。这项需要人工抽检，属于未验证状态，不放进指标里。

### 5.6 单元测试

```bash
pip install -r requirements-dev.txt
python -m pytest        # 39 passed
```

覆盖四个纯函数模块：切片、RRF 融合、库外判定、多编码解码。这几个模块不碰模型和网络，出错代价却最大（切片错了后面全错），优先覆盖。

## 6. 踩过的坑

| 现象 | 根因 | 处理 |
| --- | --- | --- |
| 拒答还返回来源 | 模型说「没找到」，前端却列了 4 个无关文件 | `rag.answer()` 出口统一识别拒答标记、归一化话术并清空 sources，加「长度 ≤ 40 字」条件避免长答案误判 |
| 代码块被压平 | 切片后 `def` / `if` / `return` 全部顶格，语义变了 | 断句时用 `rstrip()` 只去行尾空白，保留行首缩进 |
| GBK 文件上传整页报错 | 记事本存的 txt 按 utf-8 解码抛异常，Streamlit 整页 traceback | 按 `utf-8-sig` → `utf-8` → `gb18030` 依次尝试，全失败给可读提示 |
| 评测指标虚高 | 子串匹配来源，`stdlib2.txt` 被判成 `stdlib` 命中 | 改成按文件名精确比对 |
| 标注歧义被当成检索失败 | 「命令行参数」中文版在 `stdlib.txt`（argparse），英文版在 `interpreter.txt` | 改成多标签，任一命中即算对 |
| HF 联网检查打断评测 | 代理抖动时抛 `httpx.ProxyError(502)` | 模型已缓存后设 `HF_HUB_OFFLINE=1`，要下新模型时覆盖 |

## 7. 已知限制

- 评测集 27 正 + 25 负，没覆盖同义改写、多跳问题、长尾表述；负样本也偏向「带英文专名」的显式库外问题。要下结论，样本得上百条。
- faithfulness 和答案相关性都没测。检索命中不等于答案正确，这是最大的评测盲区。
- 词面兜底只认英文专名，纯中文的库外问题全靠模型自觉。
- `src/query_expand.py`（查询扩展 + RRF 融合）只在 `DOCSET=en` 下有实际作用；中文语料下 `expand_query()` 直接返回原查询、不调模型，所以 `--expand` 在默认配置里等于空跑。
- 重排的秒级延迟是 CPU 上的数字，有 GPU 会好很多，但没测。

## 8. 目录

```text
python-doc-rag/
├── app.py                Streamlit 界面
├── build.py              命令行建库
├── eval.py               检索评测：hit@K / 拒答率 / 评测集校验
├── eval_generation.py    生成评测：非拒答率 / 引用存在率 / 引用合法率
├── requirements.txt
├── requirements-dev.txt  只装 pytest
├── pytest.ini
├── .env.example
├── docs/tutorial/        英文语料 17 篇
├── docs_zh/tutorial/     中文语料 17 篇（默认）
├── tests/                39 个用例
├── tools/fetch_docs_zh.py   抓官方中文教程 HTML → 纯文本
├── src/
│   ├── config.py         路径、模型、DOCSET 切换
│   ├── chunk.py          句子感知切片
│   ├── embed.py          Embedding 单例 + 查询前缀
│   ├── ingest.py         读取入库（多编码 / 预置 / 上传）
│   ├── retriever.py      向量检索（单路 / 多路融合）
│   ├── rerank.py         CrossEncoder 重排（可选）
│   ├── query_expand.py   查询扩展（仅 en） 
│   ├── fusion.py         RRF 融合
│   ├── generator.py      DeepSeek 调用
│   └── rag.py            主流程 + 拒答归一化
└── db/                   Chroma 向量库（不入库）
```

`venv` / `db` / `.env` / `.idea` / `.pytest_cache` / `*.log` 已在 `.gitignore` 中忽略。

 ##9. 环境

| 项 | 版本 |
| --- | --- |
| Python | 3.12 |
| 向量库 | chromadb 1.5.9（SQLite 持久化） |
| Embedding | bge-small-zh-v1.5（本地，92 MB） |
| 生成 | DeepSeek `deepseek-chat`（OpenAI 兼容接口） |
| 依赖 | streamlit 1.64.0 / sentence-transformers 6.1.0 / langchain-core 1.7.1 |
