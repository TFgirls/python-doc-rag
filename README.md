# python-doc-rag

一个本地 RAG 问答服务：把一批 `.txt` 文档切片入库，用中文提问，检索相关片段交给 DeepSeek 生成答案，并在答案里标注出处。

语料是 Python 官方教程 tutorial 章节（中英两版各 17 篇），没有私有数据，克隆下来配个 Key 就能跑。

## 能做什么

- 扫目录批量建库（也可以网页上传单个 `.txt` 增量入库，支持 UTF-8 / GBK / GB18030 / 带 BOM）
- 提问后返回答案 + 参考来源文件
- 问题超出文档范围时明确拒答，不硬编、不把无关片段当依据
- 网页界面（Streamlit），浏览器打开即用

## 快速开始

前置：Python 3.12（本机实测版本）、一个 DeepSeek API Key。

```bash
pip install -r requirements.txt
copy .env.example .env          # 填入 DEEPSEEK_API_KEY
python build.py                 # 建库，中文库 275 片段，实测 28.3 秒
streamlit run app.py            # 打开 http://localhost:8501
```

Windows 上可以直接双击 `run.bat`（已切 UTF-8 代码页，中文日志不乱码）和 `build.bat`。

首次提问会先加载 bge 模型（本机实测 2.4 秒，冷启动更久），之后单次检索 21.5 毫秒。

> `db/` 必须可写。Chroma 用 SQLite 持久化，在只读盘上会报 `readonly database`。

## 链路

```
.txt → 读取(多编码) → 句子感知切片(600字/100重叠) → bge 向量化 → Chroma
                                                                    ↑
提问 → 加检索指令前缀 → 向量检索 Top-5 → 拼 [来源:文件] 上下文 → DeepSeek → 答案+来源
```

关键参数（`src/config.py`）：`CHUNK_SIZE=600`、`CHUNK_OVERLAP=100`、`TOP_K=5`，向量模型 `BAAI/bge-small-zh-v1.5`（92 MB，本地跑），生成模型 `deepseek-chat`（temperature 0.3，超时 30 秒）。

## 两套语料

同一份教程的两个语言版本，章节和文件名一一对应，写进不同 collection，可用同一套样本做对照：

| DOCSET | 目录 | collection | 文档 | 片段 |
|---|---|---|---|---|
| `zh`（默认） | `docs_zh/tutorial/` | `documents_zh` | 17 | 275 |
| `en` | `docs/tutorial/` | `documents` | 17 | 544 |

```bash
set DOCSET=en        # CMD
$env:DOCSET="en"     # PowerShell
export DOCSET=en     # bash
```

切换后对应 collection 是空的，需要重新点一次「建立 / 刷新预置知识库」。英文片段数多是因为同样 600 字符窗口装得下更少的内容，不是重复入库（17 个 `base_id`，无孤儿片段）。

## 实测结果

以下全部是 2026-11-06 在本机跑出来的，命令都在对应小节里，可复现。

### 检索（27 正样本 + 25 负样本，不调大模型）

```bash
python eval.py                     # 中文库，约 40 秒
set DOCSET=en && python eval.py    # 英文库（先按上节切 DOCSET）
```

| 配置 | hit@5 | hit@1 | MRR |
|---|---|---|---|
| 中文库（默认） | 100%（27/27） | 89%（24/27） | 0.929 |
| 英文库 | 89%（24/27） | 59%（16/27） | 0.707 |

同一套样本、同一个向量模型、同一套切片参数，只换语料语言：hit@1 从 59% 到 89%，hit@5 从 89% 到 100%。`bge-small-zh-v1.5` 是中文模型，拿它编码英文 chunk 属于跨语言借用，排序掉得厉害，而且有 3 条正样本的正确章节根本没进前 5——跨语言不只是排序变差，是真会漏召。

拒答率和误拒率（两种语料都是 100% / 0%）由 `out_of_scope` 的词面规则判定，不调模型——它只覆盖带英文专名的库外问题，纯中文的库外问题要靠下文「拒答为什么不用距离阈值」里的模型判断。

### 评测集本身也要校验

```bash
python eval.py --validate
```

防的是"自己出题自己答"：负样本里的英文专名必须真的不在语料中出现，否则它根本不是库外问题，拒答率会被凭空抬高。用这条检查剔除过 `SQLAlchemy`、`numpy` 两条——教程里确实提过。当前中文语料 132,862 字符，负样本 25/25 合格、正样本 27/27 命中。

### 重排（可选，默认关闭）

粗排取 20 条，`BAAI/bge-reranker-base` 重排后取 5。

| 配置 | hit@5 | hit@1 | MRR | 单次检索耗时 |
|---|---|---|---|---|
| 中文库 | 100% | 89% | 0.929 | 21.5 ms |
| 中文库 + 重排 | 100% | 93% | 0.963 | 6.8 s |
| 英文库 | 89% | 59% | 0.707 | 同量级 |
| 英文库 + 重排 | 96% | 74% | 0.819 | — |

两种语料下重排都是正收益，英文库增益更大（+15pt）。但中文库已经是 89%，+4pt 换 300 倍延迟，交互式问答不划算，所以默认关闭。要开：

```bash
python eval.py --rerank
```

复盘一句：这个结论我一开始搞错了。用 10 条样本测英文库，重排看着是负收益（hit@1 70%→50%），我还给它编了个「跨语言导致单语重排器失效」的解释。样本扩到 52 条后结论反了——英文库上重排增益最大。10 条样本足够得出一个自信又错误的结论，现在这 27+25 条也算不上充分。

### 拒答为什么不用距离阈值

```bash
python eval.py --calibrate
```

中文库正样本 top1 距离 0.614–0.999（中位 0.767），负样本 0.699–1.175（中位 0.930）。中位数差得明显，但区间重叠——距离最近的负样本（0.699）比距离最远的正样本（0.999）还近，没有任何单一阈值能同时不误拒、不漏拒。所以改成两层：

1. **词面兜底**：问题里的英文专名（Redis、Django、MySQL…）在 top-5 片段里一个都没出现，直接拒，不调模型。
2. **模型判断**：prompt 要求「片段无对应信息就只回一句『文档中未找到相关信息』」，纯中文的库外问题走这条。

第一条只管带专名的库外问题，这是有意的分层，不是漏判。

### 生成质量（27 条，每条一次 LLM 调用）

```bash
python eval_generation.py
```

| 指标 | 结果 |
|---|---|
| 非拒答率 | 27/27 = 100% |
| 引用存在率 | 27/27 = 100% |
| 引用合法率 | 27/27 = 100% |
| 平均答案长度 | 431 字 |

引用合法率是最有用的一条：它检查答案里标的文件是否真的在这次检索结果中，能抓住模型编造出处。

**没做 faithfulness**：让 DeepSeek 给 DeepSeek 的答案打分会系统性偏高，得出的分数没有说服力。这项需要人工抽检，属于未验证状态，不放进指标里。

### 单元测试

```bash
pip install -r requirements-dev.txt
python -m pytest        # 39 passed，约 34 秒
```

覆盖四个纯函数模块：切片、RRF 融合、库外判定、多编码解码。这几个模块不碰模型和网络，出错代价却最大（切片错了后面全错），优先覆盖。

## 几个踩过的坑

| 问题 | 现象 | 改法 |
|---|---|---|
| 拒答还返回来源 | 模型说"没找到"，前端却列了 4 个无关文件 | `rag.answer()` 出口统一识别拒答标记，归一化话术并清空 `sources`；加「长度 ≤ 40 字」条件，避免长答案里提到这句话被误判 |
| 代码块被压平 | 切片后 `def / if / return` 全部顶格，语义都变了 | 断句时用 `rstrip()` 只去行尾空白，保留行首缩进 |
| GBK 文件上传整页报错 | 记事本存的 txt 直接 `decode('utf-8')` 抛异常，Streamlit 整页 traceback | 按 `utf-8-sig → utf-8 → gb18030` 依次尝试，全失败给可读提示 |
| 评测指标虚高 | 用子串匹配来源，`stdlib2.txt` 被判成 `stdlib` 命中 | 改成按文件名精确比对 |
| 标注歧义被当成检索失败 | 「命令行参数」在中文版归到 `stdlib.txt`（argparse），英文版在 `interpreter.txt` | 改成多标签，任一命中即算对 |
| HF 联网检查打断评测 | 代理抖动时抛 `httpx.ProxyError(502)` | 模型已缓存后设 `HF_HUB_OFFLINE=1`，要用新模型时覆盖 |

## 已知限制

- 评测集 27 正 + 25 负，没覆盖同义改写、多跳问题、长尾表述；负样本也偏向"带英文专名"的显式库外问题。要下结论，样本得上百条。
- faithfulness 和答案相关性都没测。检索命中不等于答案正确，这是最大的评测盲区。
- 词面兜底只认英文专名，纯中文的库外问题全靠模型自觉。
- `src/query_expand.py`（查询扩展 + RRF 融合）只在 `DOCSET=en` 下有实际作用；中文语料下 `expand_query()` 直接返回原查询、不调模型，所以 `--expand` 在默认配置里等于空跑。
- 重排单次 6.8 秒是 CPU 上的数字，有 GPU 会好很多，但没测。

## 目录

```
python-doc-rag/
├── app.py               # Streamlit 界面
├── build.py             # 命令行建库
├── eval.py              # 检索评测：hit@K / 拒答率 / 评测集校验
├── eval_generation.py   # 生成评测：非拒答率 / 引用存在率 / 引用合法率
├── requirements.txt
├── requirements-dev.txt # 只装 pytest
├── pytest.ini
├── .env.example
├── docs/tutorial/       # 英文语料 17 篇
├── docs_zh/tutorial/    # 中文语料 17 篇（默认）
├── tests/               # 39 个用例
├── tools/fetch_docs_zh.py  # 抓官方中文教程 HTML → 纯文本
├── src/
│   ├── config.py        # 路径、模型、DOCSET 切换
│   ├── chunk.py         # 句子感知切片
│   ├── embed.py         # Embedding 单例 + 查询前缀
│   ├── ingest.py        # 读取入库（多编码 / 预置 / 上传）
│   ├── retriever.py     # 向量检索（单路 / 多路融合）
│   ├── rerank.py        # CrossEncoder 重排（可选）
│   ├── query_expand.py  # 查询扩展（仅 en）
│   ├── fusion.py        # RRF 融合
│   ├── generator.py     # DeepSeek 调用
│   └── rag.py           # 主流程 + 拒答归一化
└── db/                  # Chroma 向量库（不入库）
```

`venv / db / .env / .idea / .pytest_cache / *.log` 已在 `.gitignore` 中忽略。
