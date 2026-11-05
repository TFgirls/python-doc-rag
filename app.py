import streamlit as st
from src.rag import answer
from src.ingest import ingest_preset, ingest_upload
from src.config import PRESET_DOCS_DIR, DOCSET, COLLECTION

_DOCSET_LABEL = {"zh": "中文版教程", "en": "英文版教程"}.get(DOCSET, DOCSET)

st.set_page_config(page_title="文档知识库问答", page_icon="📚")
st.title("📚 通用文档知识库问答")
st.caption("基于本地文档的检索增强问答 · Chroma + bge-small-zh + DeepSeek")

with st.sidebar:
    st.header("知识库")
    st.caption(f"当前数据集：{_DOCSET_LABEL}（DOCSET={DOCSET}，collection={COLLECTION}）")
    st.caption(f"预置数据目录：{PRESET_DOCS_DIR}")
    if st.button("🔄 建立 / 刷新预置知识库", use_container_width=True):
        with st.spinner("解析并建立向量索引（首次会下载中文向量模型，请稍候）..."):
            n = ingest_preset()
        st.success(f"预置知识库完成，共 {n} 个文本片段")

    up = st.file_uploader("➕ 上传额外增量入库", type=["txt"])
    if up is not None:
        if st.button("加入知识库", use_container_width=True):
            with st.spinner("解析并入库..."):
                m = ingest_upload(up.getvalue(), up.name)
            if m == 0:
                st.warning(f"未入库任何片段：{up.name}（可能是扫描件/无文字层/空文件，已跳过）")
            else:
                st.success(f"已入库 {m} 个片段：{up.name}")

query = st.text_input("输入问题，例如：类和对象怎么定义？怎么读写文件？怎么处理异常？")
if st.button("提问", use_container_width=True) and query:
    with st.spinner("检索 + 生成中..."):
        ans, sources, _ = answer(query)
    st.markdown(ans)
    if sources:
        st.caption("📎 参考来源：" + "、".join(sources))
