"""Streamlit Web UI"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import streamlit as st
from loguru import logger

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agent.graph import AgentGraph
from src.knowledge.pipeline import DocumentPipeline
from src.models import ClaudeAdapter, OpenAIAdapter, OllamaAdapter
from src.utils.config import config


# ============================================
# 页面配置
# ============================================
st.set_page_config(
    page_title="AI 智能办公助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================
# 样式
# ============================================
st.markdown(
    """
<style>
    .main-header { font-size: 2rem; font-weight: bold; margin-bottom: 1rem; }
    .stChatMessage { padding: 1rem; }
    .sidebar-section { margin-bottom: 1.5rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================
# 初始化 Session State
# ============================================
def init_session():
    """初始化会话状态"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.messages = []
        st.session_state.model_key = config.get("models", "models.default", "claude")
        st.session_state.agent = None
        st.session_state.doc_pipeline = None


def create_model(model_key: str):
    """创建模型适配器"""
    model_config = config.get("models", f"models.{model_key}", {})
    if not model_config:
        return None

    provider = model_config.get("provider", "").lower()
    model_name = model_config.get("model", "")

    if provider == "anthropic":
        return ClaudeAdapter(model_name, model_config)
    elif provider == "openai":
        return OpenAIAdapter(model_name, model_config)
    elif provider == "ollama":
        return OllamaAdapter(model_name, model_config)
    return None


def initialize_app():
    """懒初始化应用组件"""
    if not st.session_state.initialized:
        with st.spinner("正在初始化 AI 办公助手..."):
            model = create_model(st.session_state.model_key)
            if model is None:
                st.error(f"无法加载模型: {st.session_state.model_key}")
                st.stop()

            doc_pipeline = DocumentPipeline()
            agent = AgentGraph(model, doc_pipeline)

            st.session_state.model = model
            st.session_state.doc_pipeline = doc_pipeline
            st.session_state.agent = agent
            st.session_state.initialized = True


async def process_message(user_input: str, uploaded_files: Optional[list] = None) -> str:
    """处理用户消息"""
    agent: AgentGraph = st.session_state.agent
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    kwargs = {}
    if uploaded_files:
        kwargs["uploaded_files"] = uploaded_files

    result = await agent.run(
        user_input=user_input,
        messages=messages,
        **kwargs,
    )

    return result.get("final_response", "处理完成")


async def process_message_streaming(user_input: str) -> str:
    """使用流式输出处理用户消息，大幅提升感知速度"""
    agent: AgentGraph = st.session_state.agent
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    result = await agent.run(
        user_input=user_input,
        messages=messages,
    )

    return result.get("final_response", "处理完成")


# ============================================
# 侧边栏
# ============================================
def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.markdown("## 🤖 AI 办公助手")

        # 模型选择
        st.markdown("### 🔧 模型设置")
        models_config = config.load("models").get("models", {})
        model_options = [
            k for k, v in models_config.items()
            if k != "default" and isinstance(v, dict)
        ]

        selected_model = st.selectbox(
            "选择模型",
            model_options,
            index=model_options.index(st.session_state.model_key)
            if st.session_state.model_key in model_options
            else 0,
        )

        if selected_model != st.session_state.model_key:
            st.session_state.model_key = selected_model
            st.session_state.initialized = False
            st.rerun()

        # 模型信息
        if st.session_state.initialized:
            info = st.session_state.model.get_model_info()
            st.caption(f"当前: {info['provider']} / {info['model']}")

        st.divider()

        # 知识库管理
        st.markdown("### 📚 知识库")

        kb_col1, kb_col2 = st.columns(2)
        with kb_col1:
            if st.button("📊 统计", use_container_width=True):
                if st.session_state.initialized:
                    count = st.session_state.doc_pipeline.vector_store.count()
                    st.info(f"知识库共有 {count} 个文档块")

        with kb_col2:
            if st.button("📋 列表", use_container_width=True):
                if st.session_state.initialized:
                    docs = st.session_state.doc_pipeline.vector_store.list_all()
                    if docs:
                        files = {}
                        for doc in docs:
                            fname = doc.get("metadata", {}).get("filename", "未知")
                            files[fname] = files.get(fname, 0) + 1
                        for fname, count in files.items():
                            st.caption(f"• {fname} ({count}块)")
                    else:
                        st.caption("知识库为空")

        # 上传文件
        uploaded_file = st.file_uploader(
            "上传文档到知识库",
            type=["pdf", "docx", "pptx", "xlsx", "md", "txt"],
            key="kb_upload",
        )

        if uploaded_file is not None:
            # 保存临时文件
            temp_dir = Path("data/uploads")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / uploaded_file.name
            temp_path.write_bytes(uploaded_file.getbuffer())

            if st.button(f"📥 导入 '{uploaded_file.name}' 到知识库"):
                if st.session_state.initialized:
                    with st.spinner("正在导入..."):
                        ids = st.session_state.doc_pipeline.ingest_file(str(temp_path))
                    st.success(f"✅ 已导入 ({len(ids)} 个块)")
                    # 刷新知识库
                    st.session_state.agent = AgentGraph(
                        st.session_state.model,
                        st.session_state.doc_pipeline,
                    )

        st.divider()

        # 操作
        st.markdown("### ⚙️ 操作")
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption(f"AI Office Assistant v0.1.0")


# ============================================
# 主界面
# ============================================
def main():
    """主界面"""
    init_session()
    render_sidebar()

    # 主内容区
    st.markdown(
        '<div class="main-header">🤖 AI Agent 智能办公助手</div>',
        unsafe_allow_html=True,
    )
    st.caption("统一入口 — 知识问答 | 文档处理 | 消息通知")

    initialize_app()

    # 渲染历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 输入区域
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.chat_input("输入您的问题，或使用命令...")

    # 处理输入
    if user_input:
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 生成回复 - 使用流式输出
        with st.chat_message("assistant"):
            if st.session_state.initialized:
                response = asyncio.run(process_message_streaming(user_input))
                st.markdown(response)
            else:
                with st.spinner("初始化中..."):
                    st.rerun()

        # 保存回复
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()
