"""Streamlit Web UI - AI 智能办公助手"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import streamlit as st
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agent.graph import AgentGraph
from src.knowledge.pipeline import DocumentPipeline
from src.models import ClaudeAdapter, OpenAIAdapter, OllamaAdapter
from src.utils.config import config


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="AI 智能办公助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Xcscv889/ai-assistant",
        "About": "AI Agent 智能办公助手 v0.1.1 — 知识问答 | 文档处理 | 消息通知",
    },
)


# ============================================================
# 全局 CSS 样式
# ============================================================
st.markdown("""
<style>
    /* ===== 全局变量 ===== */
    :root {
        --primary: #4F46E5;
        --primary-light: #818CF8;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --bg-card: rgba(255,255,255,0.05);
        --border: rgba(255,255,255,0.1);
    }

    /* ===== 主容器 ===== */
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1100px;
    }

    /* ===== 顶部标题栏 ===== */
    .app-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1.5rem 1.5rem 1rem 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
        background: linear-gradient(135deg, rgba(79,70,229,0.08) 0%, rgba(129,140,248,0.03) 100%);
        border-radius: 16px;
    }
    .app-header-icon {
        font-size: 2.8rem;
        line-height: 1;
    }
    .app-header-text h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4F46E5, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-header-text p {
        margin: 0.2rem 0 0 0;
        color: #94A3B8;
        font-size: 0.85rem;
    }

    /* ===== 统计卡片 ===== */
    .stat-cards {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        flex: 1;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid var(--border);
        background: var(--bg-card);
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(79,70,229,0.15);
    }
    .stat-card .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #E2E8F0;
    }
    .stat-card .stat-label {
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 0.2rem;
    }
    .stat-card.primary { border-left: 3px solid #4F46E5; }
    .stat-card.success { border-left: 3px solid #10B981; }
    .stat-card.warning { border-left: 3px solid #F59E0B; }

    /* ===== 功能快捷入口 ===== */
    .quick-actions {
        display: flex;
        gap: 0.6rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    .quick-action-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: 1px solid var(--border);
        background: var(--bg-card);
        font-size: 0.8rem;
        color: #CBD5E1;
        cursor: pointer;
        transition: all 0.15s;
    }
    .quick-action-chip:hover {
        border-color: var(--primary-light);
        background: rgba(79,70,229,0.1);
        color: #E2E8F0;
    }

    /* ===== 聊天消息美化 ===== */
    .stChatMessage {
        padding: 1rem 1.2rem !important;
        border-radius: 14px !important;
    }

    /* ===== 侧边栏美化 ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,23,42,0.98) 0%, rgba(30,41,59,0.96) 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 1.2rem !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.95rem !important;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 1.2rem;
    }

    /* ===== 按钮 ===== */
    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid var(--border) !important;
        transition: all 0.15s !important;
    }
    .stButton > button:hover {
        border-color: var(--primary-light) !important;
    }

    /* ===== Selectbox / 文件上传 ===== */
    .stSelectbox > div > div {
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] {
        border-radius: 12px !important;
    }

    /* ===== 聊天输入 ===== */
    [data-testid="stChatInput"] textarea {
        border-radius: 14px !important;
    }

    /* ===== Toast / 提示 ===== */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    /* ===== 滚动条 ===== */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.08);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.15); }

    /* ===== 空状态 ===== */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #64748B;
    }
    .empty-state .empty-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .empty-state h3 {
        color: #94A3B8;
        margin-bottom: 0.5rem;
    }
    .empty-state p {
        font-size: 0.85rem;
        max-width: 400px;
        margin: 0 auto;
    }

    /* ===== 响应式 ===== */
    @media (max-width: 768px) {
        .stat-cards { flex-direction: column; }
        .app-header { flex-direction: column; text-align: center; }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State
# ============================================================
def init_session():
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.messages = []
        st.session_state.model_key = config.get("models", "models.default", "deepseek")
        st.session_state.agent = None
        st.session_state.doc_pipeline = None


def create_model(model_key: str):
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


async def process_message(user_input: str) -> str:
    agent: AgentGraph = st.session_state.agent
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    result = await agent.run(user_input=user_input, messages=messages)
    return result.get("final_response", "处理完成")


# ============================================================
# 顶部标题栏
# ============================================================
def render_header():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("""
        <div class="app-header">
            <div class="app-header-icon">🤖</div>
            <div class="app-header-text">
                <h1>AI Agent 智能办公助手</h1>
                <p>统一入口 — 知识问答 · 文档处理 · 消息通知</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.session_state.initialized:
            info = st.session_state.model.get_model_info()
            kb_count = st.session_state.doc_pipeline.vector_store.count()
            st.markdown(f"""
            <div style="text-align:right;padding-top:1.5rem;color:#94A3B8;font-size:0.78rem;">
                <div style="color:#E2E8F0;font-weight:600;">{info['provider'].title()}</div>
                <div>{info['model']}</div>
                <div style="margin-top:0.3rem;">📚 {kb_count} 文档块</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# 统计卡片
# ============================================================
def render_stat_cards():
    if not st.session_state.initialized:
        return
    kb_count = st.session_state.doc_pipeline.vector_store.count()
    msg_count = len(st.session_state.messages)
    info = st.session_state.model.get_model_info()

    st.markdown(f"""
    <div class="stat-cards">
        <div class="stat-card primary">
            <div class="stat-value">{info['model']}</div>
            <div class="stat-label">当前模型</div>
        </div>
        <div class="stat-card success">
            <div class="stat-value">{kb_count}</div>
            <div class="stat-label">知识库文档块</div>
        </div>
        <div class="stat-card warning">
            <div class="stat-value">{msg_count}</div>
            <div class="stat-label">本轮对话数</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 快捷入口
# ============================================================
def render_quick_actions():
    st.markdown("""
    <div class="quick-actions">
        <div class="quick-action-chip">💡 总结文档</div>
        <div class="quick-action-chip">🔍 搜索知识库</div>
        <div class="quick-action-chip">📨 发送通知</div>
        <div class="quick-action-chip">📄 分析简历</div>
        <div class="quick-action-chip">🌐 网络搜索</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 空状态引导
# ============================================================
def render_empty_state():
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">👋</div>
        <h3>欢迎使用 AI 办公助手</h3>
        <p>我是您的智能办公伙伴，可以帮助您进行知识问答、文档处理、消息通知等任务。</p>
        <p style="margin-top:1rem;font-size:0.8rem;">💡 试试输入「帮我总结一下上周的会议纪要」或「发通知提醒大家交周报」</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 侧边栏
# ============================================================
def render_sidebar():
    with st.sidebar:
        # ---- 头部 ----
        st.markdown("""
        <div style="text-align:center;padding:0.5rem 0 1rem 0;">
            <div style="font-size:2.5rem;">🤖</div>
            <div style="font-weight:700;font-size:1.1rem;">AI 办公助手</div>
            <div style="font-size:0.7rem;color:#64748B;">v0.1.1</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ---- 模型设置 ----
        st.markdown("### ⚡ 模型设置")
        models_config = config.load("models").get("models", {})
        model_options = [
            k for k, v in models_config.items()
            if k != "default" and isinstance(v, dict)
        ]

        # 带 emoji 的模型标签
        model_labels = {
            "deepseek": "🟢 DeepSeek",
            "deepseek-reasoner": "🧠 DeepSeek R1",
            "claude": "🟣 Claude",
            "openai": "🔵 OpenAI",
            "ollama": "🟠 Ollama",
            "lightweight": "⚪ Ollama Light",
        }
        options_display = [model_labels.get(k, k) for k in model_options]

        current_idx = model_options.index(st.session_state.model_key) if st.session_state.model_key in model_options else 0
        selected_display = st.selectbox("选择模型", options_display, index=current_idx)
        selected_model = model_options[options_display.index(selected_display)]

        if selected_model != st.session_state.model_key:
            st.session_state.model_key = selected_model
            st.session_state.initialized = False
            st.rerun()

        st.divider()

        # ---- 知识库 ----
        st.markdown("### 📚 知识库管理")

        if st.button("📊 刷新统计", use_container_width=True):
            if st.session_state.initialized:
                count = st.session_state.doc_pipeline.vector_store.count()
                st.toast(f"知识库共 {count} 个文档块", icon="📚")

        if st.button("📋 文档列表", use_container_width=True):
            if st.session_state.initialized:
                docs = st.session_state.doc_pipeline.vector_store.list_all()
                if docs:
                    files = {}
                    for doc in docs:
                        fname = doc.get("metadata", {}).get("filename", "未知")
                        files[fname] = files.get(fname, 0) + 1
                    with st.expander(f"共 {len(files)} 个文件", expanded=True):
                        for fname, cnt in files.items():
                            st.caption(f"📄 {fname} ({cnt} 块)")
                else:
                    st.caption("知识库为空")

        # 上传
        uploaded_file = st.file_uploader(
            "上传文档",
            type=["pdf", "docx", "pptx", "xlsx", "md", "txt"],
            key="kb_upload",
            help="支持 PDF / Word / PPT / Excel / Markdown / 文本",
        )

        if uploaded_file is not None:
            temp_dir = Path("data/uploads")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / uploaded_file.name
            temp_path.write_bytes(uploaded_file.getbuffer())

            if st.button(f"📥 导入到知识库", use_container_width=True, type="primary"):
                if st.session_state.initialized:
                    with st.spinner("正在解析文档..."):
                        ids = st.session_state.doc_pipeline.ingest_file(str(temp_path))
                    st.toast(f"已导入 {uploaded_file.name} ({len(ids)} 个块)", icon="✅")
                    st.session_state.agent = AgentGraph(
                        st.session_state.model,
                        st.session_state.doc_pipeline,
                    )

        st.divider()

        # ---- 操作 ----
        st.markdown("### ⚙️ 工具")
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("🔄 重置知识库", use_container_width=True):
            if st.session_state.initialized:
                st.session_state.doc_pipeline.vector_store.clear()
                st.session_state.doc_pipeline = DocumentPipeline()
                st.session_state.agent = AgentGraph(
                    st.session_state.model,
                    st.session_state.doc_pipeline,
                )
                st.toast("知识库已重置", icon="🔄")

        st.divider()
        st.caption("GitHub: [Xcscv889/ai-assistant](https://github.com/Xcscv889/ai-assistant)")


# ============================================================
# 主界面
# ============================================================
def main():
    init_session()
    initialize_app()

    # 侧边栏
    render_sidebar()

    # ---- 主内容区 ----
    render_header()
    render_stat_cards()
    render_quick_actions()

    # 聊天区域
    if not st.session_state.messages:
        render_empty_state()
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 输入
    user_input = st.chat_input("输入您的问题，或使用命令...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = asyncio.run(process_message(user_input))
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()
