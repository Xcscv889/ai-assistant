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
    /* ===== 强制深色主题 ===== */
    html, body, .stApp {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
    }

    /* ===== 主容器 ===== */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 0;
        max-width: 1100px;
        background-color: #0F172A !important;
    }

    /* ===== 顶部标题栏 ===== */
    .app-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1.5rem 1.5rem 1rem 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.12);
        background: linear-gradient(135deg, rgba(99,102,241,0.10) 0%, rgba(165,180,252,0.04) 100%);
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    .app-header-icon {
        font-size: 2.8rem;
        line-height: 1;
    }
    .app-header-text h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366F1, #A5B4FC);
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
        margin-bottom: 1.2rem;
    }
    .stat-card {
        flex: 1;
        padding: 0.85rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.06);
        transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(99,102,241,0.15);
        border-color: rgba(99,102,241,0.25);
    }
    .stat-card .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.2;
    }
    .stat-card .stat-label {
        font-size: 0.72rem;
        color: #94A3B8;
        margin-top: 0.15rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-card.primary { border-left: 4px solid #6366F1; }
    .stat-card.success { border-left: 4px solid #10B981; }
    .stat-card.warning { border-left: 4px solid #F59E0B; }

    /* ===== 聊天消息 ===== */
    .stChatMessage {
        padding: 1rem 1.2rem !important;
        border-radius: 14px !important;
        margin-bottom: 0.8rem;
    }
    .stChatMessage[data-message-role="user"] {
        background: rgba(99,102,241,0.10) !important;
        border-left: 4px solid #6366F1 !important;
    }
    .stChatMessage[data-message-role="assistant"] {
        background: rgba(255,255,255,0.04) !important;
        border-left: 4px solid #6366F1 !important;
    }

    /* ===== 侧边栏 ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,23,42,0.98) 0%, rgba(30,41,59,0.96) 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] .stMarkdown h2 { font-size: 1.1rem !important; color: #F8FAFC !important; }
    [data-testid="stSidebar"] .stMarkdown h3 { font-size: 0.9rem !important; color: #94A3B8 !important; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 1rem; }
    [data-testid="stSidebar"] .stMarkdown p { color: #94A3B8 !important; }
    [data-testid="stSidebar"] .stMarkdown a { color: #A5B4FC !important; }
    [data-testid="stSidebar"] hr { border-top: 1px solid rgba(255,255,255,0.08) !important; margin: 0.8rem 0; }

    /* ===== 通用按钮 ===== */
    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        background: rgba(255,255,255,0.06) !important;
        color: #F8FAFC !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        border-color: rgba(99,102,241,0.4) !important;
        background: rgba(99,102,241,0.12) !important;
        box-shadow: 0 4px 12px rgba(99,102,241,0.15);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
        color: white !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
        box-shadow: 0 4px 16px rgba(99,102,241,0.3);
    }
    [data-testid="stSidebar"] .stButton > button {
        border-radius: 8px !important;
        background: rgba(255,255,255,0.06) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        font-size: 0.85rem;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(99,102,241,0.12) !important;
        border-color: rgba(99,102,241,0.4) !important;
    }

    /* ===== Selectbox / File Uploader ===== */
    .stSelectbox > div > div {
        border-radius: 10px !important;
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
    }
    [data-testid="stFileUploader"] {
        border-radius: 12px !important;
        background: rgba(255,255,255,0.06) !important;
    }

    /* ================================================================
       自定义输入栏 — 核心改造
       ================================================================ */
    .custom-input-bar {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.07) 0%, rgba(255,255,255,0.04) 100%);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 18px;
        padding: 0.55rem 0.7rem 0.55rem 1.1rem;
        backdrop-filter: blur(12px);
        transition: border-color 0.25s, box-shadow 0.25s;
        margin-top: 0.5rem;
    }
    .custom-input-bar:focus-within {
        border-color: rgba(99,102,241,0.55);
        box-shadow: 0 0 0 4px rgba(99,102,241,0.12), 0 4px 24px rgba(99,102,241,0.10);
    }
    .custom-input-bar input {
        flex: 1;
        background: transparent !important;
        border: none !important;
        outline: none !important;
        color: #F8FAFC !important;
        font-size: 0.95rem;
        padding: 0.3rem 0;
        min-width: 0;
        line-height: 1.5;
    }
    .custom-input-bar input::placeholder {
        color: #64748B !important;
        font-style: italic;
    }

    /* ===== 快捷按钮行 ===== */
    .action-row {
        display: flex;
        gap: 0.6rem;
        align-items: center;
        flex-wrap: wrap;
    }
    .action-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.45rem 0.9rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.05);
        font-size: 0.8rem;
        color: #CBD5E1;
        cursor: pointer;
        transition: all 0.2s;
        backdrop-filter: blur(8px);
        white-space: nowrap;
    }
    .action-chip:hover {
        border-color: rgba(99,102,241,0.45);
        background: rgba(99,102,241,0.12);
        color: #F8FAFC;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99,102,241,0.12);
    }
    .action-chip.active {
        border-color: #6366F1;
        background: rgba(99,102,241,0.20);
        color: #F8FAFC;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.12);
    }

    /* ===== 上传区域（内置到输入栏） ===== */
    .inline-upload [data-testid="stFileUploader"] {
        border-radius: 10px !important;
        background: transparent !important;
    }
    .inline-upload [data-testid="stFileUploader"] > section {
        border-radius: 10px !important;
        padding: 0.35rem 0.6rem !important;
        background: rgba(99,102,241,0.10) !important;
        border: 1px dashed rgba(99,102,241,0.35) !important;
        min-width: 100px;
    }
    .inline-upload [data-testid="stFileUploader"] button {
        border-radius: 8px !important;
        background: rgba(99,102,241,0.15) !important;
        border: none !important;
        color: #A5B4FC !important;
        font-size: 0.85rem;
        padding: 0.35rem 0.8rem;
    }
    .inline-upload [data-testid="stFileUploader"] button:hover {
        background: rgba(99,102,241,0.25) !important;
    }
    .inline-upload p {
        color: #CBD5E1 !important;
        font-size: 0.7rem;
    }

    /* ===== 功能面板（展开后显示在输入框上方） ===== */
    .function-panel {
        background: linear-gradient(135deg, rgba(30,41,59,0.92) 0%, rgba(30,41,59,0.80) 100%);
        border: 1px solid rgba(99,102,241,0.18);
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 0.6rem;
        backdrop-filter: blur(10px);
    }

    /* ===== 通用文本 / 输入颜色 ===== */
    .stMarkdown, .stMarkdown *, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #F8FAFC !important;
    }
    .stCaption { color: #94A3B8 !important; }

    /* ===== 强力覆盖 Streamlit 输入框颜色 ===== */
    /* 覆盖所有 input 元素，无论 Streamlit 怎么嵌套 */
    input, textarea,
    [data-testid="stTextInput"] input,
    [data-testid="stChatInput"] textarea,
    .stTextInput input,
    .stTextArea textarea,
    .stTextInput > div > div > input {
        color: #F8FAFC !important;
        caret-color: #A5B4FC !important;
        background-color: #1E293B !important;
        -webkit-text-fill-color: #F8FAFC !important;
    }
    input::placeholder, textarea::placeholder,
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748B !important;
        -webkit-text-fill-color: #64748B !important;
    }
    /* 确保聚焦时输入文字也可见 */
    input:focus, textarea:focus {
        color: #F8FAFC !important;
        -webkit-text-fill-color: #F8FAFC !important;
        background-color: #1E293B !important;
    }
    /* 覆盖 WebKit 自动填充 */
    input:-webkit-autofill,
    input:-webkit-autofill:focus {
        -webkit-text-fill-color: #F8FAFC !important;
        -webkit-box-shadow: 0 0 0 1000px #1E293B inset !important;
        transition: background-color 9999s ease-in-out 0s;
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 14px !important;
        background: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem;
    }

    /* ===== Toast / 提示 ===== */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        color: #F8FAFC !important;
    }
    [data-testid="stAlert"] * { color: #F8FAFC !important; }

    /* ===== 滚动条 ===== */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.10); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.18); }

    /* ===== 代码块 ===== */
    .stMarkdown code { background: rgba(255,255,255,0.06); border-radius: 6px; padding: 0.2rem 0.4rem; color: #F8FAFC; font-size: 0.9rem; }
    .stMarkdown pre { background: rgba(255,255,255,0.03); border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); color: #F8FAFC; }

    /* ===== 空状态 ===== */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #94A3B8;
    }
    .empty-state .empty-icon {
        font-size: 3.5rem;
        margin-bottom: 1.5rem;
    }
    .empty-state h3 {
        color: #F8FAFC;
        margin-bottom: 0.8rem;
        font-size: 1.4rem;
    }
    .empty-state p {
        font-size: 0.9rem;
        max-width: 450px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ===== 响应式 ===== */
    @media (max-width: 768px) {
        .stat-cards { flex-direction: column; }
        .app-header { flex-direction: column; text-align: center; }
        .action-row { gap: 0.4rem; }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State
# ============================================================
INIT_IMMUTABLE = {"initialized", "model_key", "agent", "doc_pipeline", "model"}

def init_session():
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.messages = []
        st.session_state.model_key = config.get("models", "models.default", "deepseek")
        st.session_state.agent = None
        st.session_state.doc_pipeline = None
        st.session_state.active_action = None  # 当前激活的快捷功能
        st.session_state.uploaded_file_path = None  # 上传文件的临时路径
        st.session_state.uploaded_file_name = None


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
    st.markdown("""
    <div style="text-align:center;">
        <div class="app-header" style="display:inline-flex;min-width:500px;">
            <div class="app-header-icon">🤖</div>
            <div class="app-header-text">
                <h1>AI Agent 智能办公助手</h1>
                <p>统一入口 — 知识问答 · 文档处理 · 消息通知</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.initialized:
        col1, col2, col3 = st.columns([5, 1, 2])
        with col3:
            info = st.session_state.model.get_model_info()
            kb_count = st.session_state.doc_pipeline.vector_store.count()
            st.markdown(f"""
            <div style="text-align:right;padding:0.5rem 0;color:#94A3B8;font-size:0.75rem;">
                <div style="color:#F8FAFC;font-weight:600;">{info['provider'].title()}</div>
                <div style="color:#CBD5E1;">{info['model']}</div>
                <div style="margin-top:0.2rem;color:#A5B4FC;">📚 {kb_count} 文档块</div>
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
# 执行快捷功能 — 输入文本后执行具体操作
# ============================================================
def execute_action(user_text: str):
    """根据当前激活的功能执行操作"""
    if not st.session_state.initialized:
        return

    action = st.session_state.get("active_action")

    if action == "search":
        # 知识库搜索
        with st.spinner("🔍 正在搜索知识库..."):
            results = st.session_state.doc_pipeline.search(user_text, top_k=5)

        if not results:
            response = "未找到相关文档。请先在侧边栏上传文档到知识库。"
        else:
            context = "\n\n---\n\n".join([
                f"[来源{i}: {doc.get('metadata', {}).get('filename', '未知')}]\n{doc.get('content', '')}"
                for i, doc in enumerate(results, 1)
            ])
            with st.spinner("🧠 正在整理回答..."):
                response = asyncio.run(st.session_state.model.chat(
                    messages=[
                        {"role": "system", "content": "你是一个知识库问答助手。请基于以下上下文回答用户问题，如果上下文中没有答案，请如实告知。回答要简洁清晰，使用中文。"},
                        {"role": "user", "content": f"问题：{user_text}\n\n上下文：\n{context}"}
                    ]
                ))
        st.session_state.messages.append({"role": "user", "content": f"🔍 搜索知识库: {user_text}"})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.active_action = None

    elif action == "notify":
        # 发送通知
        from src.agent.state import AgentState
        from src.agent.nodes.notification import NotificationNode

        notify_node = NotificationNode(st.session_state.model)
        platform_map = {"飞书": "feishu", "企业微信": "wecom", "钉钉": "dingtalk"}
        platform = st.session_state.get("notify_platform", "企业微信")
        platform_key = platform_map.get(platform, "wecom")

        with st.spinner(f"📨 正在发送通知到 {platform}..."):
            state = AgentState(
                user_input=user_text,
                notification_target=platform_key,
                notification_content=user_text,
                messages=[]
            )
            result = asyncio.run(notify_node(state))

        response = result.get("final_response", "通知处理完成")
        st.session_state.messages.append({"role": "user", "content": f"📨 发送通知到 {platform}"})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.active_action = None

    elif action == "resume":
        # 分析简历 — 需要上传文件
        file_path = st.session_state.get("uploaded_file_path")
        file_name = st.session_state.get("uploaded_file_name", "未知文件")

        if not file_path or not Path(file_path).exists():
            response = "请先上传简历文件（PDF 或 DOCX），再进行分析。"
            st.session_state.messages.append({"role": "user", "content": f"📄 分析简历: {user_text}"})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.active_action = None
            return

        from src.mcp.tools.document_parser import DocumentParserTool
        parser = DocumentParserTool()

        with st.spinner("📄 正在解析简历..."):
            content = asyncio.run(parser.execute(str(file_path)))

        system_prompt = """你是一个专业的简历分析顾问。请对以下简历进行全面分析：

## 📋 基本信息
- 姓名、学历、工作年限等

## 💪 核心优势
- 突出的技能和经验

## ⚠️ 潜在风险
- 不足或需要关注的地方

## 📊 能力评分
- 技术能力: X/10
- 项目经验: X/10
- 学习能力: X/10
- 沟通能力: X/10

## 🎯 综合评价
请用中文，客观公正。"""

        with st.spinner("🧠 正在分析简历..."):
            response = asyncio.run(st.session_state.model.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请分析以下简历（附加上下文：{user_text}）：\n\n{content[:50000]}"}
                ]
            ))

        st.session_state.messages.append({"role": "user", "content": f"📄 分析简历: {file_name}"})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.active_action = None
        st.session_state.uploaded_file_path = None
        st.session_state.uploaded_file_name = None

    elif action == "web_search":
        # 网络搜索
        from src.mcp.tools.web_search import WebSearchTool

        with st.spinner("🌐 正在搜索..."):
            search_tool = WebSearchTool()
            results_text = asyncio.run(search_tool.execute(user_text, max_results=5))

        with st.spinner("🧠 正在整理..."):
            response = asyncio.run(st.session_state.model.chat(
                messages=[
                    {"role": "system", "content": "你是一个信息整理助手。基于搜索结果，给出简洁清晰的答案总结，使用中文。"},
                    {"role": "user", "content": f"问题：{user_text}\n\n搜索结果：\n{results_text}"}
                ]
            ))

        full = f"**搜索结果：**\n{results_text}\n\n---\n**AI 整理：**\n{response}"
        st.session_state.messages.append({"role": "user", "content": f"🌐 网络搜索: {user_text}"})
        st.session_state.messages.append({"role": "assistant", "content": full})
        st.session_state.active_action = None

    else:
        # 普通对话
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = asyncio.run(process_message(user_text))
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        return  # 普通对话走自己的流程，不 rerun

    # 快捷操作完成后显示结果并 rerun
    with st.chat_message("user"):
        st.markdown(st.session_state.messages[-2]["content"])
    with st.chat_message("assistant"):
        st.markdown(st.session_state.messages[-1]["content"])
    st.rerun()


# ============================================================
# 功能面板（4个固定按钮 + 展开的操作面板）
# ============================================================
def render_action_bar():
    """在输入框上方渲染固定的功能按钮行"""
    active = st.session_state.get("active_action")

    # 平台选择预设（通知用）
    if "notify_platform" not in st.session_state:
        st.session_state.notify_platform = "企业微信"

    st.markdown('<div class="action-row" style="margin-bottom:0.4rem;">', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns([1.1, 1.1, 1.1, 1.1, 1])

    # --- 搜索知识库 ---
    active_class = " active" if active == "search" else ""
    with col1:
        if st.button("🔍 搜索知识库", key="btn_search", use_container_width=True):
            st.session_state.active_action = None if active == "search" else "search"
            st.rerun()
        if active == "search":
            st.markdown("""<style>#btn_search {border-color:#6366F1 !important; background:rgba(99,102,241,0.20) !important;}</style>""", unsafe_allow_html=True)

    # --- 发送通知 ---
    with col2:
        if st.button("📨 发送通知", key="btn_notify", use_container_width=True):
            st.session_state.active_action = None if active == "notify" else "notify"
            st.rerun()
        if active == "notify":
            st.markdown("""<style>#btn_notify {border-color:#6366F1 !important; background:rgba(99,102,241,0.20) !important;}</style>""", unsafe_allow_html=True)

    # --- 分析简历 ---
    with col3:
        if st.button("📄 分析简历", key="btn_resume", use_container_width=True):
            st.session_state.active_action = None if active == "resume" else "resume"
            st.rerun()
        if active == "resume":
            st.markdown("""<style>#btn_resume {border-color:#6366F1 !important; background:rgba(99,102,241,0.20) !important;}</style>""", unsafe_allow_html=True)

    # --- 网络搜索 ---
    with col4:
        if st.button("🌐 网络搜索", key="btn_web", use_container_width=True):
            st.session_state.active_action = None if active == "web_search" else "web_search"
            st.rerun()
        if active == "web_search":
            st.markdown("""<style>#btn_web {border-color:#6366F1 !important; background:rgba(99,102,241,0.20) !important;}</style>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- 展开的功能面板 ----------
    if active == "notify":
        st.markdown("""
        <div style="display:flex;align-items:center;gap:0.6rem;margin:0.3rem 0;padding:0.3rem 0.3rem;">
            <span style="font-size:0.85rem;color:#CBD5E1;white-space:nowrap;">📡 选择平台</span>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.notify_platform = st.radio(
            "平台", ["企业微信", "钉钉", "飞书"],
            horizontal=True, label_visibility="collapsed",
            index=["企业微信", "钉钉", "飞书"].index(st.session_state.notify_platform)
        )

    elif active == "resume":
        # 简历上传 — 内嵌在上方
        st.markdown("""
        <div style="display:flex;align-items:center;gap:0.6rem;margin:0.3rem 0;padding:0.3rem 0.3rem;">
            <span style="font-size:0.85rem;color:#CBD5E1;white-space:nowrap;">📎 请上传简历文件</span>
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "上传", type=["pdf", "docx"],
            key="inline_resume_upload",
            label_visibility="collapsed"
        )
        if uploaded:
            temp_dir = Path("data/uploads")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / uploaded.name
            temp_path.write_bytes(uploaded.getbuffer())
            st.session_state.uploaded_file_path = str(temp_path)
            st.session_state.uploaded_file_name = uploaded.name
            st.success(f"已选择: {uploaded.name}")


# ============================================================
# 自定义输入栏
# ============================================================
def render_input_bar():
    """渲染美观的输入栏（单行 TextInput + 发送按钮）"""
    active = st.session_state.get("active_action")

    action_label = {
        "search": "🔍 搜索知识库",
        "notify": "📨 发送通知",
        "resume": "📄 分析简历",
        "web_search": "🌐 网络搜索",
    }

    c_left, c_right = st.columns([10, 1.5])

    with c_left:
        placeholder = "输入您的问题，或点击上方按钮选择功能..."
        if active and active in action_label:
            placeholder = f"{action_label[active]} — 输入内容后点击发送"

        user_text = st.text_input(
            "",
            placeholder=placeholder,
            key="custom_input",
            label_visibility="collapsed"
        )

    with c_right:
        send_label = "↵ 执行" if active else "↵ 发送"
        send_clicked = st.button(
            send_label,
            key="btn_send",
            use_container_width=True,
            type="primary" if active else "secondary"
        )

    # 点击发送按钮时处理
    if send_clicked:
        if user_text:
            execute_action(user_text)
        else:
            st.toast("请输入内容", icon="⚠️")

    # 回车时也处理（text_input 按回车会触发 rerun，值会被保存）
    prev_input = st.session_state.get("_last_input", "")
    if user_text and user_text != prev_input:
        st.session_state._last_input = user_text
        # 回车触发：检测到输入变化且非发送按钮触发
        if not send_clicked:
            execute_action(user_text)


# ============================================================
# 空状态引导
# ============================================================
def render_empty_state():
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">👋</div>
        <h3>欢迎使用 AI 办公助手</h3>
        <p>我是您的智能办公伙伴，可以帮助您进行知识问答、文档处理、消息通知等任务。</p>
        <p style="margin-top:1rem;font-size:0.85rem;color:#94A3B8;">💡 点击上方按钮选择功能，或在输入框中直接提问</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 侧边栏
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:0.5rem 0 1rem 0;">
            <div style="font-size:2.5rem;">🤖</div>
            <div style="font-weight:700;font-size:1.1rem;color:#F8FAFC;">AI 办公助手</div>
            <div style="font-size:0.7rem;color:#94A3B8;">v0.1.1 · 2024</div>
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
            "上传文档到知识库",
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

        # ---- 工具 ----
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

    # 聊天区域
    if not st.session_state.messages:
        render_empty_state()
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ---- 底部输入区 ----
    # 添加底部间距
    st.markdown('<div class="chat-bottom-spacer" style="margin-top:1rem;"></div>', unsafe_allow_html=True)

    # 输入区容器
    with st.container():
        # 1. 功能按钮行（固定在输入框上方）
        render_action_bar()

        # 2. 自定义输入栏
        render_input_bar()

    # 底部留白
    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
