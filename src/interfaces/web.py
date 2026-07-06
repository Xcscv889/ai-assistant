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

    /* ===== 全局变量 ===== */
    :root {
        --primary: #6366F1;
        --primary-dark: #4F46E5;
        --primary-light: #A5B4FC;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --bg-primary: #0F172A;
        --bg-secondary: #1E293B;
        --bg-card: rgba(255,255,255,0.08);
        --bg-hover: rgba(255,255,255,0.12);
        --border: rgba(255,255,255,0.15);
        --text-primary: #F8FAFC;
        --text-secondary: #CBD5E1;
        --text-muted: #94A3B8;
    }

    /* ===== 主容器 ===== */
    .main .block-container {
        padding-top: 1.5rem;
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
        border-bottom: 1px solid var(--border);
        background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(165,180,252,0.05) 100%);
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    .app-header-icon {
        font-size: 2.8rem;
        line-height: 1;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3));
    }
    .app-header-text h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
        background: linear-gradient(135deg, #6366F1, #A5B4FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-header-text p {
        margin: 0.2rem 0 0 0;
        color: var(--text-muted);
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
        backdrop-filter: blur(10px);
        transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 32px rgba(99,102,241,0.2);
        border-color: rgba(99,102,241,0.3);
    }
    .stat-card .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }
    .stat-card .stat-label {
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-card.primary { border-left: 4px solid #6366F1; }
    .stat-card.success { border-left: 4px solid #10B981; }
    .stat-card.warning { border-left: 4px solid #F59E0B; }

    /* ===== 功能快捷入口 ===== */
    .quick-actions {
        display: flex;
        gap: 0.7rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    .quick-action-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.6rem 1.1rem;
        border-radius: 22px;
        border: 1px solid var(--border);
        background: var(--bg-card);
        font-size: 0.82rem;
        color: var(--text-primary);
        cursor: pointer;
        transition: all 0.2s;
        backdrop-filter: blur(8px);
    }
    .quick-action-chip:hover {
        border-color: rgba(99,102,241,0.5);
        background: rgba(99,102,241,0.15);
        color: var(--text-primary);
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(99,102,241,0.15);
    }
    .quick-action-chip:active {
        transform: translateY(0);
    }

    /* ===== 聊天消息美化 ===== */
    .stChatMessage {
        padding: 1rem 1.2rem !important;
        border-radius: 14px !important;
        margin-bottom: 0.8rem;
    }
    .stChatMessage[data-message-role="user"] {
        background: rgba(99,102,241,0.12) !important;
        border-left: 4px solid #6366F1 !important;
    }
    .stChatMessage[data-message-role="assistant"] {
        background: rgba(255,255,255,0.05) !important;
        border-left: 4px solid var(--primary) !important;
    }

    /* ===== 侧边栏美化 ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,23,42,0.98) 0%, rgba(30,41,59,0.96) 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
        padding-top: 1rem;
        backdrop-filter: blur(10px);
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 1.2rem !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.95rem !important;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 1.2rem;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: var(--text-muted) !important;
    }
    [data-testid="stSidebar"] .stMarkdown a {
        color: #A5B4FC !important;
    }

    /* ===== 按钮 ===== */
    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid var(--border) !important;
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        transition: all 0.2s !important;
        backdrop-filter: blur(8px);
    }
    .stButton > button:hover {
        border-color: rgba(99,102,241,0.5) !important;
        background: rgba(99,102,241,0.15) !important;
        box-shadow: 0 4px 16px rgba(99,102,241,0.2);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
        color: white !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.3);
    }

    /* ===== Selectbox / 文件上传 ===== */
    .stSelectbox > div > div {
        border-radius: 10px !important;
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
    }
    [data-testid="stFileUploader"] {
        border-radius: 12px !important;
        background: var(--bg-card) !important;
    }

    /* ===== 聊天输入 ===== */
    [data-testid="stChatInput"] textarea {
        border-radius: 14px !important;
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    [data-testid="stChatInput"] textarea:focus-within {
        border-color: rgba(99,102,241,0.5) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-muted) !important;
    }
    [data-testid="stChatInput"] textarea:hover {
        border-color: rgba(255,255,255,0.2) !important;
    }

    /* ===== 聊天消息文本 ===== */
    .stChatMessage .stMarkdown {
        color: var(--text-primary) !important;
        line-height: 1.6;
    }
    .stChatMessage .stMarkdown p {
        margin: 0.4rem 0;
    }
    .stChatMessage .stMarkdown a {
        color: #A5B4FC !important;
    }

    /* ===== 拖拽上传区域 ===== */
    [data-testid="stFileUploader"] .stMarkdown {
        color: var(--text-primary) !important;
    }

    /* ===== 侧边栏按钮 ===== */
    [data-testid="stSidebar"] .stButton > button {
        border-radius: 8px !important;
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        transition: all 0.15s !important;
        font-size: 0.85rem;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(99,102,241,0.15) !important;
        border-color: rgba(99,102,241,0.5) !important;
    }

    /* ===== 侧边栏分隔线 ===== */
    [data-testid="stSidebar"] hr {
        border-top: 1px solid var(--border) !important;
        margin: 1rem 0;
    }

    /* ===== Toast / 提示 ===== */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
    }

    /* ===== 滚动条 ===== */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.12);
        border-radius: 4px;
        transition: background 0.2s;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

    /* ===== 空状态 ===== */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: var(--text-muted);
    }
    .empty-state .empty-icon {
        font-size: 3.5rem;
        margin-bottom: 1.5rem;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2));
    }
    .empty-state h3 {
        color: var(--text-primary);
        margin-bottom: 0.8rem;
        font-size: 1.4rem;
    }
    .empty-state p {
        font-size: 0.9rem;
        max-width: 450px;
        margin: 0 auto;
        line-height: 1.6;
    }
    .empty-state p strong {
        color: var(--primary);
    }

    /* ===== 响应式 ===== */
    @media (max-width: 768px) {
        .stat-cards { flex-direction: column; }
        .app-header { flex-direction: column; text-align: center; }
    }

    /* ===== 代码块美化 ===== */
    .stMarkdown code {
        background: rgba(255,255,255,0.08);
        border-radius: 6px;
        padding: 0.2rem 0.4rem;
        color: var(--text-primary);
        font-size: 0.9rem;
    }
    .stMarkdown pre {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        border: 1px solid var(--border);
        color: var(--text-primary);
    }

    /* ===== Streamlit 默认元素覆盖 ===== */
    .stMarkdown, .stMarkdown *, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: var(--text-primary) !important;
    }
    .stText, .stText * {
        color: var(--text-primary) !important;
    }
    .stCaption {
        color: var(--text-muted) !important;
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
    # 居中显示标题
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

    # 模型信息显示在右侧
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
# 快捷入口
# ============================================================
def render_quick_actions():
    # 初始化快捷操作状态
    if "quick_action" not in st.session_state:
        st.session_state.quick_action = None

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    with col1:
        if st.button("💡 总结文档", use_container_width=True, key="qa_summarize"):
            st.session_state.quick_action = "summarize"
            st.rerun()
    with col2:
        if st.button("🔍 搜索知识库", use_container_width=True, key="qa_search"):
            st.session_state.quick_action = "search"
            st.rerun()
    with col3:
        if st.button("📨 发送通知", use_container_width=True, key="qa_notify"):
            st.session_state.quick_action = "notify"
            st.rerun()
    with col4:
        if st.button("📄 分析简历", use_container_width=True, key="qa_resume"):
            st.session_state.quick_action = "resume"
            st.rerun()
    with col5:
        if st.button("🌐 网络搜索", use_container_width=True, key="qa_search_web"):
            st.session_state.quick_action = "web_search"
            st.rerun()

    # 根据快捷操作显示相应的功能区域
    if st.session_state.quick_action:
        st.divider()

        if st.session_state.quick_action == "summarize":
            st.markdown("### 📄 文档总结")
            uploaded = st.file_uploader(
                "上传要总结的文档",
                type=["pdf", "docx", "pptx", "xlsx", "md", "txt"],
                key="summarize_upload"
            )
            if uploaded and st.button("📋 生成摘要", type="primary"):
                # 保存文件到临时目录
                temp_dir = Path("data/uploads")
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_path = temp_dir / uploaded.name
                temp_path.write_bytes(uploaded.getbuffer())

                # 解析文档内容
                from src.mcp.tools.document_parser import DocumentParserTool
                parser = DocumentParserTool()
                with st.spinner("正在解析文档..."):
                    content = asyncio.run(parser.execute(str(temp_path)))

                # 使用 LLM 生成摘要
                with st.spinner("正在生成摘要..."):
                    system_prompt = """你是一个专业的文档摘要助手。请对以下文档内容生成一份结构化的摘要，包括：

1. **核心观点** - 文档的主要论点和结论
2. **关键信息** - 重要的数据、事实和细节
3. **结构概览** - 文档的组织结构和主要内容块
4. **总结建议** - 基于文档内容的建议或行动项

请用中文回答，格式清晰。"""
                    response = asyncio.run(st.session_state.model.chat(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"请总结以下文档：\n\n{content[:50000]}"}
                        ]
                    ))

                # 显示结果
                st.success("✅ 摘要生成完成！")
                st.markdown(response)

                # 添加到对话历史
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"📄 总结文档: {uploaded.name}"
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

        elif st.session_state.quick_action == "search":
            st.markdown("### 🔍 知识库搜索")
            query = st.text_input("搜索关键词", key="kb_search_query")
            if st.button("搜索", type="primary") and query:
                # 执行知识库搜索
                results = st.session_state.doc_pipeline.search(query, top_k=5)

                if not results:
                    st.warning("未找到相关文档。请先上传文档到知识库。")
                else:
                    st.success(f"找到 {len(results)} 条相关结果")

                    # 显示结果
                    for i, doc in enumerate(results, 1):
                        with st.expander(f"{i}. {doc.get('metadata', {}).get('filename', '未知来源')} (相关度: {doc.get('score', 0):.2f})"):
                            st.markdown(doc.get('content', ''))

                    # 使用 LLM 整理回答
                    with st.spinner("正在整理回答..."):
                        context = "\n\n---\n\n".join([
                            f"[来源{i}: {doc.get('metadata', {}).get('filename', '未知')}]\n{doc.get('content', '')}"
                            for i, doc in enumerate(results, 1)
                        ])
                        response = asyncio.run(st.session_state.model.chat(
                            messages=[
                                {"role": "system", "content": "你是一个知识库问答助手。请基于以下上下文回答用户问题，如果上下文中没有答案，请如实告知。"},
                                {"role": "user", "content": f"问题：{query}\n\n上下文：\n{context}"}
                            ]
                        ))
                        st.markdown("---")
                        st.markdown("### 📖 AI 整理回答")
                        st.markdown(response)

                    # 添加到对话历史
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"🔍 搜索知识库: {query}"
                    })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })

        elif st.session_state.quick_action == "notify":
            st.markdown("### 📨 发送通知")
            platform_map = {"飞书": "feishu", "企业微信": "wecom", "钉钉": "dingtalk"}
            platform = st.selectbox("选择平台", ["飞书", "企业微信", "钉钉"])
            content = st.text_area("通知内容", key="notify_content", height=150)
            if st.button("发送", type="primary") and content:
                from src.agent.nodes.notification import NotificationNode
                from src.agent.state import AgentState

                notify_node = NotificationNode(st.session_state.model)

                # 构建状态
                state = AgentState(
                    user_input=content,
                    notification_target=platform_map[platform],
                    notification_content=content,
                    messages=[]
                )

                with st.spinner("正在发送通知..."):
                    result = asyncio.run(notify_node(state))

                # 显示结果
                st.markdown(result.get("final_response", "通知处理完成"))

                # 添加到对话历史
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"📨 发送通知到 {platform}"
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.get("final_response", "")
                })

        elif st.session_state.quick_action == "resume":
            st.markdown("### 📄 简历分析")
            uploaded = st.file_uploader(
                "上传简历",
                type=["pdf", "docx"],
                key="resume_upload"
            )
            if uploaded and st.button("🔬 开始分析", type="primary"):
                # 保存文件
                temp_dir = Path("data/uploads")
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_path = temp_dir / uploaded.name
                temp_path.write_bytes(uploaded.getbuffer())

                # 解析文档
                from src.mcp.tools.document_parser import DocumentParserTool
                parser = DocumentParserTool()
                with st.spinner("正在解析简历..."):
                    content = asyncio.run(parser.execute(str(temp_path)))

                # 使用 LLM 分析简历
                with st.spinner("正在分析简历..."):
                    system_prompt = """你是一个专业的简历分析顾问。请对以下简历进行全面分析，输出格式如下：

## 📋 基本信息
- 姓名、学历、工作年限等基础信息

## 💪 核心优势
- 候选人的突出技能和经验
- 与职位匹配的关键能力

## ⚠️ 潜在风险
- 可能存在的不足或需要关注的地方
- 职业发展中的潜在问题

## 📊 能力评分
- 技术能力: X/10
- 项目经验: X/10
- 学习能力: X/10
- 沟通能力: X/10

## 🎯 综合评价
- 整体评估和建议
- 是否推荐的结论

请用中文回答，客观公正。"""
                    response = asyncio.run(st.session_state.model.chat(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"请分析以下简历：\n\n{content[:50000]}"}
                        ]
                    ))

                # 显示结果
                st.success("✅ 简历分析完成！")
                st.markdown(response)

                # 添加到对话历史
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"📄 分析简历: {uploaded.name}"
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

        elif st.session_state.quick_action == "web_search":
            st.markdown("### 🌐 网络搜索")
            query = st.text_input("搜索关键词", key="web_search_query")
            if st.button("搜索", type="primary") and query:
                from src.mcp.tools.web_search import WebSearchTool

                with st.spinner("正在搜索..."):
                    search_tool = WebSearchTool()
                    results_text = asyncio.run(search_tool.execute(query, max_results=5))

                    # 显示搜索结果
                    st.success("✅ 搜索完成！")
                    st.markdown(results_text)

                    # 用 LLM 整理搜索结果
                    with st.spinner("正在整理..."):
                        response = asyncio.run(st.session_state.model.chat(
                            messages=[
                                {"role": "system", "content": "你是一个信息整理助手。请基于以下搜索结果，为用户提供清晰的答案总结。"},
                                {"role": "user", "content": f"问题：{query}\n\n搜索结果：\n{results_text}"}
                            ]
                        ))

                    st.markdown("---")
                    st.markdown("### 📖 AI 整理总结")
                    st.markdown(response)

                    # 添加到对话历史
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"🌐 网络搜索: {query}"
                    })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"搜索结果：\n{results_text}\n\nAI 整理：\n{response}"
                    })

        # 关闭按钮
        if st.button("✖️ 关闭"):
            st.session_state.quick_action = None
            st.rerun()


# ============================================================
# 空状态引导
# ============================================================
def render_empty_state():
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">👋</div>
        <h3>欢迎使用 AI 办公助手</h3>
        <p>我是您的智能办公伙伴，可以帮助您进行知识问答、文档处理、消息通知等任务。</p>
        <p style="margin-top:1rem;font-size:0.85rem;color:#94A3B8;">💡 试试输入「帮我总结一下上周的会议纪要」或「发通知提醒大家交周报」</p>
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
    st.markdown("""
    <style>
    [data-testid="stChatInput"] textarea::placeholder {
        color: #94A3B8 !important;
        font-style: italic;
    }
    /* 强制聊天输入框文字颜色 */
    [data-testid="stChatInput"] textarea {
        color: #F8FAFC !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        color: #F8FAFC !important;
    }
    </style>
    """, unsafe_allow_html=True)

    user_input = st.chat_input("输入您的问题，或使用命令...", key="chat_input")

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
