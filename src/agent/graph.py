"""LangGraph Agent 工作流定义

工作流结构：
    START → Router → [KnowledgeQA | DocumentProcessing | Notification | Generation] → END

Router 根据意图将请求分发到不同的处理节点。
"""

from typing import Any, Dict, Literal

from langgraph.graph import StateGraph, END
from loguru import logger

from ..knowledge.pipeline import DocumentPipeline
from ..models.base import BaseModelAdapter
from ..utils.config import config
from .nodes.document import DocumentProcessingNode
from .nodes.generation import GenerationNode
from .nodes.knowledge_qa import KnowledgeQANode
from .nodes.notification import NotificationNode
from .nodes.router import RouterNode
from .state import AgentState


class AgentGraph:
    """AI 办公助手 Agent 工作流"""

    def __init__(
        self,
        model_adapter: BaseModelAdapter,
        document_pipeline: DocumentPipeline,
    ):
        self.model = model_adapter
        self.doc_pipeline = document_pipeline

        # 初始化节点
        self.router_node = RouterNode(model_adapter)
        self.qa_node = KnowledgeQANode(model_adapter, document_pipeline)
        self.document_node = DocumentProcessingNode(model_adapter, document_pipeline)
        self.notification_node = NotificationNode(model_adapter)
        self.generation_node = GenerationNode(model_adapter)

        # 构建图
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("router", self.router_node)
        workflow.add_node("knowledge_qa", self.qa_node)
        workflow.add_node("document_processing", self.document_node)
        workflow.add_node("notification", self.notification_node)
        workflow.add_node("generation", self.generation_node)

        # 设置入口点
        workflow.set_entry_point("router")

        # 添加条件分支：Router → 各任务节点
        workflow.add_conditional_edges(
            "router",
            self._route_by_intent,
            {
                "knowledge_qa": "knowledge_qa",
                "document_processing": "document_processing",
                "notification": "notification",
                "generation": "generation",
            },
        )

        # 所有任务节点 → END
        workflow.add_edge("knowledge_qa", END)
        workflow.add_edge("document_processing", END)
        workflow.add_edge("notification", END)
        workflow.add_edge("generation", END)

        return workflow.compile()

    def _route_by_intent(self, state: AgentState) -> str:
        """根据意图返回下一个节点名称"""
        intent = state.get("intent", "chat")
        confidence = state.get("confidence", 0.0)

        logger.info(f"路由: intent={intent}, confidence={confidence:.2f}")

        if intent == "qa":
            return "knowledge_qa"
        elif intent == "document":
            return "document_processing"
        elif intent == "notify":
            return "notification"
        else:
            return "generation"

    async def run(self, user_input: str, messages: list = None, **kwargs) -> AgentState:
        """运行 Agent 工作流

        Args:
            user_input: 用户输入文本
            messages: 历史消息列表
            **kwargs: 额外状态参数

        Returns:
            更新后的 AgentState
        """
        initial_state: AgentState = {
            "messages": messages or [],
            "user_input": user_input,
            "intent": "chat",
            "confidence": 0.0,
            "retrieved_docs": [],
            "uploaded_files": kwargs.get("uploaded_files", []),
            "task_type": kwargs.get("task_type", ""),
            "notification_target": kwargs.get("notification_target", ""),
            "model_name": self.model.model_name,
            "final_response": "",
            "tool_calls": [],
            "error": None,
            "iteration_count": 0,
        }

        logger.info(f"Agent 开始处理: {user_input[:100]}...")

        try:
            result = await self.graph.ainvoke(initial_state)
            logger.info("Agent 处理完成")
            return result
        except Exception as e:
            logger.error(f"Agent 运行失败: {e}")
            return {
                **initial_state,
                "error": str(e),
                "final_response": f"抱歉，系统处理您的请求时出现了错误：{e}",
            }

    async def stream(self, user_input: str, messages: list = None, **kwargs):
        """流式运行 Agent（返回事件流）"""
        initial_state: AgentState = {
            "messages": messages or [],
            "user_input": user_input,
            "intent": "chat",
            "confidence": 0.0,
            "retrieved_docs": [],
            "uploaded_files": kwargs.get("uploaded_files", []),
            "task_type": kwargs.get("task_type", ""),
            "notification_target": kwargs.get("notification_target", ""),
            "model_name": self.model.model_name,
            "final_response": "",
            "tool_calls": [],
            "error": None,
            "iteration_count": 0,
        }

        async for event in self.graph.astream(initial_state):
            yield event
