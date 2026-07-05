"""知识问答节点 - 检索增强生成 (RAG)"""

from typing import Any, Dict, List

from loguru import logger

from ...knowledge.pipeline import DocumentPipeline
from ...models.base import BaseModelAdapter
from ..prompts import prompts
from ..state import AgentState


class KnowledgeQANode:
    """知识问答节点

    流程：
    1. 构建检索查询
    2. 在知识库中搜索相关文档
    3. 使用检索到的上下文生成回答
    """

    def __init__(
        self,
        model_adapter: BaseModelAdapter,
        document_pipeline: DocumentPipeline,
    ):
        self.model = model_adapter
        self.doc_pipeline = document_pipeline

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """执行知识问答"""
        user_input = state.get("user_input", "")
        messages = state.get("messages", [])
        top_k = state.get("retrieval_query")  # 可以用 retrieval_query 覆盖

        if not user_input:
            return {"final_response": "请输入您的问题。"}

        # Step 1: 检索相关知识
        logger.info(f"正在检索知识库: {user_input[:100]}...")
        retrieved = self.doc_pipeline.search(user_input)
        state["retrieved_docs"] = retrieved

        # Step 2: 构建上下文
        if retrieved:
            context_parts = []
            for i, doc in enumerate(retrieved, 1):
                source = doc.get("metadata", {}).get("filename", "未知来源")
                content = doc.get("content", "")
                context_parts.append(f"[来源{i}: {source}]\n{content}")
            context = "\n\n---\n\n".join(context_parts)
        else:
            context = "（知识库中暂无相关文档，请根据你的知识回答）"

        # Step 3: 构建 Prompt 并生成回答
        system_prompt = prompts.render_system(
            "qa",
            context=context,
            history=self._format_history(messages),
        )

        try:
            response = await self.model.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
            )

            # 追加引用信息
            if retrieved:
                sources = self._format_sources(retrieved)
                response += sources

            return {
                "context": context,
                "final_response": response,
            }

        except Exception as e:
            logger.error(f"知识问答失败: {e}")
            return {"error": str(e), "final_response": f"抱歉，处理您的问题时出现了错误：{e}"}

    def _format_history(self, messages: List[Dict[str, str]], max_turns: int = 5) -> str:
        """格式化对话历史（最近 N 轮）"""
        if not messages:
            return "（无历史对话）"

        recent = messages[-max_turns * 2:]  # 每轮包含 user + assistant
        lines = []
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}: {msg['content'][:200]}")
        return "\n".join(lines)

    def _format_sources(self, docs: List[Dict[str, Any]]) -> str:
        """格式化引用来源"""
        lines = ["\n\n---", "📚 **参考来源：**"]
        seen = set()
        for i, doc in enumerate(docs, 1):
            source = doc.get("metadata", {}).get("filename", "未知来源")
            if source not in seen:
                score = doc.get("score", 0)
                lines.append(f"{len(seen)+1}. {source} (相关度: {score:.2f})")
                seen.add(source)
        return "\n".join(lines)
