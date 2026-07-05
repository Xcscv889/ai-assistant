"""生成节点 - 结果合成与格式化（通用对话节点）"""

from typing import Any, Dict

from loguru import logger

from ...models.base import BaseModelAdapter
from ..state import AgentState


class GenerationNode:
    """结果生成节点

    负责：
    1. 闲聊场景下的直接回复
    2. 各任务节点的结果后处理
    3. 统一格式化输出
    """

    def __init__(self, model_adapter: BaseModelAdapter):
        self.model = model_adapter

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """生成最终回复"""
        # 如果已经有 final_response，直接返回
        if state.get("final_response"):
            return {}

        user_input = state.get("user_input", "")
        messages = state.get("messages", [])

        if not user_input:
            return {"final_response": "您好！有什么可以帮助您的吗？"}

        # 闲聊模式：直接对话
        system_prompt = (
            "你是一个智能办公助手，可以帮助用户进行知识问答、文档处理、消息通知等任务。"
            "请用友好的语气回答用户的问题。"
            "如果用户的问题需要处理文件、搜索知识库或发送通知，请告知用户使用相应的命令。"
        )

        try:
            # 构建消息
            chat_messages = [{"role": "system", "content": system_prompt}]
            # 添加最近的历史（最多4轮）
            if messages:
                chat_messages.extend(messages[-8:])
            chat_messages.append({"role": "user", "content": user_input})

            response = await self.model.chat(messages=chat_messages)

            return {"final_response": response}

        except Exception as e:
            logger.error(f"生成回复失败: {e}")
            return {
                "error": str(e),
                "final_response": f"抱歉，处理您的请求时出现了错误：{e}",
            }
