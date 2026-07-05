"""钉钉机器人适配器"""

import os
from typing import Any, Dict, Optional

import httpx
from loguru import logger

from .base import BaseBot


class DingtalkBot(BaseBot):
    """钉钉 Webhook 机器人"""

    def __init__(self):
        super().__init__("dingtalk")
        self.webhook_url = os.getenv("DINGTALK_WEBHOOK_URL", "")

    async def initialize(self) -> None:
        """初始化"""
        if not self.webhook_url:
            logger.warning("钉钉未配置: 缺少 DINGTALK_WEBHOOK_URL")
            self.initialized = False
        else:
            self.initialized = True
            logger.info("钉钉机器人已配置")

    async def send_text(self, content: str, chat_id: Optional[str] = None) -> bool:
        """发送文本消息"""
        if not self.initialized:
            return False

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.webhook_url,
                    json={
                        "msgtype": "text",
                        "text": {"content": content},
                    },
                    timeout=10,
                )
                if resp.status_code == 200:
                    logger.debug("钉钉消息已发送")
                    return True
                else:
                    logger.error(f"钉钉发送失败: {resp.status_code}")
                    return False
        except Exception as e:
            logger.error(f"钉钉发送异常: {e}")
            return False

    async def send_markdown(self, content: str, chat_id: Optional[str] = None) -> bool:
        """发送 Markdown 消息"""
        if not self.initialized:
            return False

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.webhook_url,
                    json={
                        "msgtype": "markdown",
                        "markdown": {
                            "title": "AI 助手通知",
                            "text": content,
                        },
                    },
                    timeout=10,
                )
                if resp.status_code == 200:
                    logger.debug("钉钉 Markdown 消息已发送")
                    return True
                else:
                    logger.error(f"钉钉发送失败: {resp.status_code}")
                    return False
        except Exception as e:
            logger.error(f"钉钉发送异常: {e}")
            return False

    async def send_card(self, card_data: Dict[str, Any], chat_id: Optional[str] = None) -> bool:
        """发送卡片消息（钉钉 ActionCard）"""
        if not self.initialized:
            return False

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.webhook_url,
                    json={
                        "msgtype": "actionCard",
                        "actionCard": card_data,
                    },
                    timeout=10,
                )
                return resp.status_code == 200
        except Exception as e:
            logger.error(f"钉钉卡片发送异常: {e}")
            return False

    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """钉钉 Webhook 模式不支持接收消息"""
        return None
