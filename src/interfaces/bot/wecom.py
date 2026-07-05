"""企业微信机器人适配器"""

import os
from typing import Any, Dict, Optional

import httpx
from loguru import logger

from .base import BaseBot


class WecomBot(BaseBot):
    """企业微信 Webhook 机器人"""

    def __init__(self):
        super().__init__("wecom")
        self.webhook_url = os.getenv("WECOM_WEBHOOK_URL", "")

    async def initialize(self) -> None:
        """初始化"""
        if not self.webhook_url:
            logger.warning("企业微信未配置: 缺少 WECOM_WEBHOOK_URL")
            self.initialized = False
        else:
            self.initialized = True
            logger.info("企业微信机器人已配置")

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
                    logger.debug("企业微信消息已发送")
                    return True
                else:
                    logger.error(f"企业微信发送失败: {resp.status_code} {resp.text}")
                    return False
        except Exception as e:
            logger.error(f"企业微信发送异常: {e}")
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
                        "markdown": {"content": content},
                    },
                    timeout=10,
                )
                if resp.status_code == 200:
                    logger.debug("企业微信 Markdown 消息已发送")
                    return True
                else:
                    logger.error(f"企业微信发送失败: {resp.status_code}")
                    return False
        except Exception as e:
            logger.error(f"企业微信发送异常: {e}")
            return False

    async def send_card(self, card_data: Dict[str, Any], chat_id: Optional[str] = None) -> bool:
        """发送卡片消息（企业微信模板卡片）"""
        if not self.initialized:
            return False

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.webhook_url,
                    json={
                        "msgtype": "template_card",
                        "template_card": card_data,
                    },
                    timeout=10,
                )
                return resp.status_code == 200
        except Exception as e:
            logger.error(f"企业微信卡片发送异常: {e}")
            return False

    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """企业微信机器人为 Webhook 模式，不支持接收消息"""
        return None
