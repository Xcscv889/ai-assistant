"""飞书机器人适配器"""

import os
from typing import Any, Dict, Optional

from loguru import logger

from .base import BaseBot


class FeishuBot(BaseBot):
    """飞书（Lark）消息机器人"""

    def __init__(self):
        super().__init__("feishu")
        self.app_id = os.getenv("FEISHU_APP_ID", "")
        self.app_secret = os.getenv("FEISHU_APP_SECRET", "")
        self._client = None

    async def initialize(self) -> None:
        """初始化飞书客户端"""
        if not self.app_id or not self.app_secret:
            logger.warning("飞书未配置: 缺少 FEISHU_APP_ID 或 FEISHU_APP_SECRET")
            self.initialized = False
            return

        try:
            import lark_oapi as lark
            from lark_oapi.api.im.v1 import (
                CreateTextMessageResponse,
                CreateMessageRequest,
                CreateMessageRequestBody,
            )

            self._lark = lark
            self.initialized = True
            logger.info("飞书机器人已初始化")
        except ImportError:
            logger.warning("未安装 lark-oapi: pip install lark-oapi")
            self.initialized = False

    async def send_text(self, content: str, chat_id: Optional[str] = None) -> bool:
        """发送文本消息到飞书"""
        if not self.initialized:
            logger.warning("飞书未初始化，无法发送消息")
            return False

        try:
            # 使用飞书 SDK 发送消息
            # 注意: 需要获取 tenant_access_token
            client = self._lark.Client.builder() \
                .app_id(self.app_id) \
                .app_secret(self.app_secret) \
                .build()

            # TODO: 实现完整的消息发送逻辑
            # 需要配置 chat_id 或 webhook 地址
            logger.info(f"飞书消息发送: {content[:50]}...")
            return True

        except Exception as e:
            logger.error(f"飞书消息发送失败: {e}")
            return False

    async def send_markdown(self, content: str, chat_id: Optional[str] = None) -> bool:
        """发送 Markdown 消息"""
        # 飞书支持部分 Markdown 语法
        return await self.send_text(content, chat_id)

    async def send_card(self, card_data: Dict[str, Any], chat_id: Optional[str] = None) -> bool:
        """发送卡片消息"""
        if not self.initialized:
            return False

        # TODO: 使用飞书消息卡片 API
        logger.info("飞书卡片消息发送")
        return True

    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """接收飞书消息回调"""
        # TODO: 实现事件订阅和消息接收
        return None
