"""消息机器人基类"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseBot(ABC):
    """消息机器人抽象基类"""

    def __init__(self, platform: str):
        self.platform = platform
        self.initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """初始化机器人连接"""
        ...

    @abstractmethod
    async def send_text(self, content: str, chat_id: Optional[str] = None) -> bool:
        """发送文本消息"""
        ...

    @abstractmethod
    async def send_markdown(self, content: str, chat_id: Optional[str] = None) -> bool:
        """发送 Markdown 消息"""
        ...

    @abstractmethod
    async def send_card(self, card_data: Dict[str, Any], chat_id: Optional[str] = None) -> bool:
        """发送卡片消息"""
        ...

    @abstractmethod
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """接收消息（用于 Webhook / 轮询）"""
        ...

    def is_configured(self) -> bool:
        """检查机器人是否已配置"""
        return self.initialized
