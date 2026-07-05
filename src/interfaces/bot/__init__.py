"""消息机器人模块"""

from .base import BaseBot
from .feishu import FeishuBot
from .wecom import WecomBot
from .dingtalk import DingtalkBot

__all__ = [
    "BaseBot",
    "FeishuBot",
    "WecomBot",
    "DingtalkBot",
]
