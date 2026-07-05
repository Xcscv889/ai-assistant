"""Router 节点 - 纯规则引擎，零延迟意图识别"""

import re
from typing import Any, Dict

from loguru import logger

from ..state import AgentState


class RouterNode:
    """意图识别路由节点 — 纯规则匹配，不触发 LLM 调用

    所有意图识别通过关键词和正则完成，延迟 < 1ms。
    避免了之前每次对话都要先走一轮 LLM 的额外延迟。
    """

    def __init__(self, model_adapter=None):
        self.model = model_adapter  # 保留参数兼容性，但不使用

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """执行意图识别（同步规则匹配，无需 LLM）"""
        user_input = state.get("user_input", "")

        if not user_input:
            return {"intent": "chat", "confidence": 1.0}

        # 快速规则匹配：文件路径检测
        if self._detect_file_reference(user_input):
            logger.info(f"规则匹配 -> document")
            return {"intent": "document", "confidence": 0.95}

        # 快速规则匹配：通知关键词
        if self._detect_notification_intent(user_input):
            logger.info(f"规则匹配 -> notify")
            return {"intent": "notify", "confidence": 0.9}

        # 知识问答关键词检测
        if self._detect_qa_intent(user_input):
            logger.info(f"规则匹配 -> qa")
            return {"intent": "qa", "confidence": 0.8}

        # 默认对话
        logger.info(f"规则匹配 -> chat")
        return {"intent": "chat", "confidence": 0.8}

    def _detect_file_reference(self, text: str) -> bool:
        """检测文件引用"""
        patterns = [
            r'\.pdf\b', r'\.docx?\b', r'\.pptx?\b', r'\.xlsx?\b',
            r'\.md\b', r'\.txt\b', r'\.csv\b',
            r'[CcDdEeFf]:\\',
            r'/~?[\w/.-]+/',
            r'上传', r'解析', r'读取.*文件',
            r'总结.*[文档论文文件]', r'分析.*[文档论文文件]',
            r'处理.*文件', r'翻译.*文档',
        ]
        return any(re.search(p, text) for p in patterns)

    def _detect_notification_intent(self, text: str) -> bool:
        """检测通知意图"""
        keywords = [
            r'发送.*通知', r'推送.*消息', r'提醒', r'通知.*群',
            r'发.*飞书', r'发.*企微', r'发.*钉钉', r'发.*消息',
            r'会议纪要.*发', r'报告.*发',
        ]
        return any(re.search(k, text) for k in keywords)

    def _detect_qa_intent(self, text: str) -> bool:
        """检测知识问答意图"""
        patterns = [
            r'[?？]',                     # 问号
            r'什么(是|叫|意思)', r'如何', r'怎么', r'为什么',
            r'介绍', r'说明', r'解释', r'定义',
            r'搜索', r'查找', r'查一下', r'帮我查',
            r'论文', r'文献', r'知识',
        ]
        return any(re.search(p, text) for p in patterns)
