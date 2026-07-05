"""Router 节点 - 混合路由引擎，规则优先 + LLM 兜底"""

import re
from typing import Any, Dict

from loguru import logger

from ...models.base import BaseModelAdapter
from ..prompts import prompts
from ..state import AgentState


class RouterNode:
    """意图识别路由节点

    策略：规则优先（快）+ LLM 兜底（准）
    - 规则匹配命中 → 直接返回，零延迟
    - 规则未命中 → 调 LLM 做意图识别，保证准确率
    """

    def __init__(self, model_adapter=None):
        self.model = model_adapter

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        if not user_input:
            return {"intent": "chat", "confidence": 1.0}

        # === 第0层：简单对话直接识别，不调 LLM ===
        if self._is_chat(user_input):
            logger.info(f"快速匹配 -> chat")
            return {"intent": "chat", "confidence": 0.95}

        # === 第一层：规则匹配 ===
        # 通知类（明确提到发消息/通知）
        if self._is_notify(user_input):
            target = self._extract_notify_target(user_input)
            logger.info(f"规则匹配 -> notify (target={target})")
            return {"intent": "notify", "confidence": 0.95, "notification_target": target}

        # 文档处理类（明确提到文件操作）
        if self._is_document(user_input):
            task = self._extract_document_task(user_input)
            logger.info(f"规则匹配 -> document (task={task})")
            return {"intent": "document", "confidence": 0.95, "task_type": task}

        # 知识问答类（明确在问问题）
        if self._is_qa(user_input):
            logger.info(f"规则匹配 -> qa")
            return {"intent": "qa", "confidence": 0.85}

        # === 第二层：LLM 兜底 ===
        if self.model:
            try:
                system_prompt = prompts.get_system("router")
                response = await self.model.chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input},
                    ],
                    temperature=0.3,
                    max_tokens=128,
                )
                intent, confidence = self._parse_json(response)
                logger.info(f"LLM 路由: intent={intent}, confidence={confidence:.2f}")
                return {"intent": intent, "confidence": confidence}
            except Exception as e:
                logger.warning(f"LLM 路由失败，回退 chat: {e}")

        return {"intent": "chat", "confidence": 0.6}

    # ============================================================
    # 通知识别
    # ============================================================

    def _is_notify(self, text: str) -> bool:
        keywords = [
            "发一条", "发送通知", "推送消息", "发通知", "推送",
            "发条消息", "发条通知", "发个通知", "发个消息",
            "发到群里", "发到群", "群里发",
            "通知大家", "提醒大家", "提醒一下",
            "飞书发", "钉钉发", "企微发",
        ]
        return any(kw in text for kw in keywords)

    def _extract_notify_target(self, text: str) -> str:
        if "钉钉" in text:
            return "dingtalk"
        if "飞书" in text:
            return "feishu"
        if "企微" in text or "企业微信" in text:
            return "wecom"
        return ""

    # ============================================================
    # 文档识别
    # ============================================================

    def _is_document(self, text: str) -> bool:
        # 文件扩展名
        if re.search(r'\.(pdf|docx?|pptx?|xlsx?|md|txt|csv)\b', text, re.IGNORECASE):
            return True
        # 路径
        if re.search(r'[A-Za-z]:\\|/~?[\w/.-]+/', text):
            return True
        # 文档操作关键词
        keywords = [
            "总结", "摘要", "概括", "提取", "翻译",
            "分析", "处理", "解析", "上传",
            "简历", "论文", "文档", "文件",
        ]
        return any(kw in text for kw in keywords)

    def _extract_document_task(self, text: str) -> str:
        if any(kw in text for kw in ["总结", "摘要", "概括"]):
            return "summarize"
        if any(kw in text for kw in ["提取", "抽取"]):
            return "extract"
        if any(kw in text for kw in ["翻译", "translate"]):
            return "translate"
        if any(kw in text for kw in ["分析", "优缺点", "建议"]):
            return "analyze"
        return "summarize"

    # ============================================================
    # 知识问答识别
    # ============================================================

    def _is_qa(self, text: str) -> bool:
        # 问号
        if "?" in text or "？" in text:
            return True
        # 疑问词
        question_words = [
            "什么", "怎么", "如何", "为什么", "哪", "谁",
            "介绍", "说明", "解释", "定义", "是什么",
            "怎么样", "多少钱", "多少",
        ]
        if any(w in text for w in question_words):
            return True
        return False

    def _is_chat(self, text: str) -> bool:
        """判断是否只是普通对话（不触发 LLM 路由）"""
        chat_keywords = ["你好", "谢谢", "再见", "好的", "嗯", "哦", "嗨"]
        stripped = text.strip()
        if len(stripped) <= 3:
            return True
        if any(stripped == kw for kw in chat_keywords):
            return True
        return False

    # ============================================================
    # JSON 解析
    # ============================================================

    def _parse_json(self, response: str) -> tuple:
        valid = {"qa", "document", "notify", "chat"}
        import json
        try:
            m = re.search(r'\{[^{}]*"intent"[^{}]*\}', response)
            if m:
                data = json.loads(m.group())
                i = data.get("intent", "chat")
                c = float(data.get("confidence", 0.5))
                if i in valid:
                    return i, min(max(c, 0.0), 1.0)
        except Exception:
            pass
        return "chat", 0.5
