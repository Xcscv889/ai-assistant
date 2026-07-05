"""Claude API 适配器"""

import os
from typing import Any, AsyncIterator, Dict, List, Optional

from loguru import logger

from .base import BaseModelAdapter


class ClaudeAdapter(BaseModelAdapter):
    """Anthropic Claude 模型适配器"""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.api_key = os.getenv(config.get("api_key_env", "ANTHROPIC_API_KEY"), "")
        if not self.api_key:
            logger.warning("未设置 ANTHROPIC_API_KEY，Claude 适配器将无法使用")

        self._client = None

    @property
    def client(self):
        """延迟初始化 Anthropic 客户端"""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic

                self._client = AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("请安装 anthropic 包: pip install anthropic")
        return self._client

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """发送对话请求到 Claude"""
        temp = temperature if temperature is not None else self.config.get("temperature", 0.7)
        max_tok = max_tokens if max_tokens is not None else self.config.get("max_tokens", 8192)

        # 提取 system prompt
        system_prompt = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                chat_messages.append({"role": msg["role"], "content": msg["content"]})

        logger.debug(f"Claude 请求: model={self.model_name}, messages={len(chat_messages)}")

        response = await self.client.messages.create(
            model=self.model_name,
            system=system_prompt or "",
            messages=chat_messages,
            max_tokens=max_tok,
            temperature=temp,
            **kwargs,
        )

        return response.content[0].text

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """流式对话"""
        temp = temperature if temperature is not None else self.config.get("temperature", 0.7)
        max_tok = max_tokens if max_tokens is not None else self.config.get("max_tokens", 8192)

        system_prompt = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                chat_messages.append({"role": msg["role"], "content": msg["content"]})

        async with self.client.messages.stream(
            model=self.model_name,
            system=system_prompt or "",
            messages=chat_messages,
            max_tokens=max_tok,
            temperature=temp,
            **kwargs,
        ) as stream:
            async for text in stream.text_stream:
                yield text
