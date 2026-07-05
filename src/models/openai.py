"""OpenAI API 适配器"""

import os
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from loguru import logger

from .base import BaseModelAdapter


class OpenAIAdapter(BaseModelAdapter):
    """OpenAI GPT 模型适配器（兼容所有 OpenAI API 格式的服务）"""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.api_key = os.getenv(config.get("api_key_env", "OPENAI_API_KEY"), "")
        self.base_url = config.get("base_url")

        self._client = None
        self._async_client = None

    @property
    def async_client(self):
        """延迟初始化异步客户端"""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI

                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._async_client = AsyncOpenAI(**kwargs)
            except ImportError:
                raise ImportError("请安装 openai 包: pip install openai")
        return self._async_client

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """发送对话请求到 OpenAI"""
        temp = temperature if temperature is not None else self.config.get("temperature", 0.7)
        max_tok = max_tokens if max_tokens is not None else self.config.get("max_tokens", 8192)

        logger.debug(f"OpenAI 请求: model={self.model_name}, messages={len(messages)}")

        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore
            max_tokens=max_tok,
            temperature=temp,
            timeout=httpx.Timeout(60.0, connect=10.0),
            **kwargs,
        )

        return response.choices[0].message.content or ""

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

        stream = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore
            max_tokens=max_tok,
            temperature=temp,
            timeout=httpx.Timeout(120.0, connect=10.0),
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """使用 OpenAI 生成 Embedding"""
        response = await self.async_client.embeddings.create(
            model=self.config.get("embedding_model", "text-embedding-3-small"),
            input=texts,
        )
        return [item.embedding for item in response.data]
