"""Ollama 本地模型适配器"""

import os
from typing import Any, AsyncIterator, Dict, List, Optional

from loguru import logger

from .base import BaseModelAdapter


class OllamaAdapter(BaseModelAdapter):
    """Ollama 本地大模型适配器"""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self._client = None

    @property
    def client(self):
        """延迟初始化 Ollama 客户端"""
        if self._client is None:
            try:
                import ollama

                self._client = ollama.AsyncClient(host=self.base_url)
            except ImportError:
                raise ImportError("请安装 ollama 包: pip install ollama")
        return self._client

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """发送对话请求到 Ollama"""
        temp = temperature if temperature is not None else self.config.get("temperature", 0.7)

        # Ollama 的 messages 格式与 OpenAI 相同
        options = {"temperature": temp}
        if max_tokens:
            options["num_predict"] = max_tokens

        logger.debug(f"Ollama 请求: model={self.model_name}, host={self.base_url}")

        try:
            response = await self.client.chat(
                model=self.model_name,
                messages=messages,
                options=options,
                **kwargs,
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama 请求失败: {e}")
            raise

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """流式对话"""
        temp = temperature if temperature is not None else self.config.get("temperature", 0.7)

        options = {"temperature": temp}
        if max_tokens:
            options["num_predict"] = max_tokens

        stream = await self.client.chat(
            model=self.model_name,
            messages=messages,
            stream=True,
            options=options,
            **kwargs,
        )

        async for chunk in stream:
            content = chunk["message"]["content"]
            if content:
                yield content

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """使用 Ollama 生成 Embedding"""
        results = []
        for text in texts:
            response = await self.client.embeddings(
                model=self.model_name,
                prompt=text,
            )
            results.append(response["embedding"])
        return results
