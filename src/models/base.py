"""模型适配器基类"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseModelAdapter(ABC):
    """所有模型适配器的抽象基类"""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.provider = config.get("provider", "unknown")

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """发送对话请求并返回文本响应

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}, ...]
            temperature: 温度参数
            max_tokens: 最大输出 token 数
            **kwargs: 其他模型参数

        Returns:
            模型的文本响应
        """
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        """流式对话接口，逐块返回响应"""
        ...

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """生成文本 Embedding（可选实现）"""
        raise NotImplementedError(f"{self.provider} 不支持 Embedding")

    def get_model_info(self) -> Dict[str, Any]:
        """返回模型信息"""
        return {
            "provider": self.provider,
            "model": self.model_name,
            "config": {k: v for k, v in self.config.items() if "api_key" not in k.lower()},
        }
