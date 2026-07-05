"""Embedding 服务 - 多后端支持，自动降级

降级链路：OpenAI API → 本地模型 → 哈希回退
DeepSeek 当前不支持 Embedding API (HTTP 404)，已确认跳过。
"""

import hashlib
import os
from typing import List

from loguru import logger

from ..utils.config import config


class EmbeddingService:
    """文本向量化服务

    优先级：
    1. DeepSeek Embedding API (deepseek-embed, 1536维)
    2. OpenAI Embedding API (text-embedding-3-small)
    3. sentence-transformers 本地模型
    4. 简易哈希回退（零依赖）
    """

    def __init__(self, provider: str = "auto"):
        self.provider = provider
        self._model = None
        self._client = None

    # ============================================================
    # 公开接口
    # ============================================================

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.provider == "auto":
            return self._embed_auto(texts)
        elif self.provider == "openai":
            return self._embed_openai(texts)
        elif self.provider == "local":
            return self._embed_local(texts)
        elif self.provider == "fallback":
            return self._embed_fallback(texts)
        else:
            raise ValueError(f"不支持的 provider: {self.provider}")

    def embed_query(self, query: str) -> List[float]:
        results = self.embed([query])
        return results[0]

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return self.embed(documents)

    # ============================================================
    # 自动选择（带异常回退）
    # ============================================================

    def _embed_auto(self, texts: List[str]) -> List[List[float]]:
        # 1. OpenAI Embedding
        if self._check_openai_available():
            try:
                logger.info("使用 OpenAI Embedding API")
                return self._embed_openai(texts)
            except Exception as e:
                logger.warning(f"OpenAI Embedding 失败，尝试下一个后端: {e}")

        # 2. 本地模型
        if self._check_local_available():
            try:
                logger.info("使用本地模型 Embedding")
                return self._embed_local(texts)
            except Exception as e:
                logger.warning(f"本地模型 Embedding 失败，使用回退: {e}")

        # 3. 哈希回退
        logger.warning("无可用的 Embedding 后端，使用简易哈希回退（检索精度会降低）")
        return self._embed_fallback(texts)

    # ============================================================
    # OpenAI Embedding
    # ============================================================

    def _check_openai_available(self) -> bool:
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        return bool(openai_key and openai_key not in ("sk-xxxxx", "sk-placeholder"))

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        try:
            from openai import OpenAI

            if self._client is None:
                api_key = os.getenv("OPENAI_API_KEY", "")
                base_url = os.getenv("OPENAI_BASE_URL", None)
                kwargs = {"api_key": api_key}
                if base_url:
                    kwargs["base_url"] = base_url
                self._client = OpenAI(**kwargs)

            resp = self._client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            return [d.embedding for d in resp.data]
        except Exception as e:
            logger.error(f"OpenAI Embedding 失败: {e}")
            raise

    # ============================================================
    # 本地 sentence-transformers
    # ============================================================

    def _check_local_available(self) -> bool:
        try:
            import torch  # noqa: F401
            from sentence_transformers import SentenceTransformer  # noqa: F401
            return True
        except (ImportError, OSError):
            return False

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            model_name = config.get("models", "embeddings.local.model", "BAAI/bge-large-zh-v1.5")
            device = config.get("models", "embeddings.local.device", "cpu")
            logger.info(f"加载本地 Embedding 模型: {model_name}")
            self._model = SentenceTransformer(model_name, device=device)

        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    # ============================================================
    # 简易哈希回退
    # ============================================================

    def _embed_fallback(self, texts: List[str]) -> List[List[float]]:
        return [self._text_to_sparse_vector(t) for t in texts]

    def _text_to_sparse_vector(self, text: str, dim: int = 384) -> List[float]:
        """字符 n-gram hash 向量"""
        vec = [0.0] * dim
        for n in range(1, 4):
            for i in range(len(text) - n + 1):
                ngram = text[i : i + n]
                h = hashlib.md5(ngram.encode("utf-8", errors="ignore")).digest()
                idx = int.from_bytes(h[:2], "little") % dim
                val = (int.from_bytes(h[2:4], "little") / 65535.0) * 2 - 1
                vec[idx] += val * (1.0 / n)
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
