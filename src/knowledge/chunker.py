"""文档分块策略"""

from typing import List, Optional

from loguru import logger

from ..utils.config import config


class DocumentChunker:
    """文档分块器，支持多种分块策略

    优先使用 langchain_text_splitters，如不可用则回退到内置分块。
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strategy: str = "recursive",
    ):
        self.chunk_size = chunk_size or config.get("settings", "knowledge_base.chunk_size", 1000)
        self.chunk_overlap = chunk_overlap or config.get(
            "settings", "knowledge_base.chunk_overlap", 200
        )
        self.strategy = strategy
        self._splitters = None  # 延迟初始化

    def _init_splitters(self):
        """延迟初始化分块器，避免 torch DLL 问题阻塞基础导入"""
        if self._splitters is not None:
            return

        try:
            from langchain_text_splitters import (
                RecursiveCharacterTextSplitter,
                MarkdownHeaderTextSplitter,
                TokenTextSplitter,
            )

            self._splitters = {
                "recursive": RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", "。", ".", " ", ""],
                    length_function=len,
                ),
                "token": TokenTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                ),
            }
            self._MarkdownHeaderTextSplitter = MarkdownHeaderTextSplitter
        except (ImportError, OSError) as e:
            logger.warning(f"langchain_text_splitters 不可用，将使用内置分块: {e}")
            self._splitters = None

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[dict]:
        """将文本分割为多个块"""
        if not text or not text.strip():
            return []

        meta = metadata or {}
        self._init_splitters()

        if self._splitters:
            splitter = self._splitters.get(self.strategy, self._splitters["recursive"])
            chunks = splitter.split_text(text)
        else:
            chunks = self._simple_chunk(text)

        results = []
        for i, chunk in enumerate(chunks):
            results.append(
                {
                    "content": chunk,
                    "metadata": {
                        **meta,
                        "chunk_index": i,
                        "chunk_count": len(chunks),
                    },
                }
            )

        logger.debug(f"文档分块完成: {len(text)} 字符 -> {len(results)} 个块")
        return results

    def _simple_chunk(self, text: str) -> List[str]:
        """简单递归分块（不依赖 langchain）"""
        separators = ["\n\n", "\n", "。", ".", " ", ""]
        chunks = [text]
        for sep in separators:
            new_chunks = []
            for chunk in chunks:
                if len(chunk) <= self.chunk_size:
                    new_chunks.append(chunk)
                else:
                    parts = chunk.split(sep)
                    current = ""
                    for part in parts:
                        piece = part + (sep if sep else "")
                        if len(current) + len(piece) <= self.chunk_size:
                            current += piece
                        else:
                            if current.strip():
                                new_chunks.append(current)
                            current = piece
                    if current.strip():
                        new_chunks.append(current)
            chunks = new_chunks
        return [c for c in chunks if c.strip()]

    def chunk_markdown(self, text: str, metadata: Optional[dict] = None) -> List[dict]:
        """按 Markdown 标题结构分块"""
        try:
            self._init_splitters()
            if self._splitters and hasattr(self, "_MarkdownHeaderTextSplitter"):
                md_splitter = self._MarkdownHeaderTextSplitter(
                    headers_to_split_on=[
                        ("#", "h1"),
                        ("##", "h2"),
                        ("###", "h3"),
                    ]
                )
                docs = md_splitter.split_text(text)
                results = []
                for i, doc in enumerate(docs):
                    results.append(
                        {
                            "content": doc.page_content,
                            "metadata": {
                                **(metadata or {}),
                                **(doc.metadata or {}),
                                "chunk_index": i,
                                "chunk_count": len(docs),
                            },
                        }
                    )
                return results
            else:
                return self.chunk(text, metadata)
        except Exception as e:
            logger.warning(f"Markdown 分块失败，回退到递归分块: {e}")
            return self.chunk(text, metadata)
