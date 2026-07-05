"""知识库 MCP 工具"""

from typing import Any, Dict, List, Optional

from loguru import logger

from ...knowledge.pipeline import DocumentPipeline
from ...knowledge.vector_store import VectorStore


class KnowledgeBaseTool:
    """知识库管理 MCP 工具集"""

    def __init__(
        self,
        document_pipeline: Optional[DocumentPipeline] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.pipeline = document_pipeline or DocumentPipeline()
        self.store = vector_store or VectorStore()

    @property
    def name(self) -> str:
        return "knowledge_base"

    @property
    def description(self) -> str:
        return "知识库管理工具，支持搜索、添加、删除、列出文档"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "add", "delete", "list", "count", "clear"],
                    "description": "操作类型",
                },
                "query": {
                    "type": "string",
                    "description": "搜索查询或文件路径",
                },
                "top_k": {
                    "type": "integer",
                    "description": "搜索结果数量（仅搜索操作）",
                    "default": 5,
                },
                "file_path": {
                    "type": "string",
                    "description": "要添加的文件路径（仅添加操作）",
                },
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要删除的文档 ID（仅删除操作）",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        query: Optional[str] = None,
        top_k: int = 5,
        file_path: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
    ) -> str:
        """执行知识库操作"""
        try:
            if action == "search":
                return await self._search(query or "", top_k)
            elif action == "add":
                return await self._add(file_path)
            elif action == "delete":
                return self._delete(document_ids or [])
            elif action == "list":
                return self._list()
            elif action == "count":
                return self._count()
            elif action == "clear":
                return self._clear()
            else:
                return f"错误: 不支持的操作 - {action}"
        except Exception as e:
            logger.error(f"知识库操作失败 ({action}): {e}")
            return f"错误: {e}"

    async def _search(self, query: str, top_k: int) -> str:
        if not query:
            return "错误: 请提供搜索查询"

        results = self.pipeline.search(query, top_k=top_k)

        if not results:
            return "未找到相关文档。您可以先使用 'add' 操作将文档添加到知识库。"

        lines = [f"搜索 '{query}' 的结果 (共 {len(results)} 条):\n"]
        for i, doc in enumerate(results, 1):
            source = doc.get("metadata", {}).get("filename", "未知")
            score = doc.get("score", 0)
            content_preview = doc.get("content", "")[:200]
            lines.append(
                f"{i}. [{source}] (相关度: {score:.2f})\n"
                f"   {content_preview}...\n"
            )

        return "\n".join(lines)

    async def _add(self, file_path: Optional[str]) -> str:
        if not file_path:
            return "错误: 请提供文件路径"

        ids = self.pipeline.ingest_file(file_path)
        return f"✅ 已添加文件到知识库: {file_path} ({len(ids)} 个块)"

    def _delete(self, doc_ids: List[str]) -> str:
        if not doc_ids:
            return "错误: 请提供要删除的文档 ID"

        self.store.delete(doc_ids)
        return f"✅ 已从知识库删除 {len(doc_ids)} 个文档"

    def _list(self) -> str:
        docs = self.store.list_all()
        if not docs:
            return "知识库中暂无文档。"

        lines = [f"知识库文档列表 (共 {len(docs)} 个):\n"]
        for doc in docs:
            meta = doc.get("metadata", {})
            source = meta.get("filename", "未知")
            file_type = meta.get("file_type", "")
            lines.append(f"- [{doc['id'][:8]}...] {source} ({file_type})")
        return "\n".join(lines)

    def _count(self) -> str:
        count = self.store.count()
        return f"知识库当前有 {count} 个文档块"

    def _clear(self) -> str:
        self.store.clear()
        return "✅ 知识库已清空"
