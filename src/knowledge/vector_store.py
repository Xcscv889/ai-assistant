"""ChromaDB 向量存储"""

import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from ..utils.config import config


class VectorStore:
    """ChromaDB 向量存储管理"""

    def __init__(self, persist_directory: Optional[str] = None, collection_name: Optional[str] = None):
        self.persist_directory = persist_directory or os.path.abspath(
            config.get("settings", "knowledge_base.persist_directory", "./data/knowledge_base/chroma_db")
        )
        self.collection_name = collection_name or config.get(
            "settings", "knowledge_base.collection_name", "office_docs"
        )

        os.makedirs(self.persist_directory, exist_ok=True)

        self._client = None
        self._collection = None

    @property
    def client(self) -> chromadb.PersistentClient:
        """延迟初始化 ChromaDB 客户端"""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info(f"ChromaDB 已连接: {self.persist_directory}")
        return self._client

    @property
    def collection(self):
        """获取或创建集合"""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "AI Office Assistant 知识库"},
            )
        return self._collection

    def add(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """添加文档到向量存储

        Args:
            documents: 文档文本列表
            embeddings: 对应的向量列表
            metadatas: 元数据列表
            ids: 文档 ID 列表（不提供则自动生成）

        Returns:
            添加的文档 ID 列表
        """
        if not documents:
            return []

        if ids is None:
            import uuid

            ids = [str(uuid.uuid4()) for _ in documents]

        try:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
            logger.debug(f"已添加 {len(documents)} 篇文档到知识库")
            return ids
        except Exception as e:
            logger.error(f"添加到向量存储失败: {e}")
            raise

    def search(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """相似度搜索

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            filter_metadata: 元数据过滤条件

        Returns:
            搜索结果列表
        """
        top_k = top_k or config.get("settings", "knowledge_base.top_k_retrieval", 5)

        where_filter = filter_metadata or {}

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"],
            )

            documents = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    documents.append(
                        {
                            "content": doc,
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                            "score": 1 - results["distances"][0][i] if results["distances"] else 0,
                            "id": results["ids"][0][i] if results["ids"] else "",
                        }
                    )

            return documents
        except Exception as e:
            logger.error(f"知识库搜索失败: {e}")
            return []

    def delete(self, ids: List[str]) -> None:
        """删除指定文档"""
        try:
            self.collection.delete(ids=ids)
            logger.debug(f"已从知识库删除 {len(ids)} 篇文档")
        except Exception as e:
            logger.error(f"删除文档失败: {e}")

    def list_all(self) -> List[Dict[str, Any]]:
        """列出所有文档"""
        try:
            result = self.collection.get(include=["metadatas"])
            documents = []
            if result["ids"]:
                for i, doc_id in enumerate(result["ids"]):
                    documents.append(
                        {
                            "id": doc_id,
                            "metadata": result["metadatas"][i] if result["metadatas"] else {},
                        }
                    )
            return documents
        except Exception as e:
            logger.error(f"列出文档失败: {e}")
            return []

    def count(self) -> int:
        """获取文档总数"""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def clear(self) -> None:
        """清空集合"""
        try:
            self.client.delete_collection(self.collection_name)
            self._collection = None
            logger.info(f"知识库集合 '{self.collection_name}' 已清空")
        except Exception as e:
            logger.error(f"清空集合失败: {e}")
