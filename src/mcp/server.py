"""MCP Server 定义 - 注册所有工具"""

from typing import List

from loguru import logger

from ..knowledge.pipeline import DocumentPipeline
from ..knowledge.vector_store import VectorStore
from .tools.document_parser import DocumentParserTool
from .tools.file_manager import FileManagerTool
from .tools.knowledge_base import KnowledgeBaseTool
from .tools.notification import NotificationTool
from .tools.web_search import WebSearchTool


class MCPServer:
    """MCP Server — 统一管理所有工具

    通过 MCP 协议将工具暴露给 LangGraph Agent 调用。
    """

    def __init__(
        self,
        document_pipeline: DocumentPipeline = None,
        vector_store: VectorStore = None,
    ):
        self.doc_pipeline = document_pipeline or DocumentPipeline()
        self.vector_store = vector_store or VectorStore()

        # 注册所有工具
        self.tools = {
            "document_parser": DocumentParserTool(),
            "knowledge_base": KnowledgeBaseTool(self.doc_pipeline, self.vector_store),
            "file_manager": FileManagerTool(),
            "web_search": WebSearchTool(),
            "notification": NotificationTool(),
        }

        logger.info(f"MCP Server 已初始化，注册了 {len(self.tools)} 个工具")

    def get_tool(self, name: str):
        """获取指定工具"""
        return self.tools.get(name)

    def list_tools(self) -> List[dict]:
        """列出所有工具及其描述"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self.tools.values()
        ]

    async def call_tool(self, tool_name: str, **kwargs) -> str:
        """调用指定工具

        Args:
            tool_name: 工具名称
            **kwargs: 工具参数

        Returns:
            工具执行结果
        """
        tool = self.get_tool(tool_name)
        if not tool:
            available = ", ".join(self.tools.keys())
            return f"错误: 未找到工具 '{tool_name}'。可用工具: {available}"

        logger.info(f"调用工具: {tool_name}({list(kwargs.keys())})")
        return await tool.execute(**kwargs)

    def get_tools_for_langchain(self) -> list:
        """将工具转换为 LangChain Tool 格式（用于 LangGraph 集成）"""
        from langchain_core.tools import tool

        langchain_tools = []

        for tool_name, tool_instance in self.tools.items():

            @tool(tool_name, description=tool_instance.description)
            async def tool_func(_tool_instance=tool_instance, **kwargs):
                return await _tool_instance.execute(**kwargs)

            langchain_tools.append(tool_func)

        return langchain_tools
