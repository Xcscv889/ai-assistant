"""网络搜索 MCP 工具"""

from typing import Any, Dict

from loguru import logger


class WebSearchTool:
    """网络搜索 MCP 工具"""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "网络搜索工具，使用 DuckDuckGo 搜索引擎搜索互联网信息"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询词",
                },
                "max_results": {
                    "type": "integer",
                    "description": "最大结果数",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, max_results: int = 5) -> str:
        """执行网络搜索"""
        try:
            from duckduckgo_search import DDGS

            logger.debug(f"网络搜索: {query}")

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(
                        {
                            "title": r.get("title", ""),
                            "url": r.get("href", ""),
                            "snippet": r.get("body", ""),
                        }
                    )

            if not results:
                return f"未找到与 '{query}' 相关的搜索结果"

            # 格式化结果
            lines = [f"搜索 '{query}' 的结果:\n"]
            for i, r in enumerate(results, 1):
                lines.append(
                    f"{i}. **{r['title']}**\n"
                    f"   URL: {r['url']}\n"
                    f"   {r['snippet']}\n"
                )

            return "\n".join(lines)

        except ImportError:
            return "错误: 请安装 duckduckgo-search: pip install duckduckgo-search"
        except Exception as e:
            logger.error(f"网络搜索失败: {e}")
            return f"搜索失败: {e}"
