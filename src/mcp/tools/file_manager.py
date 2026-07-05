"""文件管理 MCP 工具"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


class FileManagerTool:
    """文件管理 MCP 工具 — 读写、列表、搜索"""

    @property
    def name(self) -> str:
        return "file_manager"

    @property
    def description(self) -> str:
        return "文件管理工具，支持读取、写入、列出目录、搜索文件"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write", "list", "search", "delete"],
                    "description": "操作类型",
                },
                "path": {
                    "type": "string",
                    "description": "文件或目录路径",
                },
                "content": {
                    "type": "string",
                    "description": "要写入的内容（仅 write 操作）",
                },
                "pattern": {
                    "type": "string",
                    "description": "搜索模式（仅 search 操作），如 '*.pdf'",
                },
            },
            "required": ["action", "path"],
        }

    async def execute(
        self,
        action: str,
        path: str,
        content: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> str:
        """执行文件操作"""
        try:
            if action == "read":
                return await self._read(path)
            elif action == "write":
                return await self._write(path, content or "")
            elif action == "list":
                return self._list(path)
            elif action == "search":
                return self._search(path, pattern or "*")
            elif action == "delete":
                return self._delete(path)
            else:
                return f"错误: 不支持的操作 - {action}"
        except Exception as e:
            logger.error(f"文件操作失败 ({action}): {e}")
            return f"错误: {e}"

    async def _read(self, path: str) -> str:
        file_path = Path(path)
        if not file_path.exists():
            return f"错误: 文件不存在 - {path}"
        if not file_path.is_file():
            return f"错误: 路径不是文件 - {path}"

        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return file_path.read_text(encoding="gbk")
            except Exception as e:
                return f"错误: 无法读取文件 - {e}"

    async def _write(self, path: str, content: str) -> str:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"✅ 已写入文件: {path} ({len(content)} 字符)"

    def _list(self, path: str) -> str:
        dir_path = Path(path)
        if not dir_path.exists():
            return f"错误: 目录不存在 - {path}"
        if not dir_path.is_dir():
            return f"错误: 路径不是目录 - {path}"

        items = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        lines = [f"目录内容: {path}\n"]
        for item in items:
            icon = "📁" if item.is_dir() else self._file_icon(item.suffix)
            if item.is_file():
                size = f"{item.stat().st_size / 1024:.1f}KB"
                lines.append(f"  {icon} {item.name} ({size})")
            else:
                lines.append(f"  {icon} {item.name}/")
        return "\n".join(lines)

    def _search(self, directory: str, pattern: str) -> str:
        dir_path = Path(directory)
        if not dir_path.exists():
            return f"错误: 目录不存在 - {directory}"

        results = sorted(dir_path.rglob(pattern))
        if not results:
            return f"未找到匹配 '{pattern}' 的文件"

        lines = [f"搜索 '{pattern}' 结果 ({len(results)} 个):\n"]
        for item in results[:50]:  # 最多显示50个
            lines.append(f"  {item}")
        if len(results) > 50:
            lines.append(f"  ... 及其他 {len(results) - 50} 个文件")
        return "\n".join(lines)

    def _delete(self, path: str) -> str:
        file_path = Path(path)
        if not file_path.exists():
            return f"错误: 文件不存在 - {path}"

        file_path.unlink()
        return f"✅ 已删除文件: {path}"

    def _file_icon(self, suffix: str) -> str:
        icons = {
            ".pdf": "📕", ".docx": "📘", ".doc": "📘",
            ".pptx": "📊", ".ppt": "📊", ".xlsx": "📗",
            ".md": "📝", ".txt": "📄", ".py": "🐍",
            ".jpg": "🖼️", ".png": "🖼️", ".json": "📋",
        }
        return icons.get(suffix.lower(), "📄")
