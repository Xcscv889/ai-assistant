"""文档处理节点 - 解析、总结、提取、翻译、分析"""

from pathlib import Path
from typing import Any, Dict

from loguru import logger

from ...models.base import BaseModelAdapter
from ..prompts import prompts
from ..state import AgentState

from ...knowledge.pipeline import DocumentPipeline


class DocumentProcessingNode:
    """文档处理节点

    支持的子任务：
    - summarize: 生成结构化摘要
    - extract: 提取关键信息
    - translate: 翻译文档
    - analyze: 深度分析
    """

    def __init__(
        self,
        model_adapter: BaseModelAdapter,
        document_pipeline: DocumentPipeline,
    ):
        self.model = model_adapter
        self.doc_pipeline = document_pipeline

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """执行文档处理"""
        user_input = state.get("user_input", "")
        uploaded_files = state.get("uploaded_files", [])
        task_type = state.get("task_type", self._detect_task_type(user_input))

        # 尝试从用户输入中提取文件路径
        if not uploaded_files:
            uploaded_files = self._extract_file_paths(user_input)

        if not uploaded_files:
            return {
                "final_response": "请指定要处理的文件路径。例如：\n"
                "`/process 论文.pdf` — 解析并处理文件\n"
                "`/summarize 报告.docx` — 总结文档内容"
            }

        # 处理每个文件
        all_results = []
        for file_path in uploaded_files:
            try:
                result = await self._process_single_file(file_path, task_type)
                all_results.append(result)
            except FileNotFoundError:
                all_results.append(f"❌ 文件不存在: {file_path}")
            except Exception as e:
                logger.error(f"处理文件 {file_path} 失败: {e}")
                all_results.append(f"❌ 处理失败 {file_path}: {e}")

        response = "\n\n".join(all_results)

        return {
            "processed_content": response,
            "final_response": response,
            "task_type": task_type,
        }

    def _detect_task_type(self, text: str) -> str:
        """从用户输入中检测文档处理任务类型"""
        task_keywords = {
            "summarize": ["总结", "摘要", "概括", "summarize", "概述"],
            "extract": ["提取", "抽取", "摘录", "extract", "关键信息"],
            "translate": ["翻译", "translate", "译"],
            "analyze": ["分析", "analyze", "评价", "优缺点", "创新点"],
        }
        for task, keywords in task_keywords.items():
            if any(kw in text.lower() for kw in keywords):
                return task
        return "summarize"  # 默认总结

    async def _process_single_file(self, file_path: str, task_type: str) -> str:
        """处理单个文件"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        logger.info(f"处理文档: {path.name}, 任务: {task_type}")

        # 解析文件内容
        content = self.doc_pipeline._parse_file(path)

        if not content:
            return f"⚠️ 未能从 {path.name} 中提取到内容"

        # 截断过长的内容
        max_chars = 30000
        if len(content) > max_chars:
            logger.info(f"文档内容过长，截取前 {max_chars} 字符")
            content = content[:max_chars] + "\n\n... (内容已截断)"

        # 构建 Prompt
        system_prompt = prompts.render_system(
            "document",
            task_type=task_type,
            document_content=content,
        )

        # 生成结果
        task_labels = {
            "summarize": "总结",
            "extract": "提取",
            "translate": "翻译",
            "analyze": "分析",
        }
        task_label = task_labels.get(task_type, task_type)

        user_message = f"请对文档 `{path.name}` 进行{task_label}。"

        response = await self.model.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )

        return f"## 📄 {path.name} — {task_label}结果\n\n{response}"

    def _extract_file_paths(self, text: str) -> list:
        """从文本中提取文件路径"""
        import re

        paths = []

        # 匹配常见文件路径模式
        patterns = [
            r'([A-Za-z]:[\\/][^\s]*\.\w{2,5})',  # Windows 绝对路径
            r'(/[^\s]*\.\w{2,5})',                # Unix 绝对路径
            r'([./]?[\w-]+/[^\s]*\.\w{2,5})',     # 相对路径
            r'([^\s]+\.(?:pdf|docx?|pptx?|xlsx?|md|txt|csv))',  # 文件名
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match not in paths:
                    paths.append(match)

        return paths
