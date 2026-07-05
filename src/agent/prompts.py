"""Prompt 加载器 - 加载和管理 Prompt 模板"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger


class PromptLoader:
    """Prompt 模板管理器"""

    def __init__(self, prompts_dir: Optional[str] = None):
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent / "config" / "prompts"
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, Dict[str, str]] = {}

    def load(self, name: str) -> Dict[str, str]:
        """加载 Prompt 模板"""
        if name in self._cache:
            return self._cache[name]

        file_path = self.prompts_dir / f"{name}.yaml"
        if not file_path.exists():
            logger.warning(f"Prompt 模板不存在: {file_path}")
            return {"system": "", "user": ""}

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._cache[name] = data
        return data

    def get_system(self, name: str) -> str:
        """获取 system prompt"""
        prompt = self.load(name)
        return prompt.get("system", "")

    def get_user(self, name: str) -> str:
        """获取 user prompt 模板"""
        prompt = self.load(name)
        return prompt.get("user", "")

    def render_system(self, name: str, **kwargs) -> str:
        """渲染 system prompt 模板"""
        template = self.get_system(name)
        return template.format(**kwargs) if template else ""

    def render_user(self, name: str, **kwargs) -> str:
        """渲染 user prompt 模板"""
        template = self.get_user(name)
        return template.format(**kwargs) if template else ""


# 全局 Prompt 加载器
prompts = PromptLoader()
