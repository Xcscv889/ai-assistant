"""配置加载器 - 支持 YAML 配置文件和 .env 环境变量"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from loguru import logger


class ConfigLoader:
    """配置管理器，加载 YAML 配置和环境变量"""

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            # 默认配置目录在项目根目录的 config/
            config_dir = Path(__file__).parent.parent.parent / "config"
        self.config_dir = Path(config_dir)
        self._cache: Dict[str, Any] = {}

        # 加载 .env 文件
        env_path = self.config_dir.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()

    def load(self, name: str) -> Dict[str, Any]:
        """加载指定的 YAML 配置文件"""
        if name in self._cache:
            return self._cache[name]

        file_path = self.config_dir / f"{name}.yaml"
        if not file_path.exists():
            logger.warning(f"配置文件不存在: {file_path}")
            return {}

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._cache[name] = data
        logger.debug(f"已加载配置: {file_path}")
        return data

    def get(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """通过点号路径获取配置值，如 'settings.agent.max_iterations'"""
        config = self.load(config_name)
        keys = key_path.split(".")
        value = config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value

    def get_env(self, key: str, default: str = "") -> str:
        """获取环境变量值"""
        return os.getenv(key, default)

    def reload(self) -> None:
        """清除缓存，强制重新加载所有配置"""
        self._cache.clear()


# 全局配置实例
config = ConfigLoader()
