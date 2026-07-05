"""日志管理模块"""

import sys
from pathlib import Path

from loguru import logger

from .config import config


def setup_logger() -> None:
    """初始化日志配置"""
    logger.remove()  # 移除默认 handler

    log_level = config.get("settings", "app.log_level", "INFO")
    debug = config.get("settings", "app.debug", False)
    if debug:
        log_level = "DEBUG"

    # 控制台输出
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # 文件输出
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    )

    logger.info(f"日志系统已初始化 (level={log_level})")
