"""文件工具函数"""

from pathlib import Path
from typing import List, Optional


# 支持的文件格式
TEXT_EXTENSIONS = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".toml", ".xml", ".csv"}
DOC_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}

ALL_SUPPORTED = TEXT_EXTENSIONS | DOC_EXTENSIONS | IMAGE_EXTENSIONS


def is_supported(path: Path) -> bool:
    """检查文件是否被支持"""
    return path.suffix.lower() in ALL_SUPPORTED


def is_text_file(path: Path) -> bool:
    """检查是否为纯文本文件"""
    return path.suffix.lower() in TEXT_EXTENSIONS


def is_document(path: Path) -> bool:
    """检查是否为办公文档"""
    return path.suffix.lower() in DOC_EXTENSIONS


def is_image(path: Path) -> bool:
    """检查是否为图片"""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def scan_directory(directory: Path, recursive: bool = True) -> List[Path]:
    """扫描目录中的所有支持文件"""
    files: List[Path] = []
    pattern = "**/*" if recursive else "*"
    for file in directory.glob(pattern):
        if file.is_file() and is_supported(file):
            files.append(file)
    return sorted(files)


def get_file_size_mb(path: Path) -> float:
    """获取文件大小（MB）"""
    return path.stat().st_size / (1024 * 1024)


def read_text_file(path: Path) -> Optional[str]:
    """安全读取文本文件"""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="gbk")
        except Exception:
            return None
    except Exception:
        return None
