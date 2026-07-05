"""AI Agent 智能办公助手 - 主入口

Usage:
    ai-assistant                    # 启动 CLI 交互式界面
    ai-assistant --model ollama     # 使用指定模型
    ai-assistant --debug            # 调试模式
    ai-assistant web                # 启动 Streamlit Web UI
"""

import asyncio
import sys

# 修复 Windows GBK 编码问题
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import click
from loguru import logger

from .utils.config import config
from .utils.logger import setup_logger


@click.group()
@click.option("--debug", is_flag=True, help="启用调试模式")
@click.option("--model", "-m", default=None, help="模型选择: claude, openai, ollama")
@click.pass_context
def cli(ctx, debug: bool, model: str):
    """🤖 AI Agent 智能办公助手

    统一入口 — 知识问答 | 文档处理 | 消息通知
    """
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    ctx.obj["model"] = model

    setup_logger()
    if debug:
        logger.info("调试模式已启用")


@cli.command()
@click.pass_context
def chat(ctx):
    """启动交互式 CLI 对话"""
    from .interfaces.cli import CLIApp

    app = CLIApp(model_key=ctx.obj.get("model"))
    app.initialize()
    asyncio.run(app.run_loop())


@cli.command()
@click.pass_context
def web(ctx):
    """启动 Streamlit Web UI"""
    import os
    import subprocess

    web_ui_path = __file__.replace("main.py", "interfaces/web.py")
    if not os.path.exists(web_ui_path):
        # 尝试用包路径
        web_ui_path = os.path.join(
            os.path.dirname(__file__), "interfaces", "web.py"
        )

    logger.info(f"启动 Web UI: {web_ui_path}")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", web_ui_path],
        check=True,
    )


@cli.command()
@click.argument("file_path")
@click.option("--task", "-t", default="summarize", help="任务: summarize/extract/translate/analyze")
@click.pass_context
def process(ctx, file_path: str, task: str):
    """处理文档文件"""
    from .interfaces.cli import ModelFactory
    from .agent.graph import AgentGraph
    from .knowledge.pipeline import DocumentPipeline

    async def run():
        model = ModelFactory.create(ctx.obj.get("model"))
        pipeline = DocumentPipeline()
        agent = AgentGraph(model, pipeline)

        result = await agent.run(
            user_input=f"处理文件 {file_path}，任务: {task}",
            uploaded_files=[file_path],
            task_type=task,
        )
        print(result.get("final_response", "无响应"))

    asyncio.run(run())


@cli.command()
@click.argument("query")
@click.pass_context
def ask(ctx, query: str):
    """快速问答（非交互式）"""
    from .interfaces.cli import ModelFactory
    from .agent.graph import AgentGraph
    from .knowledge.pipeline import DocumentPipeline

    async def run():
        model = ModelFactory.create(ctx.obj.get("model"))
        pipeline = DocumentPipeline()
        agent = AgentGraph(model, pipeline)

        result = await agent.run(user_input=query)
        print(result.get("final_response", "无响应"))

    asyncio.run(run())


@cli.command()
@click.pass_context
def version(ctx):
    """显示版本信息"""
    from importlib.metadata import version as pkg_version

    print("🤖 AI Agent 智能办公助手")
    try:
        print(f"   版本: {pkg_version('ai-office-assistant')}")
    except Exception:
        print("   版本: 0.1.0 (开发版)")
    print(f"   Python: {sys.version}")


if __name__ == "__main__":
    cli()
