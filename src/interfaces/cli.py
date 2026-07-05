"""CLI 交互界面 - 基于 Click 和 Rich"""

import asyncio
import sys
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from loguru import logger

from ..agent.graph import AgentGraph
from ..knowledge.pipeline import DocumentPipeline
from ..knowledge.vector_store import VectorStore
from ..models import ClaudeAdapter, OpenAIAdapter, OllamaAdapter
from ..utils.config import config
from ..utils.logger import setup_logger

console = Console()


class ModelFactory:
    """模型工厂 — 根据配置创建模型适配器"""

    @staticmethod
    def create(model_key: str = None):
        """创建模型适配器实例"""
        model_key = model_key or config.get("models", "models.default", "claude")
        model_config = config.get("models", f"models.{model_key}", {})

        if not model_config:
            available = ["claude", "openai", "ollama"]
            raise ValueError(f"未找到模型配置 '{model_key}'，可用模型: {available}")

        provider = model_config.get("provider", "").lower()
        model_name = model_config.get("model", "")

        if provider == "anthropic":
            return ClaudeAdapter(model_name, model_config)
        elif provider == "openai":
            return OpenAIAdapter(model_name, model_config)
        elif provider == "ollama":
            return OllamaAdapter(model_name, model_config)
        else:
            raise ValueError(f"不支持的模型 provider: {provider}")

    @staticmethod
    def list_models() -> list:
        """列出所有可用模型"""
        models_config = config.load("models").get("models", {})
        available = []
        for key, cfg in models_config.items():
            if key != "default" and isinstance(cfg, dict):
                available.append({
                    "key": key,
                    "provider": cfg.get("provider", "?"),
                    "model": cfg.get("model", "?"),
                })
        return available


class CLIApp:
    """CLI 应用程序 — 管理 Agent 生命周期和用户交互"""

    def __init__(self, model_key: str = None):
        self.model_key = model_key
        self.model = None
        self.doc_pipeline = None
        self.agent = None
        self.messages = []

    def initialize(self) -> None:
        """初始化所有组件"""
        with console.status("[bold green]正在初始化 AI 办公助手...[/bold green]"):
            # 初始化模型
            self.model = ModelFactory.create(self.model_key)
            info = self.model.get_model_info()
            console.print(f"✅ 模型: [bold]{info['provider']}[/bold] / {info['model']}")

            # 初始化知识库
            self.doc_pipeline = DocumentPipeline()
            vector_store = self.doc_pipeline.vector_store
            doc_count = vector_store.count()
            console.print(f"✅ 知识库: 已加载 {doc_count} 个文档块")

            # 初始化 Agent
            self.agent = AgentGraph(self.model, self.doc_pipeline)
            console.print("✅ Agent 工作流已就绪")

    async def ask(self, user_input: str) -> str:
        """处理用户输入并返回响应"""
        result = await self.agent.run(
            user_input=user_input,
            messages=self.messages,
        )

        response = result.get("final_response", "（无响应）")
        error = result.get("error")

        if error:
            console.print(f"[red]⚠️ 错误: {error}[/red]")

        # 更新对话历史
        self.messages.append({"role": "user", "content": user_input})
        self.messages.append({"role": "assistant", "content": response})

        # 限制历史长度
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]

        return response

    def print_welcome(self) -> None:
        """打印欢迎信息"""
        console.clear()
        console.print()
        console.print(
            Panel.fit(
                "[bold cyan]🤖 AI Agent 智能办公助手[/bold cyan]\n\n"
                "统一入口 — 知识问答 | 文档处理 | 消息通知\n\n"
                f"[dim]模型: {self.model.provider}/{self.model.model_name}[/dim]",
                border_style="cyan",
            )
        )
        console.print()
        console.print("[dim]命令: /help 帮助 | /model 切换模型 | /kb 知识库 | /clear 清屏 | /exit 退出[/dim]")
        console.print()

    def print_help(self) -> None:
        """打印帮助信息"""
        help_table = Table(title="📋 可用命令", border_style="blue")
        help_table.add_column("命令", style="cyan", width=25)
        help_table.add_column("说明", style="white")

        help_table.add_row("/help", "显示此帮助信息")
        help_table.add_row("/model [name]", "切换模型 (claude / openai / ollama)")
        help_table.add_row("/models", "列出所有可用模型")
        help_table.add_row("/kb add <path>", "添加文件到知识库")
        help_table.add_row("/kb search <query>", "搜索知识库")
        help_table.add_row("/kb list", "列出知识库中的文档")
        help_table.add_row("/kb count", "查看知识库文档数量")
        help_table.add_row("/process <file>", "处理文档（自动解析+总结）")
        help_table.add_row("/summarize <file>", "生成文档摘要")
        help_table.add_row("/search <query>", "网络搜索")
        help_table.add_row("/notify <msg>", "发送通知消息")
        help_table.add_row("/clear", "清空对话历史")
        help_table.add_row("/exit, /quit", "退出程序")
        help_table.add_row("直接输入问题", "进行知识问答或对话")

        console.print(help_table)

    async def handle_command(self, text: str) -> Optional[str]:
        """处理斜杠命令，返回响应；如果不是命令返回 None"""
        parts = text.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit"):
            console.print("[yellow]再见！[/yellow]")
            sys.exit(0)

        elif cmd == "/help":
            self.print_help()
            return ""

        elif cmd == "/models":
            models = ModelFactory.list_models()
            table = Table(title="可用模型")
            table.add_column("Key", style="cyan")
            table.add_column("Provider", style="yellow")
            table.add_column("Model", style="green")
            for m in models:
                current = " ←" if m["key"] == self.model_key else ""
                table.add_row(m["key"] + current, m["provider"], m["model"])
            console.print(table)
            return ""

        elif cmd == "/model":
            if args:
                await self._switch_model(args)
            else:
                console.print(f"当前模型: [cyan]{self.model_key}[/cyan] ({self.model.provider}/{self.model.model_name})")
            return ""

        elif cmd == "/clear":
            self.messages.clear()
            console.print("[green]✅ 对话历史已清空[/green]")
            return ""

        elif cmd == "/kb":
            return await self._handle_kb_command(args)

        elif cmd in ("/process", "/summarize", "/analyze", "/extract", "/translate"):
            return await self._handle_document_command(cmd, args)

        elif cmd == "/search":
            return await self._handle_search_command(args)

        elif cmd == "/notify":
            return await self._handle_notify_command(args)

        return None  # 不是命令，进行普通问答

    async def _switch_model(self, model_key: str) -> None:
        """切换模型"""
        try:
            new_model = ModelFactory.create(model_key)
            self.model = new_model
            self.model_key = model_key
            # 重新创建 Agent（使用新模型）
            self.agent = AgentGraph(self.model, self.doc_pipeline)
            info = new_model.get_model_info()
            console.print(f"[green]✅ 已切换到: {info['provider']}/{info['model']}[/green]")
        except Exception as e:
            console.print(f"[red]❌ 切换失败: {e}[/red]")

    async def _handle_kb_command(self, args: str) -> str:
        """处理知识库命令"""
        parts = args.split(maxsplit=1)
        action = parts[0].lower() if parts else "list"
        param = parts[1] if len(parts) > 1 else ""

        if action == "add":
            if not param:
                return "用法: /kb add <文件或目录路径>"
            with console.status(f"[bold]正在导入: {param}...[/bold]"):
                path = __import__("pathlib").Path(param)
                if path.is_dir():
                    results = self.doc_pipeline.ingest_directory(param)
                    total = sum(len(ids) for ids in results.values())
                    return f"✅ 已从目录导入 {len(results)} 个文件 (共 {total} 个块)"
                else:
                    ids = self.doc_pipeline.ingest_file(param)
                    return f"✅ 已导入: {param} ({len(ids)} 个块)"

        elif action == "search":
            if not param:
                return "用法: /kb search <查询内容>"
            with console.status("[bold]正在搜索...[/bold]"):
                results = self.doc_pipeline.search(param)
            if not results:
                return "未找到相关文档"
            lines = [f"📚 搜索 '{param}' 结果:\n"]
            for i, doc in enumerate(results, 1):
                source = doc.get("metadata", {}).get("filename", "未知")
                score = doc.get("score", 0)
                preview = doc.get("content", "")[:150]
                lines.append(f"{i}. [{source}] (相关度: {score:.2f})\n   {preview}...\n")
            return "\n".join(lines)

        elif action == "list":
            docs = self.doc_pipeline.vector_store.list_all()
            if not docs:
                return "知识库中暂无文档"
            # 按文件名分组
            files = {}
            for doc in docs:
                fname = doc.get("metadata", {}).get("filename", "未知")
                files[fname] = files.get(fname, 0) + 1
            lines = ["📚 知识库文档:\n"]
            for fname, count in files.items():
                lines.append(f"  - {fname} ({count} 块)")
            return "\n".join(lines)

        elif action == "count":
            count = self.doc_pipeline.vector_store.count()
            return f"知识库共有 {count} 个文档块"

        elif action == "clear":
            self.doc_pipeline.vector_store.clear()
            self.doc_pipeline = DocumentPipeline()
            self.agent = AgentGraph(self.model, self.doc_pipeline)
            return "✅ 知识库已清空"

        else:
            return f"未知的知识库操作: {action}。可用: add, search, list, count, clear"

    async def _handle_document_command(self, cmd: str, args: str) -> str:
        """处理文档处理命令"""
        task_map = {
            "/process": "summarize",
            "/summarize": "summarize",
            "/analyze": "analyze",
            "/extract": "extract",
            "/translate": "translate",
        }
        task_type = task_map.get(cmd, "summarize")

        if not args:
            return f"用法: {cmd} <文件路径>"

        with console.status(f"[bold]正在处理文档...[/bold]"):
            result = await self.agent.run(
                user_input=f"请对文件 {args} 进行{task_type}",
                messages=self.messages,
                uploaded_files=[args],
                task_type=task_type,
            )

        response = result.get("final_response", "处理完成")
        self.messages.append({"role": "user", "content": f"{cmd} {args}"})
        self.messages.append({"role": "assistant", "content": response})
        return response

    async def _handle_search_command(self, args: str) -> str:
        """处理网络搜索"""
        if not args:
            return "用法: /search <搜索关键词>"

        from ..mcp.tools.web_search import WebSearchTool

        search_tool = WebSearchTool()
        return await search_tool.execute(args)

    async def _handle_notify_command(self, args: str) -> str:
        """处理通知命令"""
        if not args:
            return "用法: /notify <消息内容> [--platform feishu|wecom|dingtalk]"

        result = await self.agent.run(
            user_input=f"发送通知: {args}",
            messages=self.messages,
            notification_target="wecom",
            notification_type="custom",
        )

        response = result.get("final_response", "通知已处理")
        self.messages.append({"role": "user", "content": f"/notify {args}"})
        self.messages.append({"role": "assistant", "content": response})
        return response

    async def run_loop(self) -> None:
        """主事件循环"""
        self.print_welcome()

        while True:
            try:
                user_input = Prompt.ask("\n[bold green]你[/bold green]").strip()

                if not user_input:
                    continue

                # 处理命令
                cmd_result = await self.handle_command(user_input)
                if cmd_result is not None:
                    if cmd_result:
                        console.print(f"\n[bold cyan]助手[/bold cyan]:")
                        console.print(Markdown(cmd_result))
                    continue

                # 普通对话
                with console.status("[bold]思考中...[/bold]"):
                    response = await self.ask(user_input)

                console.print(f"\n[bold cyan]助手[/bold cyan]:")
                console.print(Markdown(response))

            except KeyboardInterrupt:
                console.print("\n[yellow]再见！[/yellow]")
                break
            except Exception as e:
                logger.error(f"运行错误: {e}")
                console.print(f"[red]❌ 错误: {e}[/red]")


@click.command()
@click.option("--model", "-m", default=None, help="模型选择: claude, openai, ollama")
@click.option("--debug", is_flag=True, help="启用调试模式")
def cli(model: str = None, debug: bool = False):
    """AI Agent 智能办公助手 - CLI 入口"""
    setup_logger()

    if debug:
        config._cache.clear()
        logger.info("调试模式已启用")

    app = CLIApp(model_key=model)
    app.initialize()
    asyncio.run(app.run_loop())


if __name__ == "__main__":
    cli()
