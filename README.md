# 🤖 AI Agent 智能办公助手

统一入口 — 知识问答 | 文档处理 | 消息通知

## 项目概述

针对个人学习及实验室办公过程中论文阅读、文档整理、知识查询等任务依赖多个工具、操作重复、信息分散的问题，设计并开发的 AI Agent 智能办公助手。

### 核心功能

- **📚 知识问答**：基于本地知识库的智能检索与问答（RAG）
- **📄 文档处理**：PDF/Word/PPT/Excel/Markdown 解析、总结、翻译、分析
- **📨 消息通知**：支持飞书、企业微信、钉钉消息推送
- **🔧 多模型支持**：Claude / OpenAI / Ollama 灵活切换
- **🖥️ 多入口**：CLI 命令行 + Streamlit Web UI

### 技术架构

```
用户交互层 (CLI / Web / Bot) → Agent 编排层 (LangGraph) → MCP 工具层 → 基础设施层
```

- **Agent 框架**: LangGraph (StateGraph + 条件路由)
- **工具集成**: MCP (Model Context Protocol)
- **向量数据库**: ChromaDB
- **Embedding**: sentence-transformers (bge-large-zh-v1.5)
- **文档解析**: PyMuPDF / python-docx / python-pptx / openpyxl

## 快速开始

### 环境要求

- Python >= 3.11
- 至少一个 LLM API Key (Claude / OpenAI) 或本地 Ollama

### 安装

```bash
# 1. 克隆/进入项目目录
cd ai-office-assistant

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. 安装依赖
pip install -e .

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 5. 安装 Embedding 模型（可选，首次运行自动下载）
# BAAI/bge-large-zh-v1.5 约 1.3GB
```

### 使用

```bash
# 启动 CLI 交互式对话
ai-assistant chat

# 指定模型
ai-assistant chat --model ollama

# 启动 Web UI
ai-assistant web

# 快速问答
ai-assistant ask "什么是 RAG？"

# 处理文档
ai-assistant process paper.pdf --task summarize
```

### CLI 命令

| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助信息 |
| `/model <name>` | 切换模型 (claude/openai/ollama) |
| `/models` | 列出可用模型 |
| `/kb add <path>` | 添加文件到知识库 |
| `/kb search <query>` | 搜索知识库 |
| `/kb list` | 列出知识库文档 |
| `/process <file>` | 处理文档 |
| `/summarize <file>` | 生成文档摘要 |
| `/search <query>` | 网络搜索 |
| `/notify <msg>` | 发送通知 |
| `/clear` | 清空对话历史 |
| `/exit` | 退出 |

## 项目结构

```
ai-office-assistant/
├── config/                    # 配置文件
│   ├── settings.yaml          # 全局配置
│   ├── models.yaml            # 模型配置
│   └── prompts/               # Prompt 模板
├── src/
│   ├── agent/                 # LangGraph Agent
│   │   ├── graph.py           # 工作流定义
│   │   ├── state.py           # 状态定义
│   │   └── nodes/             # Agent 节点
│   ├── mcp/                   # MCP 工具
│   │   ├── server.py          # MCP Server
│   │   └── tools/             # 工具实现
│   ├── knowledge/             # 知识库
│   │   ├── vector_store.py    # ChromaDB
│   │   ├── embedding.py       # Embedding
│   │   ├── chunker.py         # 分块
│   │   └── pipeline.py        # 文档摄入
│   ├── models/                # 模型适配器
│   │   ├── claude.py          # Claude
│   │   ├── openai.py          # OpenAI
│   │   └── ollama.py          # Ollama
│   ├── interfaces/            # 用户界面
│   │   ├── cli.py             # CLI
│   │   ├── web.py             # Streamlit
│   │   └── bot/               # 消息机器人
│   └── utils/                 # 工具函数
└── data/                      # 数据目录
    └── knowledge_base/        # 知识库文件
```

## 配置说明

### 模型配置 (`config/models.yaml`)

```yaml
models:
  default: claude  # 默认模型
  claude:
    provider: anthropic
    model: claude-sonnet-5
  openai:
    provider: openai
    model: gpt-4o
  ollama:
    provider: ollama
    model: qwen2.5:14b
```

### 环境变量

| 变量 | 说明 |
|------|------|
| `ANTHROPIC_API_KEY` | Claude API Key |
| `OPENAI_API_KEY` | OpenAI API Key |
| `OLLAMA_BASE_URL` | Ollama 服务地址 |
| `FEISHU_APP_ID` | 飞书应用 ID |
| `FEISHU_APP_SECRET` | 飞书应用密钥 |
| `WECOM_WEBHOOK_URL` | 企业微信 Webhook |
| `DINGTALK_WEBHOOK_URL` | 钉钉 Webhook |
| `TAVILY_API_KEY` | Tavily 搜索 API Key（可选） |

## License

MIT
