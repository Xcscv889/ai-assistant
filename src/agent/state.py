"""Agent 状态定义"""

from typing import Any, Dict, List, Optional, TypedDict


class ToolCall(TypedDict):
    """工具调用记录"""
    name: str
    args: Dict[str, Any]
    result: Optional[str]


class AgentState(TypedDict, total=False):
    """LangGraph Agent 全局状态"""

    # === 用户输入 ===
    messages: List[Dict[str, str]]          # 对话历史 [{"role": "user/assistant", "content": "..."}]
    user_input: str                          # 当前用户输入

    # === 路由信息 ===
    intent: str                              # 意图分类: qa / document / notify / chat
    confidence: float                        # 意图识别置信度 0.0-1.0

    # === 知识检索 ===
    retrieved_docs: List[Dict[str, Any]]     # 检索到的文档列表
    retrieval_query: str                     # 检索查询
    context: str                             # 拼接后的上下文

    # === 文档处理 ===
    uploaded_files: List[str]                # 文件路径列表
    processed_content: str                   # 处理后的文档内容
    task_type: str                           # 文档任务: summarize / extract / translate / analyze

    # === 消息通知 ===
    notification_target: str                 # 通知目标: feishu / wecom / dingtalk
    notification_content: str                # 通知内容
    notification_type: str                   # meeting_summary / reminder / report / alert / custom

    # === 模型调用 ===
    model_name: str                          # 当前使用的模型名称

    # === 生成结果 ===
    final_response: str                      # 最终回复
    tool_calls: List[ToolCall]               # 工具调用记录
    error: Optional[str]                     # 错误信息

    # === 元数据 ===
    iteration_count: int                     # 当前迭代次数
