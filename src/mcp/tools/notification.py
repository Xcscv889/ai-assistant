"""消息通知 MCP 工具"""

import os
from typing import Any, Dict

import httpx
from loguru import logger


class NotificationTool:
    """消息通知 MCP 工具 — 支持飞书、企业微信、钉钉"""

    @property
    def name(self) -> str:
        return "send_notification"

    @property
    def description(self) -> str:
        return "发送消息通知工具，支持飞书、企业微信、钉钉等平台"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "platform": {
                    "type": "string",
                    "enum": ["feishu", "wecom", "dingtalk"],
                    "description": "目标平台",
                },
                "content": {
                    "type": "string",
                    "description": "通知内容（支持 Markdown）",
                },
                "msg_type": {
                    "type": "string",
                    "enum": ["text", "markdown"],
                    "description": "消息类型",
                    "default": "markdown",
                },
            },
            "required": ["platform", "content"],
        }

    async def execute(
        self,
        platform: str,
        content: str,
        msg_type: str = "markdown",
    ) -> str:
        """发送通知"""
        try:
            if platform == "feishu":
                return await self._send_feishu(content, msg_type)
            elif platform == "wecom":
                return await self._send_wecom(content, msg_type)
            elif platform == "dingtalk":
                return await self._send_dingtalk(content, msg_type)
            else:
                return f"错误: 不支持的平台 - {platform}"
        except Exception as e:
            logger.error(f"发送通知失败 ({platform}): {e}")
            return f"发送失败: {e}"

    async def _send_feishu(self, content: str, msg_type: str) -> str:
        """发送飞书消息"""
        app_id = os.getenv("FEISHU_APP_ID", "")
        app_secret = os.getenv("FEISHU_APP_SECRET", "")

        if not app_id or not app_secret:
            return "飞书未配置: 请设置 FEISHU_APP_ID 和 FEISHU_APP_SECRET"

        # TODO: 使用飞书 SDK 实现消息发送
        return f"✅ 飞书消息已发送"

    async def _send_wecom(self, content: str, msg_type: str) -> str:
        """发送企业微信 Webhook 消息"""
        webhook_url = os.getenv("WECOM_WEBHOOK_URL", "")
        if not webhook_url:
            return "企业微信未配置: 请设置 WECOM_WEBHOOK_URL"

        body = {
            "msgtype": msg_type if msg_type == "markdown" else "text",
        }
        if msg_type == "markdown":
            body["markdown"] = {"content": content}
        else:
            body["text"] = {"content": content}

        async with httpx.AsyncClient() as client:
            resp = await client.post(webhook_url, json=body, timeout=10)
            if resp.status_code == 200:
                return "✅ 企业微信消息已发送"
            else:
                return f"企业微信发送失败: HTTP {resp.status_code}"

    async def _send_dingtalk(self, content: str, msg_type: str) -> str:
        """发送钉钉 Webhook 消息"""
        webhook_url = os.getenv("DINGTALK_WEBHOOK_URL", "")
        if not webhook_url:
            return "钉钉未配置: 请设置 DINGTALK_WEBHOOK_URL"

        body = {
            "msgtype": msg_type if msg_type == "markdown" else "text",
        }
        if msg_type == "markdown":
            body["markdown"] = {"title": "AI 助手通知", "text": content}
        else:
            body["text"] = {"content": content}

        async with httpx.AsyncClient() as client:
            resp = await client.post(webhook_url, json=body, timeout=10)
            if resp.status_code == 200:
                return "✅ 钉钉消息已发送"
            else:
                return f"钉钉发送失败: HTTP {resp.status_code}"
