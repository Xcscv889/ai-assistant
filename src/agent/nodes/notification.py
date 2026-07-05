"""消息通知节点 - 生成和发送通知消息"""

from typing import Any, Dict

from loguru import logger

from ...models.base import BaseModelAdapter
from ..prompts import prompts
from ..state import AgentState


class NotificationNode:
    """消息通知节点

    支持的平台：飞书、企业微信、钉钉
    支持的类型：会议纪要、提醒、报告、告警、自定义
    """

    # 通知平台适配器（延迟导入）
    PLATFORM_ADAPTERS = {
        "feishu": "lark_oapi",
        "wecom": "wecom",
        "dingtalk": "dingtalk",
    }

    def __init__(self, model_adapter: BaseModelAdapter):
        self.model = model_adapter

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """执行消息通知"""
        user_input = state.get("user_input", "")
        target = state.get("notification_target", "feishu")
        content = state.get("notification_content", user_input)
        notify_type = state.get("notification_type", self._detect_notification_type(user_input))

        # Step 1: 格式化通知内容
        system_prompt = prompts.render_system(
            "notification",
            content=content,
            notification_type=notify_type,
            platform=target,
        )

        try:
            formatted = await self.model.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"请将以下内容格式化为{target}通知消息：\n{content}",
                    },
                ],
                temperature=0.5,
                max_tokens=2048,
            )
        except Exception as e:
            logger.error(f"格式化通知失败: {e}")
            formatted = content

        # Step 2: 发送通知（如果平台已配置）
        send_result = await self._send_notification(target, formatted)

        response_parts = [
            f"📨 **通知消息已生成** (目标: {target}, 类型: {notify_type})",
            "",
            formatted,
        ]

        if send_result:
            response_parts.append(f"\n✅ 已发送到 {target}")
        else:
            response_parts.append(f"\n⚠️ {target} 平台未配置或发送失败，请检查配置")

        response = "\n".join(response_parts)

        return {
            "notification_content": formatted,
            "notification_type": notify_type,
            "notification_target": target,
            "final_response": response,
        }

    def _detect_notification_type(self, text: str) -> str:
        """检测通知类型"""
        type_keywords = {
            "meeting_summary": ["会议纪要", "会议", "meeting"],
            "reminder": ["提醒", "备忘", "remind", "reminder"],
            "report": ["报告", "汇报", "周报", "月报", "report"],
            "alert": ["告警", "警告", "紧急", "alert", "urgent"],
        }
        for ntype, keywords in type_keywords.items():
            if any(kw in text.lower() for kw in keywords):
                return ntype
        return "custom"

    async def _send_notification(self, platform: str, content: str) -> bool:
        """发送通知到指定平台"""
        try:
            if platform == "wecom":
                return await self._send_wecom(content)
            elif platform == "dingtalk":
                return await self._send_dingtalk(content)
            elif platform == "feishu":
                return await self._send_feishu(content)
            else:
                logger.warning(f"不支持的通知平台: {platform}")
                return False
        except Exception as e:
            logger.error(f"发送通知失败 ({platform}): {e}")
            return False

    async def _send_feishu(self, content: str) -> bool:
        """发送飞书消息"""
        try:
            from ...interfaces.bot.feishu import FeishuBot

            bot = FeishuBot()
            return await bot.send_text(content)
        except ImportError:
            logger.warning("未安装飞书 SDK (lark-oapi)，请运行: pip install lark-oapi")
            return False
        except Exception as e:
            logger.error(f"飞书发送失败: {e}")
            return False

    async def _send_wecom(self, content: str) -> bool:
        """发送企业微信 Webhook 消息"""
        import os

        import httpx

        webhook_url = os.getenv("WECOM_WEBHOOK_URL", "")
        if not webhook_url:
            logger.warning("未设置 WECOM_WEBHOOK_URL")
            return False

        async with httpx.AsyncClient() as client:
            response = await client.post(
                webhook_url,
                json={
                    "msgtype": "markdown",
                    "markdown": {"content": content},
                },
                timeout=10,
            )
            return response.status_code == 200

    async def _send_dingtalk(self, content: str) -> bool:
        """发送钉钉 Webhook 消息"""
        import os

        import httpx

        webhook_url = os.getenv("DINGTALK_WEBHOOK_URL", "")
        if not webhook_url:
            logger.warning("未设置 DINGTALK_WEBHOOK_URL")
            return False

        async with httpx.AsyncClient() as client:
            response = await client.post(
                webhook_url,
                json={
                    "msgtype": "markdown",
                    "markdown": {"title": "AI 助手通知", "text": content},
                },
                timeout=10,
            )
            return response.status_code == 200
