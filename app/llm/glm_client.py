"""Lab03 GLMClient：手写 HTTP/JSON 的最小 Chat Completions 客户端。

职责边界：
- 只负责把 messages（可选 tools）变成 HTTP 请求，并把响应解析为
  经过校验的完整 assistant 消息（保留 content 与 tool_calls）。
- 不读文件、不执行工具、不维护会话历史（这些属于 tools / Agent）。

支持的完整结束标记仅 ``stop`` 与 ``tool_calls``；其它（如 ``length``）
按协议错误处理。HTTP/网络错误统一封装为 :class:`LLMError`，
响应结构非法统一封装为 :class:`ProtocolError`。
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx


class LLMError(Exception):
    """HTTP / 网络 / 响应解析失败：本讲明确停止，不重试。"""


class ProtocolError(Exception):
    """模型响应不符合协议（缺字段、截断、无法配对等）：停止。"""


API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

VALID_ROLES = ("system", "user", "assistant", "tool")


def validate_message(message: Any) -> None:
    """检查一条消息是否为合法结构（角色已知、content 为字符串或 None）。"""
    if not isinstance(message, dict):
        raise ProtocolError(f"消息必须是字典，得到 {type(message).__name__}")
    role = message.get("role")
    if role not in VALID_ROLES:
        raise ProtocolError(f"未知消息角色: {role!r}")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise ProtocolError("content 必须是字符串或 None")


class GLMClient:
    """最小 GLM Chat Completions 客户端（OpenAI 兼容协议）。

    Args:
        model: 模型标识（如 ``glm-4-flash``），由 ``GLM_MODEL`` 配置。
        api_key: Bearer 认证密钥，仅进入请求头，绝不写入消息。
        base_url: Chat Completions 端点，默认智谱官方地址；也可用
            ``GLM_BASE_URL`` 环境变量覆盖（如课堂改用其他兼容服务）。
        http_client: 可注入的 ``httpx.Client``（测试时用 MockTransport）。
        timeout: HTTP 超时（秒），仅约束单次网络请求。
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        http_client: httpx.Client | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not model:
            raise ValueError("model 不能为空")
        if not api_key:
            raise ValueError("api_key 不能为空")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url or os.environ.get("GLM_BASE_URL") or API_URL
        self.timeout = timeout
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """发送一次请求，返回校验后的完整 assistant 消息。"""
        for message in messages:
            validate_message(message)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools is not None:
            payload["tools"] = tools

        try:
            response = self._client.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"GLM 请求失败: {exc}") from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"GLM HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMError(f"响应不是合法 JSON: {exc}") from exc

        return self._parse_message(body)

    def _parse_message(self, body: Any) -> dict[str, Any]:
        """从响应体提取并校验 assistant 消息。"""
        if not isinstance(body, dict) or not isinstance(
            body.get("choices"), list
        ):
            raise ProtocolError("响应缺少 choices 列表")
        if not body["choices"]:
            raise ProtocolError("choices 为空")
        choice = body["choices"][0]
        if not isinstance(choice, dict):
            raise ProtocolError("choice 必须是对象")
        finish = choice.get("finish_reason")
        if finish not in ("stop", "tool_calls"):
            raise ProtocolError(f"不支持的 finish_reason: {finish!r}")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ProtocolError("choice 缺少 message")
        if message.get("role") != "assistant":
            raise ProtocolError("message.role 必须是 assistant")
        # 只保留协议字段；部分模型附带推理过程等额外字段（如
        # reasoning_content），不回灌进下一次请求的消息历史
        message = {
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
        }

        tool_calls = message["tool_calls"]
        if tool_calls is None:
            message["tool_calls"] = None
            return message
        if not isinstance(tool_calls, list) or not tool_calls:
            raise ProtocolError("tool_calls 必须是非空列表")
        for call in tool_calls:
            if not isinstance(call, dict):
                raise ProtocolError("tool_call 必须是对象")
            if not call.get("id"):
                raise ProtocolError("tool_call 缺少 id")
            if call.get("type") != "function":
                raise ProtocolError("tool_call.type 必须是 function")
            function = call.get("function")
            if not isinstance(function, dict) or not function.get("name"):
                raise ProtocolError("tool_call 缺少 function.name")
            if not isinstance(function.get("arguments"), str):
                raise ProtocolError("function.arguments 必须是 JSON 字符串")
        return message

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "GLMClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
