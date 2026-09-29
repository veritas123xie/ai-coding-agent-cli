"""Lab03 最小 LLM 客户端抽象。

:class:`LLMClient` 是一个 ``Protocol``：Agent 只依赖 ``chat(messages,
tools)`` 这一契约，不关心对方是真实 HTTP 客户端还是 Fake。
"""

from __future__ import annotations

from typing import Any, Protocol


class LLMClient(Protocol):
    """聊天客户端契约：真实 GLMClient 与 FakeLLMClient 都必须满足。"""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """发送一次模型请求，返回完整 assistant 消息。"""
        ...
