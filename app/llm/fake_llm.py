"""Lab03 FakeLLMClient：用预设脚本替代模型决策。

- 按顺序消费预设的 assistant 消息，控制流可重复。
- 每次 ``chat`` 用 ``deepcopy`` 记录当时收到的 messages 与 tools，
  供测试断言“第 N 次请求究竟携带了什么”。
- 不读取上下文、不理解文本；它替代的是模型决策，不是工具执行。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class FakeExhaustedError(Exception):
    """预设响应被消费完仍被调用：测试应断言这不会发生。"""


class FakeLLMClient:
    """脚本化的假 LLM 客户端。

    Args:
        scripted: 依次返回的 assistant 消息列表（每次 chat 消费一个）。
    """

    def __init__(self, scripted: list[dict[str, Any]]) -> None:
        self._scripted = list(scripted)
        # requests[i] = 第 i 次 chat 调用时收到的深拷贝快照
        self.requests: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.requests.append(
            {"messages": deepcopy(messages), "tools": deepcopy(tools)}
        )
        if not self._scripted:
            raise FakeExhaustedError(
                f"Fake 响应已耗尽（第 {len(self.requests)} 次调用）"
            )
        return deepcopy(self._scripted.pop(0))

    @property
    def call_count(self) -> int:
        return len(self.requests)
