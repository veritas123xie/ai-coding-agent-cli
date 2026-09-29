"""Lab03 最小 ReAct Agent。

控制循环（伪代码）：

    追加 user 消息
    对每个允许的 iteration（一次 client.chat）：
        请求模型，校验响应，保存 assistant 消息
        无 tool_calls 且 content 非空 -> 返回 completed
        有 tool_calls -> 按顺序逐个 dispatch，追加所有 tool Observation
    循环结束仍未返回 -> 返回 iteration_limit（不额外请求“总结”）

层次划分：一次 run() 是一个用户任务；iteration 是一次模型请求；
工具调用是一次 dispatch。追问再次进入 run()，请求次数从 1 重新
计数，实例消息历史继续保留。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.llm.base import LLMClient
from app.llm.glm_client import ProtocolError
from app.tools.basic_tools import BasicTools

DEFAULT_SYSTEM_PROMPT = (
    "你是课程教学 Agent。你可以使用提供的工具完成文件与命令任务。"
    "工具结果以 JSON 形式返回，ok 为 false 时请修正参数后重试。"
    "完成全部动作后再给出最终回答，不要口头声称已经执行。"
)


@dataclass
class AgentResult:
    """一次 run() 的可区分结果。

    status: ``completed``（收到无调用的有效回答）或
            ``iteration_limit``（达到上限，任务未完成）。
    content: 最终回答文本或停止说明。
    iterations: 本次 run 消耗的模型请求次数。
    """

    status: str
    content: str
    iterations: int


class Agent:
    """组织“模型决策 <-> 工具执行”的最小 ReAct 控制程序。"""

    def __init__(
        self,
        client: LLMClient,
        tools: BasicTools,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_iterations: int = 6,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations 必须 >= 1")
        self.client = client
        self.tools = tools
        self.max_iterations = max_iterations
        self.messages: list[dict[str, Any]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def run(self, user_text: str) -> AgentResult:
        """处理一次用户目标或追问；返回可区分的结果对象。"""
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("user_text 必须是非空字符串")
        self.messages.append({"role": "user", "content": user_text})

        for iteration in range(1, self.max_iterations + 1):
            message = self.client.chat(self.messages, tools=self.tools.schemas())
            self._validate_assistant(message)
            self.messages.append(message)

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ProtocolError("模型返回了既无调用又无文本的空响应")
                return AgentResult("completed", content, iteration)

            for call in tool_calls:
                call_id = call.get("id")
                function = call["function"]
                result = self.tools.dispatch(
                    function["name"], function["arguments"]
                )
                # Observation：真实结果按调用 ID 配对回灌
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        return AgentResult(
            "iteration_limit",
            f"达到迭代上限 {self.max_iterations}，任务未完成，已停止。",
            self.max_iterations,
        )

    @staticmethod
    def _validate_assistant(message: Any) -> None:
        if not isinstance(message, dict):
            raise ProtocolError("client 必须返回 assistant 消息字典")
        if message.get("role") != "assistant":
            raise ProtocolError("响应 role 必须是 assistant")
        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, list):
                raise ProtocolError("tool_calls 必须是列表")
            for call in tool_calls:
                if not isinstance(call, dict) or not call.get("id"):
                    raise ProtocolError("tool_call 缺少 id，无法配对 Observation")
